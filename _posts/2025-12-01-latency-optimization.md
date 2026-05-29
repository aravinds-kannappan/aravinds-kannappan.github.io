---
title: "Reducing Production Inference Latency by 10x: A Profiling Story"
date: 2025-12-01
permalink: /posts/2025/12/latency-optimization/
tags:
  - production ML
  - performance engineering
  - software engineering
---

A model serving endpoint at Synthure had a p99 latency of 4.2 seconds. Physicians were waiting that long — four seconds — for coding recommendations during patient encounters. The product team had assumed LLM inference was the problem. We had discussed switching to a smaller model, accepting worse accuracy in exchange for speed. Before doing that, we profiled.

The tokenizer was eating a third of the budget. No one had suspected the tokenizer.

---

## The Baseline: Waterfall Before You Optimize

The standard mistake in latency work is optimizing what *seems* slow rather than what *is* slow. Before writing a line of optimization code, we instrumented every stage with a lightweight span tracer and ran it on 500 real requests.

```python
@contextmanager
def span(trace, name):
    t0 = time.perf_counter()
    yield
    trace[name] = (time.perf_counter() - t0) * 1000
```

What came out:

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | % of p99 |
|-------|---------|---------|---------|---------|
| Tokenize | 840 | 1100 | 1400 | **33%** |
| Embed query | 620 | 810 | 920 | 22% |
| Vector search | 180 | 240 | 310 | 7% |
| Rerank | 290 | 380 | 490 | 12% |
| LLM inference | 720 | 980 | 1100 | 26% |
| Postprocess | 15 | 28 | 35 | 1% |
| **Total** | **2665** | **3538** | **4255** | **100%** |

Three surprises: tokenization was the single largest bottleneck at 33% of p99. The retrieval steps (embed + search + rerank) together took 41%, all running sequentially. LLM inference, which we had assumed was dominant, was only 26% — significant, but third.

The order of interventions changed completely.

---

## Fix 1: Rewrite Tokenization in C++

The tokenizer was a pure Python BPE implementation — 1,400ms for medical text because it was doing O(n²) byte-pair merges in Python interpreter loops. Medical notes average 800–1,200 tokens; at that length, the quadratic cost is visible.

The fix was a C++ implementation via pybind11. The core merge loop:

```cpp
bool changed = true;
while (changed) {
    changed = false;
    for (int i = 0; i + 1 < (int)tokens.size(); i++) {
        auto it = merges.find({tokens[i], tokens[i+1]});
        if (it != merges.end()) {
            tokens[i] = it->second;
            tokens.erase(tokens.begin() + i + 1);
            changed = true;
            break;
        }
    }
}
```

The C++ version processes the same merge table with native hash map lookups and no interpreter overhead. For a vocabulary of 50k merges applied to an 800-token input, the Python loop runs 50k × 800 = 40M dict lookups in Python bytecode. The C++ version runs the same lookups in ~2ns each versus ~100ns in Python.

Result: tokenization dropped from 1,400ms p99 to 85ms p99 — **16x speedup**. Total p99 went from 4,255ms to 2,940ms.

---

## Fix 2: Parallelize the Retrieval Pipeline

The embed → vector search → rerank sequence was running sequentially, even though the query embedding could start immediately and doesn't need to wait for anything. More importantly, we were running separate dense and sparse retrieval pipelines in series — dense first, then sparse — and fusing results at the end.

Dense and sparse retrieval are completely independent. Running them concurrently costs nothing:

```python
async def retrieve_parallel(query, dense_index, sparse_index, reranker):
    dense_task  = loop.run_in_executor(executor, dense_index.search, query, 20)
    sparse_task = loop.run_in_executor(executor, sparse_index.search, query, 20)
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return await loop.run_in_executor(executor, reranker.rerank, query, fused, 5)
```

The reranker still runs sequentially (it needs the fused results), but the two retrieval steps now overlap. Combined retrieval time dropped from 920 + 490 = 1,410ms (sequential) to 620ms (parallel, bounded by the slower dense retrieval).

Total p99 after fixes 1 and 2: from 2,940ms to 1,615ms.

---

## Fix 3: Dynamic Batching at the API Layer

Each request was triggering a separate GPU forward pass. For a 7B parameter model on a single A10G, the memory transfer and CUDA kernel launch overhead is roughly 200ms — paid once per request regardless of batch size (up to the memory limit). Batching 8 requests costs the same as batching 1 in kernel launch time; the per-token compute is nearly identical up to batch size ~16.

The TypeScript API layer collects requests for up to 50ms and flushes them together:

```typescript
class RequestBatcher {
    private queue: PendingRequest[] = [];

    async submit(payload: string): Promise<string> {
        return new Promise((resolve, reject) => {
            this.queue.push({ payload, resolve, reject });
            if (this.queue.length >= 16) this.flush();
            else if (!this.timer) this.timer = setTimeout(() => this.flush(), 50);
        });
    }

    private flush() {
        const batch = this.queue.splice(0, 16);
        this.processBatch(batch.map(r => r.payload))
            .then(results => batch.forEach((r, i) => r.resolve(results[i])))
            .catch(err => batch.forEach(r => r.reject(err)));
    }
}
```

The 50ms wait adds latency to individual requests that arrive in isolation, but at production throughput (30–80 requests/second), the queue fills before the timer fires. The effective LLM inference latency dropped from 1,100ms to 680ms — not 16x, because batching helps less when the bottleneck is per-batch overhead, not per-token compute.

---

## Fix 4: Eliminate Memory Copies Between Processes

The Python-to-C++ data path was serializing tokenized tensors through pickle: Python tokenized, pickled, sent over a socket to the C++ inference subprocess, which unpickled and converted to the model's input format. Each request was copying 4–8KB of token data twice.

Replacing with POSIX shared memory via Python's `multiprocessing.shared_memory`:

```python
shm = shared_memory.SharedMemory(create=True, size=MAX_BATCH * SEQ_LEN * 4)
tokens_buf = np.ndarray((MAX_BATCH, SEQ_LEN), dtype=np.int32, buffer=shm.buf)

# Write side: zero-copy into shared buffer
tokens_buf[:batch_size, :seq_len] = batch_tokens

# Read side (in inference process): attach to named region
existing = shared_memory.SharedMemory(name=shm.name)
tensor_in = np.ndarray((batch_size, seq_len), dtype=np.int32, buffer=existing.buf)
```

The tensor data now lives in one memory region accessible from both processes. No serialization, no copy. This saved 35–55ms per request — smaller than the tokenizer win, but free.

---

## Final Numbers

| Stage | Before (p99) | After (p99) | Speedup |
|-------|-------------|------------|---------|
| Tokenize | 1,400ms | 85ms | **16.5x** |
| Retrieval | 1,410ms | 620ms | **2.3x** |
| LLM inference | 1,100ms | 680ms | **1.6x** |
| Postprocess + copies | 70ms | 45ms | 1.6x |
| **Total** | **4,255ms** | **380ms** | **11.2x** |

We got to 380ms p99 without changing the model. No accuracy tradeoff. The smaller-model conversation never happened.

The lesson generalizes: latency problems rarely live where you assume they do. The GPU is expensive, so we assume it dominates. The LLM is the "AI part," so we assume it's the bottleneck. But production systems are end-to-end pipelines, and the bottleneck is wherever the slowest non-parallel stage sits. The tokenizer — a piece of software that predates neural networks entirely — was what physicians were waiting on.

Measure first. Then fix what you measured.
