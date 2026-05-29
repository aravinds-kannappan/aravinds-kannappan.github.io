---
title: "Reducing Production Inference Latency by 10x: A Profiling Story"
date: 2025-12-01
permalink: /posts/2025/12/latency-optimization/
tags:
  - production ML
  - performance engineering
  - software engineering
---

A model serving endpoint at Synthure had a p99 latency of 4.2 seconds. Physicians were waiting that long for coding recommendations during patient encounters. After three weeks of profiling and systematic optimization, we got it to 380ms. This post is the exact playbook: what we measured, what we found, and what we changed.

The four problems, in order of impact:
1. **Synchronous tokenization** in Python blocking the event loop
2. **No request batching** — each request triggered a separate model forward pass
3. **Sequential retrieval** — RAG pipeline steps running in series when they could be parallel
4. **Memory copies** between Python and the inference runtime

---

## Step 1: Establish a Latency Baseline

Before optimizing anything, measure everything. A single "average latency" number is useless for diagnosis — you need the full distribution and a waterfall breakdown of each stage.

```python
# latency_profiler.py
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, Any
from contextlib import contextmanager
import json

@dataclass
class LatencyTrace:
    request_id: str
    spans: dict = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    
    @contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans[name] = (time.perf_counter() - t0) * 1000  # ms
    
    def total_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000
    
    def report(self) -> dict:
        total = self.total_ms()
        return {
            'total_ms': total,
            'spans': self.spans,
            'unaccounted_ms': total - sum(self.spans.values()),
        }

class LatencyProfiler:
    def __init__(self, window: int = 1000):
        self.window = window
        self.samples: list[dict] = []
    
    def record(self, trace: LatencyTrace):
        self.samples.append(trace.report())
        if len(self.samples) > self.window:
            self.samples.pop(0)
    
    def percentiles(self, stage: str = 'total_ms') -> dict:
        vals = sorted(s[stage] if stage == 'total_ms' 
                      else s['spans'].get(stage, 0) 
                      for s in self.samples)
        if not vals:
            return {}
        n = len(vals)
        return {
            'p50': vals[int(n*0.50)],
            'p95': vals[int(n*0.95)],
            'p99': vals[int(n*0.99)],
            'mean': statistics.mean(vals),
        }
    
    def bottleneck_report(self) -> list[dict]:
        """Rank stages by their contribution to p99 latency."""
        if not self.samples:
            return []
        all_stages = set()
        for s in self.samples:
            all_stages.update(s['spans'].keys())
        
        results = []
        total_p99 = self.percentiles()['p99']
        for stage in all_stages:
            p = self.percentiles(stage)
            results.append({
                'stage': stage,
                'p99_ms': p['p99'],
                'pct_of_total': p['p99'] / total_p99 * 100,
                **p,
            })
        return sorted(results, key=lambda x: x['p99_ms'], reverse=True)
```

Running this on 500 real requests showed:

```
Stage                    p50    p95    p99    % of total
─────────────────────────────────────────────────────────
tokenize                 840    1100   1400    33.3%
retrieval_embed_query    620     810    920    21.9%
vector_search            180     240    310     7.4%
retrieval_rerank         290     380    490    11.7%
llm_inference            720     980   1100    26.2%
postprocess               15      28     35     0.8%
─────────────────────────────────────────────────────────
TOTAL                   2665   3538   4255   100.0%
```

Three surprises: tokenization was the biggest single bottleneck at 33% of p99. Retrieval steps were sequential and together took 41%. And LLM inference, which we expected to dominate, was only 26%.

---

## Fix 1: Fast Tokenization in C++

The tokenizer was a pure Python BPE implementation — 1400ms for medical text because it was doing O(n²) byte-pair merges in Python interpreter loops. The fix: rewrite the merge pass in C++.

```cpp
// fast_tokenizer.cpp
// Byte-Pair Encoding tokenizer in C++ with Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>

namespace py = pybind11;

using Pair = std::pair<int,int>;

struct PairHash {
    size_t operator()(const Pair& p) const {
        return std::hash<long long>()((long long)p.first << 32 | p.second);
    }
};

class FastBPETokenizer {
    std::unordered_map<Pair, int, PairHash> merges;    // pair → merged token id
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    
public:
    FastBPETokenizer(
        const std::vector<std::pair<std::pair<int,int>, int>>& merge_list,
        const std::unordered_map<std::string, int>& vocab_
    ) : vocab(vocab_) {
        for (auto& [pair, token_id] : merge_list)
            merges[pair] = token_id;
        id_to_token.resize(vocab_.size());
        for (auto& [tok, id] : vocab_)
            id_to_token[id] = tok;
    }
    
    std::vector<int> encode(const std::string& text) {
        // Initialize with character-level tokens
        std::vector<int> tokens;
        tokens.reserve(text.size());
        for (unsigned char c : text) {
            std::string ch(1, (char)c);
            auto it = vocab.find(ch);
            tokens.push_back(it != vocab.end() ? it->second : vocab.at("<unk>"));
        }
        
        // Apply BPE merges
        bool changed = true;
        while (changed) {
            changed = false;
            // Find the highest-priority merge in the current token sequence
            int best_pos = -1;
            int best_token = -1;
            
            // In production: use a priority queue indexed by merge rank
            // Here: linear scan for clarity
            for (int i = 0; i+1 < (int)tokens.size(); i++) {
                Pair p = {tokens[i], tokens[i+1]};
                auto it = merges.find(p);
                if (it != merges.end()) {
                    // Apply this merge immediately (greedy left-to-right)
                    best_pos = i;
                    best_token = it->second;
                    break;
                }
            }
            
            if (best_pos >= 0) {
                tokens[best_pos] = best_token;
                tokens.erase(tokens.begin() + best_pos + 1);
                changed = true;
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            if (id < (int)id_to_token.size())
                result += id_to_token[id];
        }
        return result;
    }
    
    std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<int>> results(texts.size());
        // Note: in production, use OpenMP or a thread pool here
        for (int i = 0; i < (int)texts.size(); i++)
            results[i] = encode(texts[i]);
        return results;
    }
};

PYBIND11_MODULE(fast_tokenizer, m) {
    py::class_<FastBPETokenizer>(m, "FastBPETokenizer")
        .def(py::init<
             const std::vector<std::pair<std::pair<int,int>,int>>&,
             const std::unordered_map<std::string,int>&>())
        .def("encode",       &FastBPETokenizer::encode)
        .def("decode",       &FastBPETokenizer::decode)
        .def("encode_batch", &FastBPETokenizer::encode_batch);
}
```

Result: tokenization dropped from 1400ms p99 to 85ms p99 — 16x speedup. Total p99 went from 4255ms to 2940ms.

---

## Fix 2: Async Request Batching

Each request was triggering a synchronous LLM forward pass. For a model on GPU, the per-request overhead (memory transfer, kernel launch) is fixed regardless of batch size up to a point. Batching 8 requests costs nearly the same GPU time as 1.

The TypeScript API layer batches incoming requests and flushes to the Python backend every 50ms:

```typescript
// request-batcher.ts
interface BatchedRequest<T, R> {
    payload:  T;
    resolve:  (result: R) => void;
    reject:   (error: Error) => void;
    enqueued: number;
}

export class RequestBatcher<T, R> {
    private queue:     BatchedRequest<T, R>[] = [];
    private flushTimer: NodeJS.Timeout | null = null;
    
    constructor(
        private readonly processBatch: (payloads: T[]) => Promise<R[]>,
        private readonly maxBatchSize: number = 16,
        private readonly maxWaitMs:    number = 50,
    ) {}
    
    async submit(payload: T): Promise<R> {
        return new Promise((resolve, reject) => {
            this.queue.push({ payload, resolve, reject, enqueued: Date.now() });
            
            if (this.queue.length >= this.maxBatchSize) {
                this.flush();
            } else if (!this.flushTimer) {
                this.flushTimer = setTimeout(() => this.flush(), this.maxWaitMs);
            }
        });
    }
    
    private async flush(): Promise<void> {
        if (this.flushTimer) {
            clearTimeout(this.flushTimer);
            this.flushTimer = null;
        }
        if (this.queue.length === 0) return;
        
        // Drain the queue
        const batch = this.queue.splice(0, this.maxBatchSize);
        
        try {
            const results = await this.processBatch(batch.map(r => r.payload));
            batch.forEach((req, i) => req.resolve(results[i]));
        } catch (err) {
            batch.forEach(req => req.reject(err instanceof Error ? err : new Error(String(err))));
        }
        
        // If more requests arrived during processing, flush again
        if (this.queue.length > 0) {
            setImmediate(() => this.flush());
        }
    }
}

// Usage in the API
const batcher = new RequestBatcher<string, string>(
    async (texts: string[]) => {
        const response = await fetch('http://localhost:8000/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts }),
        });
        return response.json();
    },
    /* maxBatchSize */ 16,
    /* maxWaitMs */    50,
);
```

---

## Fix 3: Parallelizing the RAG Pipeline in Python

The retrieval pipeline was sequential: embed query → vector search → rerank → LLM inference. The first three steps are independent of each other in some configurations, and the embedding + initial retrieval can run in parallel with preprocessing.

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def retrieve_parallel(
    query: str,
    dense_index,
    sparse_index,
    reranker,
    top_k: int = 20,
    top_n: int = 5,
) -> list[str]:
    loop = asyncio.get_event_loop()
    
    # Run dense and sparse retrieval in parallel (both are CPU/IO bound)
    dense_task  = loop.run_in_executor(executor, dense_index.search,  query, top_k)
    sparse_task = loop.run_in_executor(executor, sparse_index.search, query, top_k)
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)[:top_k]
    
    # Reranking (sequential — depends on fused results)
    reranked = await loop.run_in_executor(
        executor, reranker.rerank, query, [r['text'] for r in fused], top_n
    )
    
    return reranked


async def handle_request(query: str, llm_client) -> str:
    # Start retrieval and query preprocessing in parallel
    retrieval_task   = retrieve_parallel(query, dense_idx, sparse_idx, reranker)
    preprocess_task  = asyncio.create_task(preprocess_query(query))
    
    retrieved_docs, processed_query = await asyncio.gather(
        retrieval_task, preprocess_task
    )
    
    # LLM inference (sequential — depends on both)
    context = "\n\n".join(retrieved_docs)
    response = await llm_client.generate(
        prompt=build_prompt(processed_query, context)
    )
    return response
```

---

## Fix 4: Eliminating Memory Copies

The Python-to-C++ data path was copying tokenized tensors through pickle serialization. Replacing with shared memory via `multiprocessing.shared_memory`:

```python
from multiprocessing import shared_memory
import numpy as np

class ZeroCopyTensorBus:
    """Share numpy arrays between processes without copying via shared memory."""
    
    def __init__(self, max_batch: int, seq_len: int, dtype=np.int32):
        self.shape = (max_batch, seq_len)
        self.nbytes = int(np.prod(self.shape)) * np.dtype(dtype).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.array = np.ndarray(self.shape, dtype=dtype, buffer=self.shm.buf)
        self.dtype = dtype
    
    def write(self, tokens: np.ndarray) -> str:
        """Write tokenized batch into shared memory. Returns shm name."""
        n = min(tokens.shape[0], self.shape[0])
        seq = min(tokens.shape[1], self.shape[1])
        self.array[:n, :seq] = tokens[:n, :seq]
        return self.shm.name
    
    def read_in_inference_process(self, name: str, shape: tuple) -> np.ndarray:
        """Attach to existing shared memory in the inference subprocess."""
        existing_shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=self.dtype, buffer=existing_shm.buf)
        return arr.copy()  # copy once, then release shm
    
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()
```

---

## Final Results

```
Stage                  Before (p99)   After (p99)   Speedup
───────────────────────────────────────────────────────────
tokenize               1400ms          85ms          16.5x
retrieval (parallel)    920+490ms      310ms          4.2x (combined)
llm_inference          1100ms         680ms           1.6x (batching)
postprocess              35ms          35ms           1.0x
───────────────────────────────────────────────────────────
TOTAL                  4255ms         380ms          11.2x
```

The most important lesson: measure before you optimize. We initially assumed LLM inference was the bottleneck — everyone does. It was third. The tokenizer, which no one thought about, was eating a third of the budget. Systematic profiling changed the order of attack entirely.
