---
title: "Implementing the Transformer in C++ Without ML Libraries: What You Learn From the Metal"
date: 2025-09-01
permalink: /posts/2025/09/transformer-cpp/
tags:
  - transformers
  - C++
  - deep learning
  - mathematics
---

There is a version of understanding a transformer where you can recite the equations and draw the architecture diagram. Then there is a deeper version where you know, concretely, what happens to a float when it enters the attention mechanism, which cache line it lives on, what instruction the CPU uses to multiply it, how many copies of it exist simultaneously in memory. I wanted the second kind of understanding. The only way to get it was to implement a transformer from nothing: no PyTorch, no NumPy, no BLAS wrappers.

What follows is not primarily about the code. It is about what the code forced me to see.

---

## The Mathematics You Think You Know

A transformer block is two sublayers connected by residual paths:

$$X' = \text{LayerNorm}(X + \text{MHA}(X))$$
$$Y = \text{LayerNorm}(X' + \text{FFN}(X'))$$

Multi-head attention runs $h$ attention functions in parallel, each on a projected subspace of dimension $d_k = d_{\text{model}}/h$:

$$\text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V)$$

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

These equations look simple. What they obscure is that $QK^T$ is a matrix multiply of shape $(n \times d_k) \cdot (d_k \times n)$, producing an $n \times n$ score matrix. For $n = 512$ tokens and $d_k = 64$, that is 262,144 floats, per head, per layer, per forward pass. On 8 heads and 12 layers, a single forward pass materializes $\approx 25$ million attention score floats. Before computing anything. That is what the $O(n^2)$ complexity *means*, concretely.

---

## What Building It Forced Me to Learn

**Memory layout is everything.** My first implementation used `vector<vector<float>>`, a 2D array where each row is a separate heap allocation. For row-major access (iterating over a row), this is fine. For column access (as in matrix multiply), it is catastrophic: every column element lives in a different cache line. The fix is a flat `vector<float>` with manual index arithmetic:

```cpp
Matrix matmul(const float* A, const float* B, float* C,
              int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++)      // k-loop in the middle = cache-friendly
            for (int j = 0; j < n; j++)
                C[i*n + j] += A[i*k + l] * B[l*n + j];
}
```

The loop order `i, l, j` keeps `A[i*k + l]` constant in the inner loop (one load, $n$ multiply-adds) and accesses `B[l*n + j]` sequentially (streaming from one cache line). This alone gave a 4x speedup over the jagged-array version.

**Softmax numerical stability is not optional.** The scores $QK^T / \sqrt{d_k}$ can easily reach magnitudes of 10–20 for well-trained models. For float32, $e^{89} = \infty$. The standard fix, subtract the row maximum before exponentiating, works because softmax is scale-invariant: $\text{softmax}(x) = \text{softmax}(x - c)$ for any constant $c$. In a raw C++ implementation there is nothing to save you if you forget this. I forgot, and my first attention forward produced a matrix of NaNs.

**GELU is a one-liner but hides real cost.** The approximation $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ requires a `tanh` call per element. `tanh` is 5–10x more expensive than multiplication. For a feed-forward layer with $d_{\text{ff}} = 2048$ and sequence length 512, that is one million `tanh` calls per forward pass. Production implementations approximate GELU differently (or use SiLU, which is just $x \cdot \sigma(x)$) precisely because of this.

---

## The Benchmark Against PyTorch

After getting the implementation correct, I ran it against PyTorch on CPU, sequence length 64, $d_{\text{model}} = 256$, 8 heads:

| Implementation | p50 (ms) | p99 (ms) | Notes |
|---|---|---|---|
| C++ naive (jagged array) | 51.2 | 58.4 | Separate heap alloc per row |
| C++ flat array | 12.8 | 14.1 | Cache-friendly layout |
| C++ flat + softmax fix | 12.3 | 13.7 | Minor; stability, not speed |
| PyTorch (CPU, MKL) | 0.81 | 0.94 | OpenBLAS + AVX-512 + threading |

The gap between my 12ms and PyTorch's 0.84ms is almost entirely the matrix multiply. PyTorch calls MKL, which uses AVX-512 SIMD to process 16 floats per instruction, multi-threaded across CPU cores, with cache-oblivious tiling that keeps working sets in L2. Implementing that correctly is the job of BLAS libraries and takes years of CPU-specific engineering. When I linked my flat-array matmul against OpenBLAS directly (one function call to `cblas_sgemm`), the gap closed to 1.1ms.

The lesson: the transformer mathematics is not complex. The engineering, the SIMD vectorization, the cache tiling, the instruction-level parallelism, is what separates PyTorch's performance from mine. Understanding that gap is what the exercise was for.

---

## What Lives in the Residual Stream

One thing you only see clearly when you implement it yourself: the residual connections make every layer an *additive update* to a shared vector. The input $X$ flows through the entire network unchanged at the top level; each attention and FFN block adds a correction term. This is not an architectural detail, it is the reason residual networks train stably. The gradient flows directly from the loss to the input through the residual path, bypassing the non-linearities. A vanishing gradient problem in one layer does not cut off gradient flow to earlier layers.

In the C++ implementation you can inspect $\|X' - X\|_2$ after each sublayer and watch the residual magnitudes during training. In the early epochs the corrections are large, the network is making big adjustments. As training converges, the corrections shrink, and the residual stream begins to look like a smooth interpolation toward the final answer. This is visible in the code in a way it simply is not in PyTorch, where the residual connections are implicit in the `forward` method.

---

## The FlashAttention Insight, From the Inside

When I ran a sequence of length 1024 and profiled memory allocations, the attention score matrix was the dominant allocation: 1024×1024 floats per head = 4MB, across 8 heads = 32MB, just for the score matrices. None of this fits in L2 cache (typically 256KB–1MB per core). Every access to the score matrix in the softmax and the final $V$ multiply is an L3 or RAM fetch.

FlashAttention's key insight, tiling the attention computation so that the score matrix never fully materializes in memory, instead computing block-by-block and keeping running statistics for the softmax, made immediate intuitive sense once I had held the 32MB score matrix in my hands (so to speak) and watched the cache miss rate spike. The optimization is not about FLOPS; it is about not writing those 32MB to RAM in the first place.

This is what building from the metal gives you: not just the ability to implement FlashAttention, but the felt understanding of *why it matters*.
