---
title: "The Mathematics of Attention: Deriving the Transformer from First Principles"
date: 2025-09-01
permalink: /posts/2025/09/transformers-from-scratch/
tags:
  - transformers
  - deep learning
  - mathematics
---

The transformer architecture has become the backbone of modern AI, yet most practitioners use it without understanding why the attention mechanism is designed the way it is. This post derives self-attention, multi-head attention, and positional encoding from first principles — asking not "what does this do?" but "why does this have to be this way?"

## The Core Problem: Variable-Length Sequence Processing

Before transformers, sequence modeling was dominated by RNNs: process one token at a time, maintain a hidden state, pass it forward. This has two fundamental problems. First, it's sequential — you can't parallelize across time steps during training. Second, long-range dependencies decay: information from 100 tokens ago has been compressed through 100 nonlinear transformations before it reaches the current state.

The key insight of the transformer is to abandon sequential processing entirely. Instead, compute a representation of each token as a weighted function of **all** other tokens simultaneously. The question is: what should the weights be?

## Deriving Attention from a Retrieval Analogy

Think of attention as a soft key-value lookup. You have:
- A **query** q: what you're looking for
- A set of **keys** K: labels describing what each stored value contains
- A set of **values** V: the actual content to retrieve

The retrieval weight for value i is proportional to how well query q matches key kᵢ. For a soft, differentiable version, use the softmax of inner products:

```
Attention(q, K, V) = softmax(q Kᵀ / √dₖ) V
```

Why divide by √dₖ? The inner product q·kᵢ has variance proportional to dₖ (the key dimension). Without scaling, high-dimensional vectors produce large dot products, pushing the softmax into saturation — near-zero gradients everywhere. Dividing by √dₖ keeps the dot products in a well-scaled regime regardless of dimension.

In the full matrix form, computing attention for all queries simultaneously:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

where Q ∈ ℝⁿˣᵈ, K ∈ ℝⁿˣᵈ, V ∈ ℝⁿˣᵈ, and n is sequence length.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
```

## Why Multiple Heads?

A single attention head produces a single weighted average of values. This is a rank-1 operation in attention weight space — one "reason" to attend to each position. But natural language has multiple simultaneous relationships between tokens: syntactic (subject-verb agreement), semantic (coreference), positional (nearby context).

Multi-head attention runs h attention functions in parallel on linearly projected subspaces, then concatenates and projects the results:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ
where headᵢ = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)
```

Each head learns its own projection matrices Wᵢ, allowing different heads to specialize in different relationship types. With h=8 heads and model dimension d=512, each head operates in a 64-dimensional subspace.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq, d_k)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.split_heads(self.W_q(Q), B)
        K = self.split_heads(self.W_k(K), B)
        V = self.split_heads(self.W_v(V), B)
        x, _ = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(x)
```

## The Positional Encoding Problem

Self-attention is permutation-equivariant: shuffling the input sequence produces identically shuffled outputs. There is no notion of position baked into the architecture. But position matters — "dog bites man" and "man bites dog" are very different sentences.

The original transformer uses fixed sinusoidal encodings added to token embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Why sinusoidal? Two reasons. First, for any fixed offset k, PE(pos+k) can be expressed as a linear function of PE(pos) — this means the model can learn to attend by relative position through linear operations. Second, the frequencies span many scales, from very fast (i=0) to very slow (i=d/2), giving the model information at multiple positional resolutions.

Modern models (RoPE, ALiBi) use learned or relative positional encodings that generalize better to sequence lengths longer than those seen in training.

## Complexity: The Quadratic Bottleneck

Self-attention computes QKᵀ, a matrix of n² dot products. This is O(n²d) time and O(n²) memory — quadratic in sequence length. For n=512 this is fine; for n=100,000 (long documents, audio) it's prohibitive.

This has motivated a wave of efficient attention variants:
- **Sparse attention** (Longformer, BigBird): attend only to local windows + global tokens — O(n)
- **Linear attention** (Performer): approximate softmax attention with random features — O(n)
- **Flash Attention**: IO-aware exact computation that avoids materializing the full n×n matrix — same complexity, 5-10× faster in practice through careful GPU memory management

## Putting It Together: The Transformer Block

A full transformer block is:

```
x = x + MultiHeadAttention(LayerNorm(x))     # attention sublayer + residual
x = x + FFN(LayerNorm(x))                    # feedforward sublayer + residual
```

The FFN is a two-layer MLP applied position-wise: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ with inner dimension 4d. It provides the "thinking" capacity per token that attention doesn't — attention routes information between tokens, FFN transforms it.

Pre-LayerNorm (normalize before each sublayer, as above) trains more stably than post-LayerNorm and is standard in modern architectures.

The residual connections are critical: they allow gradients to flow directly from output to input across many layers, enabling training of very deep networks. Combined with careful initialization (small weights so early training is dominated by the residual path), this is why transformers can be trained 100+ layers deep.

## What Attention Actually Learns

Interpretability studies have shown that attention heads specialize in systematic ways. Some heads track syntactic dependencies (noun-verb agreement, coreference). Others attend to positional patterns (previous token, delimiter tokens). A few behave like n-gram detectors. The emergent specialization from end-to-end training is one of the more surprising properties of the architecture — no supervision is provided for these roles.

Understanding this helps debug: if a model fails on long-range dependencies, check whether the relevant attention weights are diluted across many positions. If it fails on entity coreference, the corresponding head may not have developed properly. Attention visualization is a useful debugging tool, with the caveat that high attention weight doesn't always imply causal importance.
