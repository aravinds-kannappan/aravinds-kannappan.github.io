---
title: "Context Forge: The Pareto Frontier of Prompt Compression"
excerpt: "An open, reproducible benchmark of context compression — tokens saved vs. quality retained vs. latency — across the tokenizers production apps actually pay for, on real public data."
collection: portfolio
date: 2026-06-15
---

Every token sent to an LLM costs money and latency, and long contexts blow up quadratically. Context Forge answers the question that matters with hard numbers: how much can you cut before the answer breaks, and what does it cost in compute? It runs a fixed task set through several compression strategies and multiple tokenizers, then plots the Pareto frontier — the configurations where you can't get more savings without losing quality or paying more latency.

**Key work:** Four strategies benchmarked — hard head/tail pruning, TF-IDF embedding chunk drop, a KV-cache-eviction proxy, and optional LLMLingua-2 · Token savings counted with the tokenizers a real deployment pays for (GPT-4o `o200k_base`, GPT-4 `cl100k_base`, GPT-2 BPE) · Result: relevance-aware strategies retain ~92% of task signal at ~51% fewer tokens, while naive truncation collapses to 71% (and to 0.0 on RAG QA, where the answer span gets cut) · A learned per-token droppability classifier on character n-gram + positional features reaching ROC-AUC 0.964 · A frameworkless static dashboard that renders whatever the last run produced — no baked-in numbers

**Stack:** Python, tiktoken, HuggingFace, scikit-learn, vanilla-JS dashboard

[GitHub →](https://github.com/aravinds-kannappan/Context-Forge) · [Live demo →](https://context-forge-ten.vercel.app)
