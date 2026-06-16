---
title: "How Much of a Prompt Can You Delete Before the Answer Breaks?"
date: 2026-06-15
permalink: /posts/2026/06/prompt-compression/
tags:
  - llms
  - efficiency
  - open source
---

Every token you send to an LLM costs money and latency, and long contexts blow up quadratically with attention. Context compression trims the prompt before inference — but the question that actually matters is the tradeoff: how much can you cut before the answer breaks, and what does it cost you in compute? I built [Context Forge](https://github.com/aravinds-kannappan/Context-Forge) to answer that with hard numbers instead of intuition.

---

## Why a single "compression ratio" is the wrong unit

People talk about compression as if it were one number. It is really three axes that trade against each other: **tokens saved**, **quality retained**, and **latency**. A strategy that deletes 60% of the prompt is worthless if it deletes the sentence that answers the question, and a strategy that retains 99% of quality is worthless if it takes longer than just sending the full prompt. The honest object is the **Pareto frontier**: the set of configurations where you can't get more savings without losing quality or paying more latency.

---

## The strategies

Context Forge benchmarks four families, from a near-zero-latency baseline to a relevance-aware compressor:

- **hard_prompt_pruning** — deterministic head/tail token-budget truncation. The dumb baseline.
- **embedding_chunk_drop** — split into chunks, score each by TF-IDF cosine similarity to the task query, drop the least relevant.
- **kv_cache_eviction** — keep the prompt prefix, the recent suffix, and the salient middle (a text-level proxy for cache eviction).
- **llmlingua** — an optional wrapper around Microsoft's LLMLingua-2.

---

## The result that matters

From the bundled run of 1,620 measurements across three tokenizers and five ratios:

| Strategy | Quality retained | Tokens saved | Latency (median) |
|---|---:|---:|---:|
| kv_cache_eviction | 0.919 | 52.2% | 9.3 ms |
| embedding_chunk_drop | 0.914 | 50.7% | 10.2 ms |
| hard_prompt_pruning | 0.709 | 49.1% | 2.7 ms |

The takeaway is sharp: relevance-aware strategies retain about **92% of the task signal at roughly 51% fewer tokens**, while naive head/tail truncation collapses to **71%** for the same savings — because it routinely cuts the very span that answers the question. On RAG question-answering specifically, truncation drops quality to **0.0** at high compression (the answer span is simply gone), whereas chunk-relevance dropping holds at **1.0**. The 6 ms of extra latency buys you a 21-point quality swing. That is the whole argument for not truncating.

---

## Count tokens with the tokenizer you actually pay for

A subtle trap: token savings are only meaningful against the tokenizer a real deployment bills on. A GPT-2 BPE count is not a GPT-4o count. Context Forge measures all three:

| Model | Backend | Encoding |
|---|---|---|
| `gpt-4o` | tiktoken | `o200k_base` |
| `gpt-4` | tiktoken | `cl100k_base` |
| `gpt-2` | HuggingFace | BPE reference |

The same compressed text can save a different fraction of tokens under each, which is exactly why the savings axis has to be computed per tokenizer rather than with a stand-in.

---

## A learned token-droppability classifier

The most interesting piece is the seed of an actual learned compressor: a small model that predicts, *per token*, whether it can be dropped. It trains on weak labels — tokens overlapping the target are "keep," the rest "droppable" — using character n-gram and positional features.

| ROC-AUC | F1 (droppable) | Accuracy | Precision / Recall |
|---:|---:|---:|---:|
| 0.964 | 0.917 | 88.1% | 0.98 / 0.86 |

What it learns is intuitive: morphological fragments and inflections (`-ating`, `-ated`, `-ative`, `izer`) are droppable, while salient content nouns (`healthcare`, `claims`, `agency`, `relationships`) are kept. The model rediscovers, from weak supervision alone, that function-word morphology carries less task signal than content words.

---

## The methodology caveat I won't hide

"Quality retained" here is a preservation proxy — answer-span recall for QA, reference-keyword coverage for summarization, tail-trace coverage for agents — not a model-graded answer score. That is deliberate: the benchmark runs end-to-end with no paid inference, so anyone can reproduce it for free, and every dataset id, split, tokenizer, and ratio lives in one manifest. The report contains no baked-in numbers; the static dashboard renders whatever the last run produced. An LLM judge can be slotted in without changing the format — but the cheap proxy is already enough to show that *what* you delete matters far more than *how much*.
