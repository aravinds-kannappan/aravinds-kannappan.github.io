---
title: "How Much of a Prompt Can You Delete Before the Answer Breaks?"
date: 2026-06-15
permalink: /posts/2026/06/prompt-compression/
tags:
  - llms
  - efficiency
  - open source
---

Every token you send to a language model costs you twice. It costs money, because providers bill per token, and it costs latency, because attention is quadratic in sequence length, so doubling the context can quadruple the work the model does to read it. For a single short prompt none of this matters. For a production application that stuffs retrieved documents, conversation history, tool outputs, and system instructions into every call, it matters enormously, and it is the difference between an app that is cheap and fast and one that is neither.

The obvious response is context compression: trim the prompt before inference so the model reads fewer tokens. The non-obvious part, the part that actually decides whether compression is a good idea, is the tradeoff. How much can you cut before the answer breaks, and what does the cutting cost you in compute? I built [Context Forge](https://github.com/aravinds-kannappan/Context-Forge) to answer that question with hard numbers instead of intuition, on real public data, with the tokenizers production apps actually pay for.

---

## Why a single compression ratio is the wrong unit

People talk about compression as if it were one number, a ratio: this method cuts the prompt by half, that one by two thirds. That framing hides the only thing that matters, which is what you give up to get the savings. Compression is really a point in a three-dimensional space, and the three axes pull against each other.

The first axis is **tokens saved**, the thing you are buying. The second is **quality retained**, how much of the task signal survives the cut. The third is **latency**, how long the compression itself takes, because a compressor that is slower than the inference it saves is worse than useless. A strategy that deletes 60 percent of the prompt is worthless if it deletes the one sentence that answers the question. A strategy that retains 99 percent of quality is worthless if running it takes longer than just sending the full prompt would have. Any honest evaluation has to hold all three axes at once.

---

## The three axes, formally

Make the tradeoff precise. A configuration $c$, meaning a strategy at a particular compression ratio under a particular tokenizer, produces a triple

$$\big(t(c),\, q(c),\, \ell(c)\big) \in [0,1] \times [0,1] \times \mathbb{R}_{\ge 0},$$

the fraction of tokens saved, the quality retained, and the latency. Define a dominance preorder on configurations: $c \preceq c'$ when

$$t(c') \ge t(c), \qquad q(c') \ge q(c), \qquad \ell(c') \le \ell(c),$$

with at least one inequality strict. In words, $c'$ is at least as good on every axis and strictly better on one. The **Pareto frontier** is the set of $\preceq$-maximal configurations, the ones that no other configuration dominates,

$$\mathcal{P} = \{\, c : \nexists\, c' \ \text{with}\ c \prec c' \,\}.$$

This is an antichain: within it, every pair of configurations is incomparable, so each represents a real, non-dominated choice where buying more of one axis costs you another. Everything off the frontier is dominated, which means there exists a strictly better option on all three axes at once, so choosing it is simply a mistake. Reporting $\mathcal{P}$ rather than a single ratio is reporting the actual menu of sensible options instead of one arbitrary dish.

---

## What gets measured, and on what

Context Forge runs three task families, all drawn from real public datasets, because the right amount you can cut depends entirely on what the prompt is for.

The first is retrieval-augmented question answering, built on SQuAD. Here the signal that must be preserved is the answer span: the specific stretch of text that contains the answer. Cut that, and quality goes to zero no matter how fluent the rest of the prompt remains. The second is agent traces, built on a public reasoning-trace dataset, where the signal is the tail of the trace and the eventual outcome, because what an agent concluded matters more than the middle of its deliberation. The third is long-context summarization, built on GovReport, where the signal is coverage of the reference keywords, because a summary that drops the key entities and findings has failed even if it reads smoothly.

Measuring across three task families with three different notions of "signal preserved" is deliberate. A compression method that looks great on summarization, where losing a sentence costs you a little coverage, can be catastrophic on retrieval QA, where losing the wrong sentence costs you the entire answer. A benchmark that tested only one family would mislead you about the others.

---

## The strategies

Context Forge benchmarks four families of compressor, spanning from a near-zero-latency baseline to relevance-aware methods that think about what the prompt is for.

**Hard prompt pruning** is the dumb baseline, and it is dumb on purpose. It truncates to a token budget by keeping the head and the tail and dropping the middle. It is deterministic, it has essentially no latency, and it knows nothing about the content. It exists so every other method has something to beat, and because a surprising number of real systems do exactly this when they hit a context limit.

**Embedding chunk drop** is the first relevance-aware method. It splits the prompt into chunks, scores each chunk by TF-IDF cosine similarity against the task query, and drops the least relevant chunks first. The intuition is simple: if a chunk has little lexical overlap with what is being asked, it probably is not carrying the answer, so it is the safest thing to cut. This is a cheap, classical relevance signal, no embeddings model required, and it turns out to be remarkably effective.

**KV-cache eviction** is a text-level proxy for a technique usually applied inside the attention cache. It keeps the prompt prefix, keeps the recent suffix, and keeps the salient middle, evicting the rest. The reasoning is that the beginning often carries instructions, the end often carries the immediate question and recent context, and the salient middle carries the facts, so those three regions are where the signal concentrates.

**LLMLingua** is an optional wrapper around Microsoft's LLMLingua-2, a learned compression model. It is the heavyweight option, it pulls in an extra dependency and downloads a model, and it is included so the benchmark can place a dedicated learned compressor on the same frontier as the lightweight methods rather than treating it as a separate world.

---

## Compression through the information bottleneck

There is a clean theoretical reason that *what* you cut dominates *how much*, and it comes from information theory. Model the prompt as a random variable $X$, the desired answer as $Y$, and a compressor as a possibly stochastic map $\phi$ producing $\tilde{X} = \phi(X)$. Because $\tilde{X}$ is computed from $X$ alone, with no independent access to $Y$, the three variables form a Markov chain

$$Y \rightarrow X \rightarrow \tilde{X},$$

and the data-processing inequality applies:

$$I(\tilde{X};\, Y) \le I(X;\, Y).$$

Compression can never *increase* the information the prompt carries about the answer. The best any compressor can do is preserve it. The whole game is to throw away tokens while keeping $I(\tilde{X}; Y)$ as close to $I(X; Y)$ as possible. That is precisely the information-bottleneck objective: minimize the retained description length $I(\tilde{X}; X)$, which measures how much of the raw prompt you are still paying to carry, while maximizing the relevant information $I(\tilde{X}; Y)$, trading the two off through a multiplier $\beta$,

$$\mathcal{L}(\phi) = I(\tilde{X};\, X) - \beta\, I(\tilde{X};\, Y).$$

The relevance-aware strategies are crude, cheap approximations to the second term. TF-IDF chunk scoring is a surrogate for $I(\text{chunk};\, Y)$, keeping the chunks that carry the most answer-relevant information and discarding the rest.

This framing predicts the truncation-to-zero result exactly. In grounded QA the answer span $s \subseteq X$ is close to a sufficient statistic for $Y$, meaning $I(s;\, Y) \approx I(X;\, Y)$ and, conditioned on $s$, the rest of the prompt tells you nothing more, $Y \perp X \mid s$. Hard truncation deletes a fixed region regardless of content. When that region contains $s$, the surviving prompt becomes independent of the answer, so

$$I(\tilde{X};\, Y) = 0,$$

and quality collapses to its floor: the model is now being asked a question whose answer is no longer present. Relevance-aware dropping scores $s$ as high-information and keeps it, holding $I(\tilde{X};\, Y) \approx I(X;\, Y)$. Same token budget, opposite outcome, and the entire difference is in *which* tokens survive.

---

## The result that matters

From the bundled run of 1,620 measurements across three tokenizers and five compression ratios, here is the headline comparison:

| Strategy | Quality retained | Tokens saved | Latency (median) |
|---|---:|---:|---:|
| kv_cache_eviction | 0.919 | 52.2% | 9.3 ms |
| embedding_chunk_drop | 0.914 | 50.7% | 10.2 ms |
| hard_prompt_pruning | 0.709 | 49.1% | 2.7 ms |

The story is sharp. The two relevance-aware strategies retain about 92 percent of the task signal while cutting roughly 51 percent of the tokens. Naive head-and-tail truncation, for essentially the same token savings, collapses to 71 percent quality. The relevance-aware methods cost about 6 to 7 extra milliseconds of latency, and in exchange they buy a 21-point swing in quality. For any application where the answer matters, that trade is not close. The few milliseconds are invisible next to the inference call they precede, and the quality difference is the difference between a usable system and a broken one.

The benchmark makes the information-theoretic story brutally concrete at the extreme: on RAG question answering at high compression, hard truncation drops quality to 0.0, because the span that contains the answer has been deleted and, as the Markov argument says, $I(\tilde{X}; Y) = 0$. Over the same task at the same compression, chunk-relevance dropping holds quality at 1.0, because it scored the answer-bearing chunk as relevant and kept it. The lesson generalizes well beyond this benchmark: what you delete matters far more than how much you delete.

---

## Count tokens with the tokenizer you actually pay for

There is a subtle measurement trap here that is easy to fall into and quietly invalidates a lot of compression numbers. Token savings are only meaningful against the tokenizer your deployment is actually billed on. Different tokenizers split text differently, so the same compressed string can correspond to a different number of tokens, and a savings figure computed against the wrong tokenizer is a savings figure for a model you are not using.

Context Forge measures all three of the tokenizers a real deployment is likely to pay for:

| Model | Backend | Encoding |
|---|---|---|
| `gpt-4o` | tiktoken | `o200k_base` |
| `gpt-4` | tiktoken | `cl100k_base` |
| `gpt-2` | HuggingFace | BPE reference |

The `o200k_base` encoding used by GPT-4o is a different vocabulary from the `cl100k_base` used by GPT-4 and the embedding models, and both differ from the older GPT-2 BPE reference. A given chunk of English might be 100 tokens under one and 115 under another. Adding the production tokenizers was the entire point of the multi-backend design, because the original version of this kind of benchmark counted only GPT-2 tokens, which is a fine reference point and a poor bill. The savings axis therefore has to be computed per tokenizer rather than with a single stand-in, and the frontier is reported within each tokenizer slice so the numbers correspond to a model you might actually deploy.

---

## A learned token-droppability classifier

The most forward-looking piece of the project is the seed of an actual learned compressor: a small model that predicts, for each individual token, whether it can be dropped. Rather than scoring chunks, it scores tokens, which is a finer-grained question and a step toward a compressor that learns what to keep instead of relying on a fixed heuristic.

It trains on weak labels, which is a pragmatic way to get supervision without hand-annotating anything. Tokens that overlap the target answer are labeled "keep," and the rest are labeled "droppable." The features are deliberately cheap: character n-grams plus the token's position. From the bundled run of 1.6 million labeled tokens under the GPT-4o tokenizer:

| ROC-AUC | F1 (droppable) | Accuracy | Precision / Recall |
|---:|---:|---:|---:|
| 0.964 | 0.917 | 88.1% | 0.98 / 0.86 |

Under the per-token view the connection to the bottleneck objective becomes concrete. The classifier assigns each token $x_i$ a keep probability $p_i$, and the expected length of the compressed prompt is just

$$\mathbb{E}\big[\,|\tilde{X}|\,\big] = \sum_i p_i.$$

Minimizing that sum, subject to keeping the tokens that carry answer information, is the discrete shadow of minimizing $I(\tilde{X}; X)$ while preserving $I(\tilde{X}; Y)$, learned from weak labels rather than solved exactly. What the classifier learns is genuinely intuitive, which is the reassuring part. Morphological fragments and inflectional endings, things like `-ating`, `-ated`, `-ative`, and `izer`, get scored as droppable, because they carry grammatical form rather than content. Salient content nouns, words like `healthcare`, `claims`, `agency`, and `relationships`, get kept. From weak supervision and character features alone, the model rediscovers a fact every editor knows: function-word morphology carries less task signal than content words, so it is the first thing you can afford to lose. A ROC-AUC of 0.964 on a per-token keep-or-drop decision says the signal is strongly learnable, which is the encouraging premise for using such a classifier as a fifth, fully learned compression strategy down the line.

---

## The methodology caveat I will not hide

The quality numbers here are a preservation proxy, not a model-graded answer score. For QA it is answer-span recall, for summarization it is reference-keyword coverage, for agents it is tail-trace coverage. None of these asks a judge model whether the final answer was good; they ask whether the signal that would let a model produce a good answer survived compression. In the language above, the proxy is a tractable lower bound on $I(\tilde{X}; Y)$ rather than the end-task accuracy itself. This is a deliberate choice, and it has a real cost and a real benefit.

The cost is that a preservation proxy can in principle diverge from end-task quality. A method could preserve the answer span and still produce a worse answer for reasons the proxy does not see. The benefit is that the benchmark runs end to end with no paid inference, which means anyone can reproduce every number for free, and the results cannot be quietly massaged, because every dataset id, split, tokenizer, and compression ratio lives in a single manifest that is the only source of truth. The deployed report contains no baked-in numbers; the static dashboard renders whatever the last run produced, so the charts are downstream of the data rather than decoration on top of it. The code is structured so that an LLM judge can be slotted in behind a flag without changing the report format, for anyone who wants to pay for graded scoring. But the cheap proxy is already enough to establish the result that matters, which is that the choice of what to cut dominates the choice of how much, and that relevance-aware compression is close to free quality compared to truncation.

---

## What to actually do with this

If you are compressing prompts in production, the practical guidance falls out of the frontier. Do not truncate, because truncation is the one method that can take your quality to zero on exactly the tasks, retrieval and grounded QA, where you most need the answer to survive, and the Markov argument says it will whenever the deleted region holds the answer span. Use a relevance-aware method, either chunk-relevance dropping or the cache-eviction proxy, both of which sit on the frontier at roughly 51 percent savings for roughly 92 percent quality. Measure your savings against the tokenizer you are billed on, not a convenient reference, because the whole point is the bill. And if you are willing to invest in a learned compressor, the token-droppability classifier shows the signal is there to be learned, which means a model that decides what to keep can do better than any fixed rule. The headline is small enough to carry around: in prompt compression, what you delete matters far more than how much.
