---
title: "Building Production RAG Systems: Architecture Decisions and Hard-Won Lessons"
date: 2026-01-01
permalink: /posts/2026/01/rag-architecture/
tags:
  - RAG
  - LLMs
  - applied ML
  - production
---

Retrieval-Augmented Generation has become the default architecture for grounding LLM outputs in factual, domain-specific knowledge. But the gap between a demo RAG system and a production RAG system is enormous — most of the interesting engineering happens in the retrieval layer, not the generation layer. This post covers the architecture decisions that actually matter, based on building RAG in production for clinical workflows at Synthure.

## What RAG Actually Is (and Isn't)

The standard description: embed a query, retrieve relevant documents from a vector store, concatenate them into the LLM context, generate an answer conditioned on the retrieved context. This is correct but incomplete as a mental model for production systems.

A more useful framing: RAG is a **retrieval pipeline with a generation step at the end**. The generation step matters — prompt design, output parsing, citation grounding — but the retrieval pipeline is where most production failures occur and where most engineering effort should go. A mediocre retrieval system fed to a great LLM produces mediocre answers. A great retrieval system fed to a mediocre LLM still produces useful answers.

## The Retrieval Stack: Dense, Sparse, and Hybrid

**Dense retrieval** (embedding-based) encodes queries and documents into vectors and retrieves by cosine similarity. It handles semantic similarity well: "cardiac arrest" and "heart failure" are close in embedding space even though they share no words. The tradeoffs: it struggles with exact keyword matching (a drug name, a specific code, a proper noun), it's sensitive to the choice of embedding model, and it requires approximate nearest-neighbor infrastructure (FAISS, Pinecone, Weaviate) that has its own operational complexity.

**Sparse retrieval** (BM25, TF-IDF) retrieves by keyword overlap. It handles exact matching perfectly and is interpretable — you can always see which terms drove the match. It fails on semantic similarity and suffers from vocabulary mismatch.

**Hybrid retrieval** combines both, typically using Reciprocal Rank Fusion (RRF) to merge ranked lists from dense and sparse retrievers:

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

In our clinical setting, hybrid retrieval was non-negotiable. Medical coding requires exact ICD-10 code matching (sparse) and semantic understanding of clinical descriptions (dense). Either alone was insufficient.

## Chunking Strategy: The Decision Nobody Talks About

How you split documents into chunks has a larger impact on retrieval quality than most architectural choices, and it gets far less attention.

**Fixed-size chunking** (split every N tokens) is simple and used by most tutorials. It's also fragile: it cuts sentences mid-thought, separates related concepts across chunk boundaries, and produces chunks that are semantically incoherent. Retrieving the "right" chunk by embedding similarity may miss the immediately adjacent context that makes the answer complete.

**Semantic chunking** uses sentence boundaries, paragraph breaks, or section headers to create chunks that are coherent units of meaning. More complex to implement, but retrieval quality improves substantially.

**Hierarchical chunking** stores documents at multiple granularities — sentence, paragraph, section — and retrieves at the fine level while expanding to the coarse level for context. A query about a specific side effect retrieves the relevant sentence, but the LLM gets the full paragraph, which includes the dosing context that makes the answer interpretable.

```python
def hierarchical_chunk(document, sentence_splitter, window_size=3):
    sentences = sentence_splitter(document)
    chunks = []
    for i, sentence in enumerate(sentences):
        # Fine chunk: single sentence for retrieval
        fine = {'text': sentence, 'index': i, 'level': 'sentence'}
        # Coarse context: surrounding window for generation
        start, end = max(0, i - window_size), min(len(sentences), i + window_size + 1)
        coarse = ' '.join(sentences[start:end])
        chunks.append({'fine': fine, 'coarse': coarse})
    return chunks
```

## Reranking: Fixing Retrieval Errors Before They Reach the LLM

First-stage retrieval (dense or hybrid) optimizes for recall — retrieve a candidate set that likely contains the right documents. Reranking optimizes for precision — given the candidate set, identify the genuinely relevant documents.

Cross-encoders (models that take a (query, document) pair as input and output a relevance score) dramatically outperform bi-encoders for reranking but are too slow for first-stage retrieval. Running a cross-encoder over a shortlist of 50-100 candidates takes ~200ms — acceptable for a second stage, prohibitive for scanning thousands of documents.

The pipeline: dense/sparse retrieval → top-100 candidates → cross-encoder reranker → top-5 → LLM generation. This two-stage architecture captures the best of both worlds: embedding retrieval's speed and cross-encoder's precision.

## Grounding and Hallucination Control

Grounding is the hardest problem in production RAG: ensuring the LLM's output is faithful to the retrieved context, not confabulated from parametric knowledge.

Structural approaches that help:

**Constrained generation:** prompt the LLM to only use information from the provided context, and to explicitly state when the context doesn't contain the answer. This reduces but doesn't eliminate hallucination — LLMs have a strong prior to be helpful and will sometimes fill gaps from parametric memory.

**Citation verification:** require the LLM to cite specific spans from the retrieved context for each claim. Post-hoc, verify that the cited spans actually support the claim using an NLI model. Discard or flag responses where citations don't entail the claim.

**Typed output schemas:** constrain the LLM to output structured data (JSON with a defined schema) rather than free text. It's much harder to hallucinate a specific ICD-10 code when the schema requires a valid code from a controlled vocabulary. We enforced this at Synthure using tool-use APIs and schema validation — this reduced hallucinated clinical codes by a significant margin.

```python
def validate_rag_response(response, retrieved_context, nli_model, threshold=0.7):
    claims = extract_claims(response)
    for claim in claims:
        citation_text = get_cited_span(claim, retrieved_context)
        if citation_text is None:
            return {'valid': False, 'reason': 'uncited_claim', 'claim': claim}
        entailment_score = nli_model(premise=citation_text, hypothesis=claim)
        if entailment_score < threshold:
            return {'valid': False, 'reason': 'unsupported_claim', 'claim': claim}
    return {'valid': True}
```

## Evaluation: How Do You Know Your RAG Works?

The naive evaluation metric — "does the answer look right to me?" — doesn't scale. Production RAG needs systematic evaluation.

**Retrieval evaluation:** does the retrieval pipeline actually return the documents needed to answer the question? Requires a labeled dataset of (question, relevant_documents) pairs. Metrics: Recall@K (is the relevant document in the top K results?), MRR (mean reciprocal rank of the first relevant document).

**Generation evaluation:** given the correct context, does the LLM generate a correct, grounded answer? Metrics: faithfulness (does the answer contradict the context?), relevance (does the answer address the question?), correctness (is it factually right, per a gold standard?).

**End-to-end evaluation:** the composition of both. Requires a labeled (question, gold answer) dataset and either human evaluation or an LLM-as-judge setup. LLM-as-judge has known biases (verbosity preference, self-preference) but scales.

The expensive part is building the evaluation dataset. In the clinical domain, this meant working with domain experts to annotate (query, relevant policy passage, correct code) triples. There's no shortcut — the quality of your evaluation dataset determines the quality of your system signal. Invest in it early.

## What I'd Do Differently

If I were rebuilding from scratch: start with hybrid retrieval and a cross-encoder reranker before touching the generation layer. Invest in evaluation infrastructure before building features — you can't improve what you can't measure. Define the grounding contract (what "faithful to context" means for your domain) before writing your first prompt. And be skeptical of benchmarks from generic RAG papers — domain-specific retrieval has idiosyncrasies that general benchmarks don't capture.

RAG is the right architecture for knowledge-grounded generation. Getting it right in production is an engineering problem, not a prompting problem.
