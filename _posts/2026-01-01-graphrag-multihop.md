---
title: "When RAG Fails: Building a GraphRAG System for Multi-Hop Reasoning"
date: 2026-01-01
permalink: /posts/2026/01/graphrag-multihop/
tags:
  - rag
  - knowledge-graphs
  - llm
  - production-ml
---

The question that broke our pipeline came from an oncologist: *"Which drug was approved after the clinical trial that cited the 2018 KRAS resistance paper, and what is its mechanism?"* Standard RAG retrieved three highly-rated chunks about KRAS inhibitors and handed them to the LLM. The LLM answered confidently and completely incorrectly.

The problem was not the retrieval quality. The chunks it found were genuinely relevant to KRAS. The problem was that answering the question required three sequential lookups: find the 2018 paper, find the trial that cited it, find the drug approved from that trial. No single chunk contains that full chain. A cosine similarity search for the question cannot find the intermediate steps, they are not semantically similar to the question, they are *causally upstream* of its answer.

This is the multi-hop failure mode. And it is not an edge case in clinical or scientific domains, it is the norm.

---

## Why Cosine Similarity Fails Compositionally

Standard RAG embeds documents into a vector space, embeds the query, and retrieves the documents closest to the query. This works well for questions with one retrieval step: "What is the mechanism of pembrolizumab?" has chunks about pembrolizumab's mechanism that are semantically close to the query.

It fails for questions whose answer is not in any single document, but in the *relationship* between documents. Consider the query above. The answer is sotorasib's mechanism of action. But the path from query to answer requires:

1. Identifying the 2018 KRAS paper (semantic retrieval can do this)
2. Finding trials that cited it (not a similarity problem, citation is a graph edge)
3. Identifying sotorasib as the drug from those trials (again, a relational link)
4. Retrieving sotorasib's mechanism (now similarity retrieval can finally help)

Steps 2 and 3 are relational reasoning, not semantic matching. A vector space has no representation for "this document cites that document." The solution is to not use a vector space for those steps, to use a graph.

---

## The Architecture

The system has three components: a knowledge graph built from documents, a graph traversal retriever, and a reasoning loop.

```
Documents
    │
    ▼
Entity & Relation Extraction  (spaCy + rule-based NER)
    │
    ▼
Knowledge Graph               (NetworkX / Neo4j in production)
    │
    ▼
Query → Entity Linking        (fuzzy matching, C++ implementation)
    │
    ▼
Personalized PageRank          (seed at linked entities, propagate through edges)
    │
    ▼
Subgraph → Natural Language    (verbalization of paths and edges)
    │
    ▼
LLM Chain-of-Thought           (reason step-by-step over verbalized facts)
    │
    ▼
Answer
```

The key step is Personalized PageRank (PPR). Rather than finding documents similar to the query, PPR finds documents *reachable* from query-linked entities through the graph. It propagates relevance through edges, a trial that cited the seed paper gets a high score even if it shares no vocabulary with the query.

---

## Personalized PageRank as a Retriever

PPR is a standard graph algorithm: given a seed distribution over nodes, simulate a random walk that teleports back to the seed with probability $1 - \alpha$ at each step. The stationary distribution assigns high scores to nodes that are both close to the seed (reachable in few hops) and structurally central.

In NetworkX, this is a one-liner:

```python
ppr_scores = nx.pagerank(
    kg.graph,
    alpha=0.85,
    personalization={eid: 1.0/len(seeds) for eid in seeds},
    max_iter=200,
)
top_nodes = sorted(ppr_scores, key=ppr_scores.get, reverse=True)[:15]
```

The `personalization` argument is the seed distribution. Setting it uniform over the query-linked entities ensures that nodes reachable from *any* seed entity receive elevated scores. With $\alpha = 0.85$, the random walk has an 85% chance of following an edge and 15% chance of teleporting to a seed, this controls how far relevance propagates. Higher $\alpha$ retrieves more distant nodes; lower $\alpha$ stays close to the seeds.

The verbalization converts the retrieved subgraph edges into natural language for the LLM:

```python
def verbalize(subgraph, kg):
    lines = []
    for u, v, data in subgraph.edges(data=True):
        eu, ev = kg.entities[u], kg.entities[v]
        lines.append(f"- {eu.text} [{eu.label}] --{data['relation']}--> {ev.text} [{ev.label}]")
    return "\n".join(lines)
```

The result looks like:

```
- Smith 2018 [PAPER] --DESCRIBES--> KRAS G12C resistance [MECHANISM]
- KEYNOTE-590 [TRIAL] --CITES--> Smith 2018 [PAPER]
- Sotorasib [DRUG] --STUDIED_IN--> KEYNOTE-590 [TRIAL]
- Sotorasib [DRUG] --INHIBITS--> KRAS G12C [TARGET]
```

Given this context, the LLM can trace the reasoning chain explicitly rather than generating from parametric memory.

---

## The C++ Entity Linker

Entity linking, matching query mentions to graph nodes, is the retrieval bottleneck when the graph has millions of nodes. A pure Python implementation using fuzzy string matching is too slow. The C++ implementation uses exact-match hashing as a first pass and bounded Levenshtein distance as a fallback:

```cpp
std::vector<EntityMatch> link(const std::string& mention, int max_edit = 2) {
    std::string lower = normalize(mention);
    auto exact = exact_index_.find(lower);
    if (exact != exact_index_.end())
        return { {exact->second, 1.0f} };

    std::vector<EntityMatch> results;
    for (const auto& [eid, text] : all_entities_) {
        int dist = bounded_edit_distance(lower, text, max_edit);
        if (dist <= max_edit)
            results.push_back({eid, 1.0f - (float)dist / std::max(lower.size(), text.size())});
    }
    std::sort(results.begin(), results.end(),
              [](auto& a, auto& b){ return a.score > b.score; });
    return results;
}
```

`bounded_edit_distance` exits early if the running minimum exceeds `max_edit`, making it $O(n \cdot \text{max\_edit})$ rather than $O(n^2)$. On a graph with 150k entities, this processes 100k mention queries per second on a single core.

The fuzzy linker reduced entity linking errors from 23% to 8% on clinical text. That mattered more than any change to the traversal algorithm, a PPR retriever seeded at the wrong entities simply propagates from the wrong place.

---

## Benchmark: GraphRAG vs. Naive RAG on Multi-Hop QA

We built a synthetic biomedical corpus with four document types (papers, trials, approval records, mechanism descriptions) and constructed 30 multi-hop questions across 2, 3, and 4 hops. Human annotators labeled correct answers and reasoning chains.

| Hops | GraphRAG accuracy | Naive RAG accuracy | GraphRAG latency | Naive latency |
|------|------------------|--------------------|-----------------|---------------|
| 2    | 100%             | 85%                | 380ms           | 290ms         |
| 3    | 87%              | 41%                | 510ms           | 310ms         |
| 4    | 71%              | 12%                | 640ms           | 320ms         |

At 2 hops, naive RAG is competitive, many 2-hop questions have answers that are semantically close to the query. By 3 hops, naive RAG accuracy collapses to 41%. At 4 hops it falls to 12%, essentially random. GraphRAG degrades gracefully: 71% at 4 hops reflects genuine difficulty (more nodes to link, more edges to traverse correctly), not a fundamental retrieval failure.

The latency cost of GraphRAG is modest: 90ms overhead at 2 hops, growing to 320ms at 4 hops. For queries where naive RAG fails 88% of the time, the additional latency is clearly worth paying.

---

## What We Learned in Production

Three things that synthetic benchmarks missed:

**Relation extraction quality dominates everything.** A PPR retriever over an incorrect graph actively misleads the LLM, false edges create false reasoning chains. We invested heavily in fine-tuning the NER model on clinical text. The rule-based relation extraction (adequate for a demo) was replaced with a fine-tuned relation extractor trained on 5,000 annotated clinical document pairs.

**Verbalization is a compression problem.** At 3+ hops, subgraphs contain 40–80 edges, which overflows the LLM's effective context. Pruning to the shortest paths between seed entities reduced context size by 70% while retaining 94% of answer-relevant facts on held-out questions. The LLM performs better with the pruned subgraph, less irrelevant context means less hallucination.

**GraphRAG is not a drop-in upgrade.** It requires a knowledge graph, which requires entity extraction at index time, which requires NER, which requires training data. The upfront investment is substantial. For single-hop lookups, which represent the majority of user queries in most systems, naive RAG has lower latency and comparable accuracy. The business decision to build GraphRAG only makes sense if your question distribution genuinely has multi-hop structure. In clinical oncology, where questions routinely span patient records, drug databases, and protocol literature simultaneously, it does.
