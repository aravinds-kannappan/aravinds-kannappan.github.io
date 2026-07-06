---
title: "ACDC: Teaching the Computer to Find the Circuit for You"
excerpt: "Conmy et al. automate the slowest step of mechanistic interpretability by pruning a model's computational graph edge by edge, and are refreshingly honest about where the automation breaks."
collection: research
order: 2
date: 2023-10-28
tags:
  - interpretability
  - circuits
  - automation
  - AI safety
---

If path patching gives you a way to test a circuit hypothesis, the obvious next question is who writes the hypotheses. In practice it has been a human expert staring at attention patterns for weeks. "Towards Automated Circuit Discovery for Mechanistic Interpretability" by Conmy, Mavor-Parker, Lynch, Heimersheim, and Garriga-Alonso (NeurIPS 2023) tries to take that person out of the innermost loop. I find it valuable less for the algorithm itself, which is almost embarrassingly simple, and more for the discipline it brings: it names the workflow everyone was already following, and it measures its own automation's failures with unusual candor.

---

## The workflow, named

The paper's first contribution is to write down the mechanistic interpretability workflow that prior circuit papers had followed implicitly:

1. **Pick a behavior, dataset, and metric.** Choose a crisp behavior (indirect object identification, greater-than, docstring completion), curate a dataset that elicits it, and pick a metric that quantifies how well the model does it (logit difference, probability difference, KL).
2. **Divide the network into a graph of units.** Decide the granularity: attention heads and MLPs, or finer (query/key/value activations, individual neurons, per-token-position nodes). Node connectivity must be faithful to the real computation, and following the residual-stream logic, components in non-adjacent layers are treated as directly connected.
3. **Patch activations to isolate the subgraph.** Run many activation-patching experiments to prune away everything that does not matter, iterating until a sparse circuit remains.

Steps 1 and 2 are human setup. **ACDC automates step 3**, which is the part that eats the weeks.

## The algorithm

ACDC walks the graph in reverse topological order, from the output back toward the input. For each candidate edge $w \to v$, it overwrites that edge's activation with the value it would take on a *corrupted* input $x'$, runs the forward pass, and measures how much the model's output distribution moves relative to the full model, using KL divergence. If cutting the edge barely changes that KL, the edge is judged unimportant and removed for good; then the recursion continues into the parents of $v$. In pseudocode the whole thing is a dozen lines:

$$\textbf{for } v \textbf{ in reverse-topological order:} \quad \textbf{for } w \in \text{parents}(v):$$
$$\text{if } \; D_{KL}\big(G \,\|\, H_{\setminus\{w\to v\}}\big) - D_{KL}\big(G \,\|\, H\big) < \tau \;\Rightarrow\; \text{prune } w \to v.$$

Everything hinges on the single threshold $\tau$: sweep it, and you trace out a family of circuits from tiny-and-lossy to large-and-faithful. One design choice matters a lot. ACDC uses *interchange interventions*, a real corrupted activation drawn from a related input, rather than zero or mean ablation. Following the causal-scrubbing argument, zero and mean ablation push the network off its own activation distribution and can produce misleading attributions, whereas an interchange keeps the patched network in a regime it actually visits.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 220" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="220" rx="10" fill="#f8fafc"/>
  <text x="120" y="26" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="bold">full graph</text>
  <text x="510" y="26" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="bold">pruned circuit</text>
  <g stroke="#9ca3af" stroke-width="1.2" fill="none">
    <path d="M60 70 L120 120"/><path d="M60 70 L180 120"/><path d="M120 70 L60 120"/>
    <path d="M120 70 L180 120"/><path d="M180 70 L120 120"/><path d="M60 120 L120 170"/>
    <path d="M120 120 L180 170"/><path d="M180 120 L60 170"/><path d="M120 70 L120 120"/>
  </g>
  <g fill="#cbd5e1" stroke="#64748b" stroke-width="1.2">
    <circle cx="60" cy="70" r="10"/><circle cx="120" cy="70" r="10"/><circle cx="180" cy="70" r="10"/>
    <circle cx="60" cy="120" r="10"/><circle cx="120" cy="120" r="10"/><circle cx="180" cy="120" r="10"/>
    <circle cx="60" cy="170" r="10"/><circle cx="120" cy="170" r="10"/><circle cx="180" cy="170" r="10"/>
  </g>
  <path d="M250 120 L360 120" stroke="#1f2937" stroke-width="2" marker-end="url(#aa)"/>
  <text x="305" y="110" fill="#1f2937" font-size="12" text-anchor="middle">prune if &#916;KL &lt; &#964;</text>
  <g stroke="#2563eb" stroke-width="2" fill="none" marker-end="url(#ab2)">
    <path d="M450 70 L510 118"/><path d="M570 70 L516 118"/><path d="M510 128 L510 168"/>
  </g>
  <g fill="#bfdbfe" stroke="#2563eb" stroke-width="1.6">
    <circle cx="450" cy="70" r="10"/><circle cx="570" cy="70" r="10"/>
    <circle cx="510" cy="122" r="10"/><circle cx="510" cy="172" r="10"/>
  </g>
  <defs>
    <marker id="aa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker>
    <marker id="ab2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2563eb"/></marker>
  </defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">ACDC walks output to input, cutting each edge whose removal barely changes the KL to the full model, leaving a sparse subgraph.</figcaption>
</figure>

## Two baselines, adapted

To argue ACDC is doing something reasonable and not just getting lucky, the authors adapt two gradient-based methods to the same task. **Subnetwork Probing (SP)** learns a differentiable mask over components with an objective that trades off performance against sparsity, then rounds the mask to a hard subnetwork. **Head Importance Score for Pruning (HISP)** ranks heads by an importance score and keeps the top $k$. Both were originally built with zero ablation, so the paper generalizes them to use corrupted activations for a fair comparison. The point of the comparison is not to crown a winner but to see whether an entirely different optimization lands on the same circuits.

## How well it works

Evaluation is framed as **binary edge classification** against five canonical, human-discovered circuits (indirect object identification, docstring, greater-than, and two `tracr`-compiled algorithmic tasks) plus induction. Each edge is either "in the circuit" or not, and the methods are scored by ROC AUC, where a high true-positive rate answers "did you recover the circuit?" and a low false-positive rate answers "did you avoid stapling on junk?". They also report a task-agnostic measure: KL divergence of the recovered circuit versus number of edges, tracing a Pareto frontier where lower KL at fewer edges is better.

The headline results: ACDC rediscovered **all five component types** in GPT-2 Small's greater-than circuit, and on the indirect-object task it selected **68 of roughly 32,000 edges**, all of which had previously been found by hand. On AUC it is competitive with the gradient methods, and above roughly 20 edges it tends to lead on the KL-versus-edges frontier. A nice sanity check comes from *reset networks*: SP and HISP can partly fit a randomly initialized network, which is evidence they sometimes hallucinate circuits that are not really there, whereas the KL all methods achieve on trained networks is much lower, confirming they are picking up real signal.

## The honesty I came for

Two limitations are stated forcefully, and both matter more than the positive results. First, ACDC is **not robust**: performance swings hard with the threshold $\tau$, with the metric (optimizing logit difference versus KL gives different circuits), and even with the order in which parents are visited. A method whose output depends on iteration order is not something I would trust to certify a safety-relevant circuit without a lot of care. Second, and deeper, every single-metric method **systematically misses "negative" components**. In the indirect-object circuit there are negative name-mover heads that actively push *against* the correct answer; because ACDC only rewards recovering behavior, it never finds pieces that suppress it unless you crank the threshold down absurdly far. And the "ground-truth" circuits are themselves human artifacts, over a thousand edges in the IOI case, and admittedly imperfect, which puts a soft ceiling on what the ROC numbers can tell you.

## Where ACDC sits next to path patching

It helps to hold ACDC and path patching side by side, because they are two halves of one loop. Path patching *tests* a hypothesis a human already wrote down, and scores how much of the behavior a named set of paths explains. ACDC *searches* for the hypothesis in the first place, by greedily pruning edges. They share the same causal substrate: both represent the model as a computational graph, both intervene with real interchange activations rather than zeros to stay on-distribution, and both quantify importance with a distributional metric. The difference is direction. One is a referee, the other is a scout. In an ideal workflow the scout proposes a sparse subgraph and the referee stress-tests it on a held-out distribution, which is roughly the pairing the field has converged on.

The comparison with the gradient baselines is also more instructive than the raw AUC numbers suggest. Subnetwork Probing and HISP both optimize a single differentiable objective (a mask trained for performance-plus-sparsity, or a top-$k$ importance ranking), whereas ACDC does discrete, greedy, edge-by-edge pruning. Gradient methods can be faster and smoother, but they inherit the *same* single-metric blindness, and the paper's reset-network control is the sharpest evidence of the danger: SP and HISP can partially "recover" a circuit from a *randomly initialized* network, which means they will sometimes hand you a confident, sparse subgraph for a mechanism that does not exist. ACDC with corrupted activations is more resistant to that particular hallucination, but only because its stopping rule is tied to a faithful KL comparison against the real model. The lesson I take is that sparsity plus good performance on one metric is not the same as having found the mechanism, and any automated circuit finder needs a faithfulness check it cannot game. Later work on edge attribution patching (a linear-approximation shortcut to ACDC's edge scores) makes the search dramatically cheaper, and benchmarks like MIB find its improved variants competitive or better, but the faithfulness-versus-sparsity tension this paper exposes does not go away with a faster search; it is a property of the target, not the algorithm.

## My take

I read ACDC as a proof of concept for a direction rather than a finished tool, and the authors pitch it exactly that way. The value of the paper is twofold. It shows that a large fraction of what expert humans do in step 3 is mechanical and can be handed to a search, which is a real result for a field whose central bottleneck is expert labor. And by measuring failure carefully, it exposes a conceptual problem I think is underappreciated: "the circuit" is not a single sparse subgraph that optimizes one metric. It is a system with cooperating *and competing* parts, and greedy single-objective pruning is structurally blind to the competition. The negative-heads failure is not a tuning issue; it is the method telling you that behavior-recovery and mechanism-recovery are not the same target.

The fragility worries me for a different reason. If we want interpretability to feed high-stakes decisions, we need methods whose outputs are stable under nuisance choices like visitation order, and ACDC is candidly not there. Where I would most like to see progress is the two steps ACDC leaves to humans: automating the choice of the corrupting distribution (which silently determines what "unimportant" even means), and automating the *functional interpretation* of the surviving components, which is the step that turns a subgraph into an explanation. ACDC automates the bookkeeping; the understanding is still ours to supply. That is real progress, and it is also a map of how far we have to go.


The broader point I draw from ACDC is about what it means to trust an interpretability result at all. A hand-built circuit carries the implicit credibility of an expert who stared at it and can defend each edge; an automatically discovered one carries none of that, so it needs its faithfulness to be legible in the method rather than in the researcher. ACDC gets part of the way there by tying its stopping rule to a KL comparison you can inspect, and that, more than the specific circuits it recovers, is the template I want automated interpretability to follow: not just find a small subgraph, but make the argument for why that subgraph is enough, in a form another person can check without rerunning the search.
