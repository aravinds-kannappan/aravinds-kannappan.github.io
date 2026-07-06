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

If path patching gives you a way to test a circuit hypothesis, the obvious next question is who writes the hypotheses. In practice it has been a human expert staring at attention patterns for weeks. "Towards Automated Circuit Discovery for Mechanistic Interpretability" by Conmy, Mavor-Parker, Lynch, Heimersheim, and Garriga-Alonso (NeurIPS 2023) tries to take that person out of the innermost loop. I find it valuable less for the algorithm itself, which is almost embarrassingly simple, and more for how carefully it measures its own failure.

---

## The workflow, then the automation

The paper first distills the mechanistic interpretability workflow into three steps that everyone was already doing implicitly: (1) pick a crisp behavior, curate a dataset that elicits it, and choose a metric; (2) represent the model as a computational graph of units at some granularity; (3) run many activation-patching experiments to prune away everything that does not matter. Steps 1 and 2 are human setup. **ACDC automates step 3.**

The algorithm walks the graph from output back toward input. For each candidate edge $w \to v$, it overwrites that edge's activation with the value it would take on a *corrupted* input and measures how much the model's output distribution moves, using KL divergence:

$$\text{prune } w \to v \quad \text{if} \quad D_{KL}\big(G \,\|\, H_{\setminus\{w\to v\}}\big) - D_{KL}\big(G \,\|\, H\big) < \tau.$$

If cutting the edge barely changes the KL to the full model, the edge is deemed unimportant and removed for good; then you recurse to its parents. Everything hinges on the threshold $\tau$. Importantly, ACDC uses *interchange interventions* (a real corrupted activation) rather than zero or mean ablation, which keeps the patched network closer to its own activation distribution.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 220" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="220" rx="10" fill="#f8fafc"/>
  <text x="120" y="26" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="bold">full graph</text>
  <text x="510" y="26" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="bold">pruned circuit</text>
  <!-- left dense graph -->
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
  <!-- arrow -->
  <path d="M250 120 L360 120" stroke="#1f2937" stroke-width="2" marker-end="url(#aa)"/>
  <text x="305" y="110" fill="#1f2937" font-size="12" text-anchor="middle">prune if &#916;KL &lt; &#964;</text>
  <!-- right sparse circuit -->
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

## How well does it work

They score circuit recovery against five human-found circuits (indirect object identification, docstring, greater-than, and two `tracr`-compiled tasks) plus induction, framing it as binary edge classification and reading off ROC AUC: high true-positive rate means you recovered the circuit, low false-positive rate means you did not staple on junk. The headline result is that ACDC rediscovered all five component types in GPT-2 Small's greater-than circuit and selected 68 of roughly 32,000 edges, all of which had been found by hand. On AUC it is competitive with gradient-based pruning methods like Subnetwork Probing.

## The honesty I came for

Two limitations are stated forcefully, and both matter. First, ACDC is **not robust**: performance swings hard with the threshold, the metric, and even the order in which you visit parents. Second, every single-metric method **systematically misses "negative" components**, like the negative name-mover heads in the IOI circuit that actively push against the correct answer. If your search only rewards recovering behavior, it will never find the pieces that suppress it. And the "ground truth" circuits are themselves human artifacts with over a thousand edges, so the ROC numbers have a soft floor.

## My take

I read ACDC as a proof of concept for a direction rather than a finished tool, and that is how the authors pitch it too. The simplicity of the algorithm is a feature: it shows that a lot of what expert humans do in step 3 is mechanical and can be handed to a search. But the fragility is the real story. A method whose output depends on the parent-visitation order is not yet something I would trust to certify a safety-relevant circuit in a frontier model. The deeper problem, which the negative-heads failure exposes, is that "the circuit" is not a single sparse subgraph optimizing one metric; it is a system with cooperating and competing parts, and greedy single-objective pruning is blind to the competition. I would love to see the automation pushed to the two steps ACDC leaves to humans: choosing the corrupting distribution, and interpreting what each surviving component actually computes. That is where scaling interpretability to real models will be won or lost.
