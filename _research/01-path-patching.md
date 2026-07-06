---
title: "Path Patching: Turning 'This Circuit Explains the Behavior' Into a Testable Claim"
excerpt: "Goldowsky-Dill et al. give interpretability a quantitative language for localization: express a hypothesis as a set of paths, patch the rest with a counterfactual input, and measure what is left unexplained."
collection: research
order: 1
date: 2023-04-16
tags:
  - interpretability
  - causal analysis
  - AI safety
---

Most mechanistic interpretability claims have the shape "this small set of components is responsible for this behavior." The trouble is that, stated that way, the claim is almost impossible to falsify. Which components, at what granularity, on which inputs, and how much of the behavior do they actually carry? "Localizing Model Behavior with Path Patching" by Goldowsky-Dill, MacLeod, Sato, and Arora (Redwood Research, 2023) is the paper that made this vague claim precise. I keep coming back to it because it does the unglamorous but essential thing: it defines what a localization claim *is* and how to measure whether it holds.

---

## The setup: computation as a graph of paths

Represent the forward pass as a directed acyclic graph $\mathcal{G}$ where nodes are functions (attention heads, MLPs, individual neurons) and edges carry activations. The key move is to work with *paths* through this graph rather than individual nodes. A hypothesis is a tuple $(\mathcal{G}, \delta, P, D)$: a set of *important paths* $P$, a dissimilarity metric $\delta$, and an input distribution $D$.

To test it, you run two inputs at once. A *reference* input $x_r$ flows along the important paths; a *counterfactual* input $x_c$ flows along everything else. The paper's clever bit of plumbing is what they call *treeify*: because the residual stream lets a component feed many downstream consumers, you first copy the graph so that every path has its own private copy of the input, and only then can you route $x_r$ and $x_c$ independently without cross-talk.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 250" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="250" rx="10" fill="#f8fafc"/>
  <!-- nodes -->
  <g stroke="#1f2937" stroke-width="1.5" fill="#ffffff">
    <rect x="40" y="105" width="70" height="40" rx="6"/>
    <rect x="200" y="40" width="90" height="40" rx="6"/>
    <rect x="200" y="170" width="90" height="40" rx="6"/>
    <rect x="380" y="105" width="90" height="40" rx="6"/>
    <rect x="540" y="105" width="70" height="40" rx="6"/>
  </g>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="75" y="130">input</text>
    <text x="245" y="65">head A</text>
    <text x="245" y="195">head B</text>
    <text x="425" y="130">head C</text>
    <text x="575" y="130">logits</text>
  </g>
  <!-- important path (blue): input -> A -> C -> logits -->
  <g stroke="#2563eb" stroke-width="2.5" fill="none" marker-end="url(#ab)">
    <path d="M110 118 L200 66"/>
    <path d="M290 60 L380 118"/>
    <path d="M470 125 L540 125"/>
  </g>
  <!-- unimportant path (grey dashed): input -> B -> C -->
  <g stroke="#9ca3af" stroke-width="2" stroke-dasharray="5 4" fill="none" marker-end="url(#ag)">
    <path d="M110 132 L200 188"/>
    <path d="M290 188 L380 132"/>
  </g>
  <defs>
    <marker id="ab" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2563eb"/></marker>
    <marker id="ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#9ca3af"/></marker>
  </defs>
  <text x="320" y="235" fill="#2563eb" font-size="12" text-anchor="middle">blue = important path carries the reference input; grey dashed = patched with the counterfactual</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">A hypothesis names the important paths. Everything else receives a counterfactual input. If the output barely moves, the named paths were sufficient.</figcaption>
</figure>

## The metric that makes it quantitative

The heart of the paper is one honest number. The *Average Unexplained Effect* of a hypothesis $H$ is

$$\mathrm{AUE}(H) = \mathbb{E}_{(x_r,x_c)\sim D}\big[\,\delta\big(G(x_r),\, G_H(x_r,x_c)\big)\,\big],$$

where $G(x_r)$ is the real output and $G_H(x_r,x_c)$ is the output after routing the counterfactual through the unimportant paths. A perfect hypothesis has $\mathrm{AUE}=0$: swapping the "irrelevant" paths changes nothing. To make this readable across behaviors, they normalize by the *Average Total Effect* (the AUE when nothing is important) to get a **proportion explained**,

$$\Big(1 - \frac{\mathrm{AUE}(H)}{\mathrm{ATE}(H)}\Big)\times 100\%.$$

One subtlety I appreciate: they warn against metrics like "difference in expected loss," because errors on different inputs can cancel and flatter a bad hypothesis. They recommend KL divergence, which does not cancel. That kind of care about the measurement, not just the method, is what separates this from a lot of interpretability work.

## What they found

Applied to induction heads in a two-layer attention-only transformer, path patching lets you refine a hypothesis iteratively. Starting from a naive "direct paths only" story that explained 28% of the behavior, successive refinements (positional information, a longer key/query window) pushed it past 74%, and along the way exposed that "induction" heads also run a distinct **parroting** heuristic, copying tokens simply because they appeared before. Compared against causal tracing and zero ablation on GPT-2, path patching stays on-distribution, so it flags fewer and more precisely targeted components.

## My take

What I like most is the intellectual honesty baked into the formalism. The AUE measures *sufficiency*, not completeness: a superset of the true circuit also scores well, and the authors say so plainly. It makes no claims off the tested distribution, and it cannot by itself *prove* a hypothesis because that would require every input. Those are not footnotes, they are the correct epistemic scope for the tool, and stating them is what lets you trust the positive results.

Where I think it is thin is the human bottleneck. Path patching is a beautiful referee, but it still needs a person to invent the hypotheses to referee, which is exactly why the next paper, ACDC, tries to automate the search. My other worry is the sufficiency-versus-completeness gap in safety settings: a circuit that is sufficient on your distribution can quietly rely on a second mechanism that only shows up on rare inputs, which is precisely where safety failures live. Still, if I want to say "these paths carry this behavior" and mean something by it, this is the vocabulary I reach for.
