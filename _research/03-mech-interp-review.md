---
title: "A Field Guide to Mechanistic Interpretability"
excerpt: "Rai et al. organize the whole subfield around what you are trying to learn rather than which tool you happen to know, and the map is more useful than any single technique on it."
collection: research
order: 3
date: 2025-10-13
tags:
  - interpretability
  - survey
  - AI safety
---

After reading enough individual circuit papers, I wanted a map. "A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models" by Rai, Zhou, Feng, Saparov, and Yao (2025) is the one that finally gave me a coherent one. What sets it apart from the other surveys is that it is **task-centric** rather than technique-centric: it organizes the field around what you are trying to find out, not around whichever tool you already know how to use. That framing is the contribution, and it is the right one for a newcomer.

---

## Three things you can study

The paper roots everything in three objects, following Olah's original taxonomy.

- **Features**: human-interpretable properties encoded in activations. The token *dog* might light up directions for "animal," "has four legs," "pet."
- **Circuits**: computational subgraphs that connect features and implement a behavior. The canonical example is the previous-token-head feeding an induction-head to continue a repeated sequence.
- **Universality**: whether the same features and circuits recur across models and tasks. This is the load-bearing question, because it decides whether anything you learn on a toy model transfers.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 210" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="210" rx="10" fill="#f8fafc"/>
  <g font-size="13" text-anchor="middle" fill="#1f2937">
    <rect x="30" y="80" width="140" height="46" rx="8" fill="#eff6ff" stroke="#2563eb"/>
    <text x="100" y="100">Features</text><text x="100" y="116" font-size="11" fill="#555">what is encoded</text>
    <rect x="250" y="80" width="140" height="46" rx="8" fill="#ecfdf5" stroke="#059669"/>
    <text x="320" y="100">Circuits</text><text x="320" y="116" font-size="11" fill="#555">how it is computed</text>
    <rect x="470" y="80" width="140" height="46" rx="8" fill="#fef3c7" stroke="#d97706"/>
    <text x="540" y="100">Universality</text><text x="540" y="116" font-size="11" fill="#555">does it transfer</text>
  </g>
  <g stroke="#1f2937" stroke-width="1.8" marker-end="url(#ar)">
    <path d="M170 103 L248 103"/><path d="M390 103 L468 103"/>
  </g>
  <text x="320" y="40" font-size="14" text-anchor="middle" fill="#1f2937" font-weight="bold">Techniques: logit lens &#183; activation/path patching &#183; sparse autoencoders &#183; probing</text>
  <text x="320" y="180" font-size="12" text-anchor="middle" fill="#555">each technique is a way to interrogate one of the three objects above</text>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker></defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">The survey's spine: three objects of study, with the technique families as different lenses onto them.</figcaption>
</figure>

## The techniques, sorted

The tools fall into four families: **vocabulary projection** (the logit lens, reading intermediate activations directly in vocabulary space), **intervention-based** methods (activation patching, path patching, ablation, causal mediation), **sparse autoencoders** that decompose polysemantic activations into monosemantic features to fight superposition, and **others** like probing and visualization. The "beginner's roadmap" then pairs each object of study with a concrete workflow, always ending in an evaluation step, which is exactly the discipline the field often skips.

## Findings worth carrying around

Two clusters of results stuck with me. On **capabilities**: in-context learning runs through induction heads, factual knowledge lives in the feed-forward layers (which is why localized editing like ROME works at all), and reasoning has identifiable circuit structure. On **learning dynamics and post-training** the safety-relevant news is sharper. Fine-tuning tends to *enhance existing* mechanisms rather than build new ones. Safety fine-tuning appears to push unsafe inputs into a kind of null space while the underlying "toxic" directions persist, which is a mechanistic story for why jailbreaks are so easy to find. And direct preference optimization seems to *route around* toxic regions rather than remove them. If that picture is right, a lot of alignment is currently a thin veneer over intact machinery.

## My take

The honest verdict the survey reaches about its own field is the part I respect. The recurring failure mode it names is "streetlight interpretability": beautiful results on toy tasks, chosen because they are tractable, with little downstream validation and heavy reliance on a human to eyeball whether an explanation is real. That reliance is a reproducibility and subjectivity problem, and it is why I read the automation papers (ACDC and friends) as necessary rather than optional.

My own worry, which the survey gestures at, is the leap from decoding low-level features to decoding *propositions*. Knowing a model has a "kill" feature is not the same as knowing whether it represents "I support killing humans" versus "I oppose killing humans," and it is the proposition, not the feature, that carries the safety content. Until interpretability can read structured beliefs and goals rather than isolated concepts, I think its promise for alignment stays partly aspirational. But this paper is the best single starting point I know for someone who wants to see the whole board before picking a square.
