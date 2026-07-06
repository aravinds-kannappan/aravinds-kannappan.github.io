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

After reading enough individual circuit papers, I wanted a map. "A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models" by Rai, Zhou, Feng, Saparov, and Yao (2025) is the one that finally gave me a coherent one. What sets it apart from the other surveys is that it is **task-centric** rather than technique-centric: it organizes the field around what you are trying to find out, not around whichever tool you already know how to use. That framing is the contribution, and I think it is the right one, both for a newcomer and for anyone trying to see the whole board before betting on a square.

---

## Three things you can study

The paper roots everything in three objects, following Olah's original taxonomy, and almost every technique in the field is best understood as a lens onto one of them.

- **Features**: human-interpretable properties encoded in activations. The token *dog* might activate directions for "animal," "has four legs," "pet." The complication is *superposition*: models encode far more features than they have neurons by packing them into overlapping, non-orthogonal directions, which is why a single neuron is usually polysemantic and why sparse autoencoders exist.
- **Circuits**: computational subgraphs that connect features and implement a behavior. The canonical example is the previous-token-head feeding an induction-head to continue a repeated sequence. Circuits are where "what is represented" becomes "how it is computed."
- **Universality**: whether the same features and circuits recur across models and tasks. This is the load-bearing question for the whole enterprise, because it decides whether anything you learn on a toy model transfers to a frontier one. If features are universal, small-model insights scale; if not, every model needs its own archaeology.

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

The tools fall into four families. **Vocabulary projection** (the logit lens and its descendants) reads intermediate activations directly in vocabulary space by multiplying them through the unembedding, which lets you watch a prediction form layer by layer. **Intervention-based** methods (activation patching, path patching, causal mediation, ablation) establish causal claims by editing activations and measuring the effect on the output; this is the family the two previous papers in this collection belong to. **Sparse autoencoders (SAEs)** attack superposition head-on by learning an overcomplete, sparse dictionary that decomposes a polysemantic activation into monosemantic features. And **others** cover probing (train a classifier to read a property off activations, which is correlational and lacks causal power) and visualization. The "beginner's roadmap" then pairs each object of study with a concrete workflow: targeted feature study (hypothesize a feature, then validate), open-ended feature study (observe, then explain), circuit study (choose behavior and dataset, define the graph, localize, explain the components), and universality study. Every workflow ends in an evaluation step, which is the discipline the field most often skips.

## Findings worth carrying around

Two clusters of results stuck with me. On **capabilities**: in-context learning runs through induction heads; factual knowledge lives largely in the feed-forward layers, which is why localized editing like ROME and MEMIT works by finding the responsible MLP components and rewriting them; and multi-step reasoning has identifiable circuit structure rather than being an inscrutable blur.

The **learning-dynamics and post-training** findings are where the safety relevance sharpens. Sudden "phase changes" in capability coincide with the formation of specific circuits (the emergence of induction circuits being the cleanest case), and grokking is framed as a *competition between a memorization circuit and a generalization circuit*, with generalization eventually winning as training continues past the point of overfitting. Post-training is even more pointed. Fine-tuning tends to *enhance existing* mechanisms rather than build new ones. Safety fine-tuning appears to push unsafe inputs into a kind of "null space" while leaving the underlying harmful directions intact, which is a mechanistic story for why jailbreaks are so easy to find and why they often generalize. And direct preference optimization seems to *route around* toxic regions rather than remove them, so the toxic vectors persist and can be reactivated. If that picture is right, a lot of what we call alignment is currently a thin veneer over machinery that is still fully present.

## Applications, including safety

The survey is generous about downstream use. **Knowledge editing** localizes a fact and rewrites only the responsible parameters. **Generation steering** manipulates activations at inference time, either by activating and suppressing specific neurons or by pushing along a linear direction (the Othello "flip a player's turn" result is a vivid example). On **AI safety** specifically, it catalogues *enumerative safety* (try to enumerate all features and inspect the dangerous ones), *latent-space monitoring* (find linear directions for truthfulness, refusal, and jailbreaking, and watch or steer them), prompt-injection and adversarial-example detection from circuit-level insight, and *SHIFT*, which improves a classifier's generalization by identifying and ablating spurious human-interpretable features. Benchmarks are maturing too: CausalGym, RAVEL, MIB, SAEBench, and FIND for features, and `tracr` plus MIB for circuits, with methods like distributed alignment search, ACDC, and EAP-IG among the stronger performers.

## Sparse autoencoders, in a bit more depth

Because superposition is the reason a single neuron is usually uninterpretable, sparse autoencoders deserve more than a one-line mention, and the survey treats them as their own technique family for good reason. The setup is dictionary learning. You take a model's activation vector $h \in \mathbb{R}^d$ and learn an over-complete dictionary of features so that $h \approx \sum_i f_i(h)\, \mathbf{d}_i$, where the codes $f_i(h)$ are encouraged to be sparse (most near zero for any given input) by an $L_1$ penalty, and the dictionary size can be many times $d$. The bet is that the true features the model uses are sparse and roughly linear in activation space, so an over-complete sparse code will recover them one-per-dictionary-element even though they were packed into overlapping directions in the original neurons. When it works, you get *monosemantic* features you can name, which is what neurons stubbornly refuse to be.

The survey is careful to keep this honest, and the caveats are the interesting part. SAEs introduce a reconstruction-versus-sparsity trade-off with no ground truth, so "did we find the real features or a convenient basis?" is genuinely hard to answer, which is exactly why benchmarks like SAEBench, RAVEL, and MIB exist. One finding it highlights is deflating: on the MIB benchmark, SAE features were reported to be *no more effective* than individual neurons at isolating a target concept in some settings, contrary to the earlier optimism, and distributed alignment search often came out ahead. This connects back to the survey's spine. Probing, the other feature-reading tool, is *correlational*: a probe can decode a property from activations without that property being causally used by the model, so a high probe accuracy is compatible with the feature being a bystander. Interventions (patching, SAE steering) are what promote a correlational feature to a causal one, and the whole methodological arc of the field is the move from "I can read X off the activations" to "editing X changes the output the way my theory predicts." SAEs are exciting precisely because they promise scalable, unsupervised *features* to then test causally, but the survey's sober read is that we do not yet have a settled way to know when an SAE has carved the model at its joints rather than at ours.

## My take

The honest verdict the survey reaches about its own field is the part I respect most. The recurring failure mode it names is **"streetlight interpretability"**: beautiful results on toy tasks chosen because they are tractable, with little downstream comparison against baselines, and heavy reliance on a human to eyeball whether an explanation is real. That reliance is a reproducibility and subjectivity problem, and it is exactly why I read the automation papers, ACDC and its relatives, as necessary rather than optional. A field that cannot check its own explanations without an expert's intuition has not yet earned the trust we want to place in it for safety.

My own worry, which the survey gestures at in its future-work section, is the leap from decoding low-level *features* to decoding *propositions*. Knowing a model has a "kill" feature is not the same as knowing whether it represents "I support killing humans" versus "I oppose killing humans," and it is the proposition, not the feature, that carries the safety content. The work on *binding vectors* (how models associate a subject with a value, like `lives(Alice, Paris)`) is a first step toward reading structured belief rather than isolated concepts, but we are early. Until interpretability can read goals and beliefs at the level of propositions, and do it without a human in the loop deciding what counts as an explanation, I think its promise for alignment stays partly aspirational. That said, this paper is the best single starting point I know: it gave me a coordinate system, and a coordinate system is exactly what you need before you can tell whether you are making progress or just polishing the tractable corner under the streetlight.


If I had to compress the survey into one working belief, it would be this: interpretability is currently strongest at description and weakest at guarantee. We can describe features, name circuits, and even edit facts, but we cannot yet certify the absence of a mechanism, which is the thing safety actually needs. The task-centric map is valuable precisely because it makes that gap visible. It shows you which questions have mature tooling and which are still aspirational, so you can tell the difference between a solved problem and a well-lit corner. For anyone deciding where to spend effort, that honesty about the frontier is worth more than any single technique the survey catalogues.
