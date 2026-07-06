---
title: "Path Patching: Turning 'This Circuit Explains the Behavior' Into a Testable Claim"
excerpt: "Goldowsky-Dill et al. give interpretability a quantitative language for localization: express a hypothesis as a set of paths, patch the rest with a counterfactual input, and measure exactly what is left unexplained."
collection: research
order: 1
date: 2023-04-16
tags:
  - interpretability
  - causal analysis
  - AI safety
---

Most mechanistic interpretability claims have the shape "this small set of components is responsible for this behavior." The trouble is that, stated that way, the claim is almost impossible to falsify. Which components, at what granularity, on which inputs, and how much of the behavior do they actually carry? "Localizing Model Behavior with Path Patching" by Goldowsky-Dill, MacLeod, Sato, and Arora (Redwood Research, 2023) is the paper that made this vague claim precise. I keep coming back to it because it does the unglamorous but essential thing: it defines what a localization claim *is*, gives a procedure to test one, and attaches an honest number to how well it holds.

---

## Computation as a graph, and why paths

Start by representing the forward pass as a directed acyclic graph $\mathcal{G}$. Nodes are functions (attention heads, MLPs, individual neurons) and edges carry activations. Because transformers are residual networks, the same component feeds many downstream consumers through the residual stream, and the whole computation decomposes into a sum over paths from input to output. A handful of relatively shallow paths often carry most of the effect, which is exactly what makes path-level localization a reasonable bet.

The simplest version treats *nodes* as mediators. Suppose you think a node $f_1$ is unimportant. You define a modified computation $G_H(x_r, x_c)$ that runs the network on a *reference* input $x_r$ everywhere except that the unimportant node is fed activations computed from a *counterfactual* input $x_c$. In the language of causal mediation this is the natural indirect effect of moving from $x_r$ to $x_c$ through $f_1$, and it is a deterministic version of Pearl's path-specific effect. If $f_1$ really is unimportant, swapping it out should not change the output.

The move from nodes to *paths* is what makes the method expressive, and it needs one piece of machinery. Because a node's output can flow down several paths at once, you cannot feed a counterfactual to one path without contaminating the others. The fix is what the authors call **treeify**: you copy the graph so that every path gets its own private copy of the input. Algebraically, treeify turns $G$ into a function $G_T(x_1, \ldots, x_N)$ of $N$ arguments, one per path, obtained by fully expanding the network's equation and relabeling each occurrence of the input with its own variable. A path-level hypothesis then feeds $x_r$ to the arguments on important paths $P$ and $x_c$ to the rest:

$$G_H(x_r, x_c) = G_T(x_1, \ldots, x_N), \qquad x_i = \begin{cases} x_r & p_i \in P \\ x_c & p_i \notin P. \end{cases}$$

This is strictly more general than treating a set of nodes as the atom of explanation, because a single node participates in many paths, and you can now say "the path input to head 0 to head 1 matters, but not head 0's effect on the logits directly."

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 250" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="250" rx="10" fill="#f8fafc"/>
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
  <g stroke="#2563eb" stroke-width="2.5" fill="none" marker-end="url(#ab)">
    <path d="M110 118 L200 66"/>
    <path d="M290 60 L380 118"/>
    <path d="M470 125 L540 125"/>
  </g>
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

where $G(x_r)$ is the real output, $G_H(x_r,x_c)$ is the output after routing the counterfactual through the unimportant paths, and $\delta$ is a dissimilarity metric such as KL divergence. A perfect hypothesis has $\mathrm{AUE}=0$: swapping the "irrelevant" paths changes nothing in expectation. To make this readable across behaviors, they normalize by the *Average Total Effect* (the AUE of the hypothesis that says *nothing* is important, so every path gets the counterfactual) to get a **proportion explained**,

$$\Big(1 - \frac{\mathrm{AUE}(H)}{\mathrm{ATE}(H)}\Big)\times 100\%.$$

On this scale, 0% means the hypothesis is as bad as claiming the identified components do not matter at all, and 100% means the patched abstraction matches the full model on every input in $D$. It can even go below 0%, because a badly chosen hypothesis can be *worse* than the null.

Two subtleties show real craftsmanship. First, they warn against metrics like "difference in expected loss," because errors on different inputs can cancel: if the abstraction gets lower loss on some inputs and higher on others, the discrepancies average out and flatter a bad hypothesis. KL divergence does not cancel, so they recommend it. Second, they define a **path patching attribution**: fix a single reference input, vary the counterfactual, and drop the absolute value so you can see whether patching a set of paths pushes loss *up or down*. That directional signal is what lets you debug a hypothesis rather than just score it, because it points you at the inputs where your story is incomplete.

## A worked example: refining the induction story

The paper's centerpiece is a careful study of induction heads in a two-layer attention-only transformer, and I find the iteration more instructive than any single result. Induction is the mechanism behind "\[A\]\[B\] ... \[A\] then predict \[B\]": a previous-token head writes "A is previous" into the residual stream, and a later induction head uses that to attend from the second A to B and copy it.

They start with an "All-Initial" hypothesis (direct value, direct query, and previous-token-to-key paths) that explains only **28.2%** of the behavior. Adding positional paths jumps the value component's proportion explained from 64.1% to **86.2%**, and query from 48.7% to 72.6%, which tells you the heads read positional information, not just token identity. Extending to a longer key/query window (their "Long-QK", with a window of 3) pushes the combined story to about 55%. Then path patching attribution exposes what is still missing: head 1.5 is not a pure induction head, it also implements a **parroting** heuristic, copying a token simply because it appeared before, independent of the induction pattern. By splitting the data into an "induction" subset and a "repeats" subset, they show the gap directly: the same hypothesis gets an AUE of 0.219 on the induction subset but 0.777 on repeats. Adding the parroting paths lifts the final proportion explained to about **74%**, up from the 28% they started with.

I like this example because it shows the method being used the way it should be: not to bless a preconceived circuit, but to keep rejecting an incomplete story until the residual points at a mechanism you had not named.

## Path patching versus causal tracing and zero ablation

The paper also contrasts path patching with two popular alternatives on GPT-2 (small and XL). Causal tracing corrupts the input with Gaussian noise; zero ablation simply sets a component's output to zero. Both take the network *off* its own activation distribution, which tends to implicate many components and produce diffuse, hard-to-read attributions. Because path patching's counterfactual is a real, similar input, activations stay on-distribution, and the method flags fewer and more precisely targeted components. On a task where GPT-2 estimates a number after a prompt, path patching cleanly isolates heads 5.6 and 6.1 as recognizing "the next token is a number"; corrupting the input to those heads reduces the chance of a numeric completion, while patching a different number leaves the 88.1% numeric-completion rate essentially unchanged. That precision is the payoff of staying on-distribution.

## Relationship to causal scrubbing and mediation

Path patching is a simpler special case of *causal scrubbing* (Chan et al., 2022), which additionally reasons about equivalence classes on nodes and generally needs many more samples; path patching needs only two, $x_r$ and $x_c$, which is why it is cheap enough to run at the 1.5B-parameter scale. It also generalizes the mean-ablation used in the original indirect-object-identification circuit work, and it sits squarely in the causal mediation tradition, giving a deterministic, path-level version of natural indirect effects.

## Limitations: sufficiency, not completeness

The caveat I respect most is stated plainly: AUE measures *sufficiency*, not *completeness*. Consider a behavior implemented by a majority vote of several redundant voters. Any set of paths containing a majority reaches AUE zero, because their agreement determines the output regardless of the minority. So a hypothesis can score perfectly while including redundant machinery, or while missing a second mechanism that only fires on rare inputs. Path patching also makes no claims off the tested distribution $D$, and it cannot by itself *prove* a hypothesis, since that would require checking every input. These are not footnotes; they are the correct epistemic scope, and stating them is what lets you trust the positive results.

## My take

What I value most here is the intellectual honesty baked into the formalism rather than any single circuit. Interpretability spent years making qualitative claims that no one could adjudicate, and this paper hands the field a referee: a hypothesis is a precise object, testing it is a defined procedure, and the outcome is a number you can compare across hypotheses and datasets.

Where I stay uneasy is the human bottleneck and the sufficiency gap, and I think they compound. Path patching is a beautiful referee, but it still needs a person to invent the hypotheses to referee, which is exactly why the next paper, ACDC, tries to automate the search. And in safety settings the sufficiency-versus-completeness gap is not academic: a circuit that is sufficient on your distribution can quietly lean on a second mechanism that only surfaces on rare or adversarial inputs, which is precisely where failures live. A tool that certifies "these paths suffice here" is genuinely useful, but I would not confuse it with "these paths are the whole story." Used with that discipline, though, this is the vocabulary I reach for whenever I want "this circuit explains this behavior" to mean something I can check.
