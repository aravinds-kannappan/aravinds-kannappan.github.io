---
title: "Safe RLHF: Helpfulness and Harmlessness as a Lagrangian"
excerpt: "Peking University's Beaver decouples 'is it helpful' from 'is it safe' into two models, then balances them with a Lagrange multiplier that moves during training instead of a fixed weight you guess in advance."
collection: research
order: 10
date: 2023-10-19
tags:
  - RLHF
  - constrained optimization
  - alignment
  - AI safety
---

There is a tension at the heart of RLHF that a single reward number papers over: helpfulness and harmlessness pull against each other. A model that refuses everything is safe and useless; a model that answers everything is helpful and dangerous. "Safe RLHF: Safe Reinforcement Learning from Human Feedback" by Dai, Pan, Sun, Ji and colleagues at Peking University (2023), the method behind the Beaver models, is my favorite treatment of this because it stops pretending the two objectives are one and reformulates alignment as *constrained* optimization. The math is clean enough to actually reason about, which is rare in alignment.

---

## The problem with a single reward

Standard RLHF collapses everything into one scalar. Human preferences are modeled with the Bradley-Terry model: given a prompt $x$ and two responses, the probability the winner $y_w$ is preferred over the loser $y_l$ is

$$p^*(y_w \succ y_l \mid x) = \sigma\big(R(y_w, x) - R(y_l, x)\big),$$

and you fit the reward $R_\phi$ by minimizing the negative log-likelihood over a preference dataset. The trouble is that "which response is better?" forces a crowdworker to fuse helpfulness and harmlessness into one judgment, and the two genuinely conflict. A refusal is safe but unhelpful; a detailed answer to "how do I become a serial killer" is helpful in form but harmful. Baking both into one number both confuses the annotator and forces the optimizer to pick a fixed trade-off it can never revisit.

## Decouple the preferences into two models

Safe RLHF's first move is at the annotation stage: collect *two* separate judgments per response pair, which is more helpful, and which is more harmless, plus a binary safety label across 14 harm categories (a response is "safe" only if it poses no risk in any of them). This yields two datasets and lets you train two independent models.

The **reward model** $R_\phi$ is trained on the helpfulness data with the standard Bradley-Terry pairwise loss:

$$\mathcal{L}_R(\phi) = -\,\mathbb{E}_{(x,y_w,y_l)}\big[\log \sigma\big(R_\phi(y_w,x) - R_\phi(y_l,x)\big)\big].$$

The **cost model** $C_\psi$ is trained on the harmlessness data, but with an extra classification term so it not only *ranks* responses by harm but learns the *sign* of harm. Introduce a virtual boundary response $y_0$ that sits exactly on the safe/unsafe boundary with $C_\psi(y_0, x) = 0$, and a harmfulness sign $s(y) = +1$ if $y$ is harmful and $-1$ if harmless. Then the model is trained so that

$$p(y \succ y_0 \mid x) = \sigma\big(s(y)\cdot C_\psi(y,x)\big),$$

which pushes $C_\psi(y,x) > 0$ for unsafe responses and $C_\psi(y,x) \le 0$ for safe ones. That calibrated zero is the load-bearing detail: without a meaningful zero, "is this response safe?" would have no threshold, and the constraint below would be arbitrary. Empirically the cost model cleanly separates the response distribution into safe and unsafe clusters, which is what makes it usable as a live safety signal during RL.

## Safe RL as a constrained MDP

Now the objective. Rather than maximize a blended reward, Safe RLHF maximizes helpfulness *subject to a harmlessness constraint*, which is exactly the setting of a Constrained Markov Decision Process. The primal goal is

$$\max_\theta \; \mathbb{E}_{x\sim D,\, y\sim\pi_\theta}\big[R_\phi(y,x)\big] \quad \text{s.t.} \quad C_\psi(y,x) \le 0 \;\; \forall\, y.$$

Enforcing the constraint on every possible response is intractable, so they push it into expectation with a margin hyperparameter $d$ and solve the Lagrangian dual, alternately updating the policy parameters $\theta$ and the multiplier $\lambda \ge 0$:

$$\min_\theta \max_{\lambda \ge 0}\; \big[-\mathcal{J}_R(\theta) + \lambda\cdot \mathcal{J}_C(\theta)\big], \qquad \mathcal{J}_R(\theta) = \mathbb{E}[R_\phi(y,x)], \quad \mathcal{J}_C(\theta) = \mathbb{E}[C_\psi(y,x)] + d.$$

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 220" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="220" rx="10" fill="#f8fafc"/>
  <line x1="70" y1="185" x2="560" y2="185" stroke="#1f2937" stroke-width="1.4"/>
  <text x="315" y="210" font-size="12" text-anchor="middle" fill="#1f2937">training step &#8594;</text>
  <text x="30" y="100" font-size="12" text-anchor="middle" fill="#1f2937" transform="rotate(-90 30 100)">value</text>
  <path d="M70 170 C170 150, 210 40, 300 45 S 470 150, 560 165" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <text x="300" y="35" font-size="12" fill="#2563eb" text-anchor="middle">Lagrange multiplier &#955;</text>
  <path d="M70 90 C200 100, 320 140, 560 158" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5 4"/>
  <line x1="70" y1="150" x2="560" y2="150" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="120" y="145" font-size="11" fill="#555">cost = 0 (safety boundary)</text>
  <text x="440" y="115" font-size="12" fill="#dc2626">cost moving average</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">When the model is unsafe (cost above zero) the multiplier climbs and penalizes harm harder; once safe, it relaxes so helpfulness is not over-sacrificed.</figcaption>
</figure>

The multiplier $\lambda$ is the elegant part: it is not a hyperparameter you guess, it *moves*. When the model's average cost is above zero (unsafe), $\lambda$ rises and the harm penalty bites harder; once the model is safe, $\lambda$ relaxes so you stop over-paying for harmlessness at the expense of usefulness. This is the dynamic that a fixed reward-shaping weight, $R_\nu(y,x) = R_\phi(y,x) - \nu\, C_\psi(y,x)$ for a constant $\nu$, can never capture: too small a $\nu$ and the model stays unsafe, too large and it becomes a refusal machine, and no single $\nu$ tracks the model as its safety changes over training.

## Results

They ran the pipeline for three iterations, producing Beaver-v1 through v3, with red-teaming from the second round on to keep expanding the set of adversarial prompts. The harmful-response rate on their evaluation set dropped from **53.08%** for the base Alpaca-7B to **2.45%** for Beaver-v3, *while helpfulness improved* rather than degrading. Elo scores judged by both GPT-4 and humans rose substantially on both axes across the three rounds, and the two evaluators mostly agreed. Against static reward shaping tested at seven weights ($\nu \in \{0.01, 0.5, 1, 2, 5, 10, 100\}$), Safe RLHF dominated: extreme weights over-optimized one objective, and even moderate weights could not match the constrained method's balance. Decoupling the annotations also raised inter-rater agreement among crowdworkers, from 61.65% to **69.00%** on helpfulness, because they no longer had to fuse two criteria in their heads. An ablation confirmed both design choices matter: removing the cost model's classification ability degrades it toward reward shaping, and removing the dynamic multiplier update hurts too.

## Why the cost model is still a Bradley-Terry model

The cost model's extra classification term can look like an ad hoc bolt-on, so it is worth seeing that it is really the same Bradley-Terry preference model, just anchored to an absolute reference. Introduce the virtual boundary response $y_0$ with $C_\psi(y_0, x) = 0$. If a response $y$ is unsafe ($s(y) = +1$), the cost model should prefer it over the boundary (it is *more* costly), so we want to maximize

$$p(y \succ y_0 \mid x) = \sigma\big(C_\psi(y,x) - C_\psi(y_0,x)\big) = \sigma\big(C_\psi(y,x)\big) = \sigma\big(s(y)\cdot C_\psi(y,x)\big),$$

and symmetrically, if $y$ is safe ($s(y) = -1$), we want the boundary preferred over $y$,

$$p(y_0 \succ y \mid x) = \sigma\big(-C_\psi(y,x)\big) = \sigma\big(s(y)\cdot C_\psi(y,x)\big).$$

Both cases collapse to the same expression, so the classification term is just Bradley-Terry preference *against a fixed zero-cost reference*, and the beauty is that you never need the content of $y_0$: the label $s(y)$ carries all the information. That single design choice is what upgrades the cost model from a relative ranker (which response is more harmful?) into an absolute classifier (is this response over the line?), and only an absolute classifier can define the feasibility constraint $C_\psi \le 0$ that the constrained optimization needs.

The training dynamics of the multiplier are worth picturing concretely, because they are what a fixed weight cannot imitate. Early in a Safe RLHF iteration the model still emits unsafe responses, the moving-average cost sits above zero, so $\lambda$ climbs and the effective objective tilts hard toward suppressing harm. As the policy becomes safe the average cost drops below zero, the constraint goes slack, and $\lambda$ decays back down so the optimizer stops paying for extra harmlessness and can recover helpfulness. The published training curve shows exactly this hump-shaped multiplier: a rise while the model is unsafe, a peak near the crossover, and a decay as safety is achieved. That is a feedback controller, and it is why the ablation that freezes the multiplier degrades the method back toward plain reward shaping. The constraint is not a number you set, it is a set-point the training loop servos toward.

## My take

I like this paper because it takes a fuzzy alignment goal and gives it a control-theoretic backbone. Framing harmlessness as a *constraint* rather than a term in a weighted sum is the right modeling choice, and it is not just aesthetics: "be safe" is genuinely a constraint you must satisfy, not a quantity you trade linearly against helpfulness, and the adaptive multiplier operationalizes that intuition in a way that responds to the model's current behavior instead of a guess made before training started. The virtual-boundary trick that gives the cost model a calibrated zero is a small piece of craft I admire, because it is what turns a vague "safer is better" into a crisp feasibility condition $C_\psi \le 0$.

The honest limits are worth stating, and they connect this paper to the rest of the collection. The safety guarantee is only ever as good as the cost model, so this is really *learned-constraint* satisfaction, and a cost model with blind spots yields a policy with matching blind spots, which is exactly the failure mode the agent and reward-hacking papers dramatize at agentic scale. It also operates on single-turn conversations, which the OpenAgentSafety results suggest is the *least* dangerous setting, since the hard failures emerge over multi-turn context and tool use. And "harmful response rate 2.45%" is a distribution-dependent number on their own evaluation set, not a bound; a determined red-teamer or a distribution shift can move it. So I read Safe RLHF as the correct *shape* for the objective, constrained rather than blended, wrapped around learned components that remain the weak link. Get better, harder-to-fool cost models and this framework scales gracefully; keep guessing a fixed safety weight and you will always be either too timid or too reckless, and never the right amount as the model changes underneath you.


The reason I hold this paper up as the right shape is that it is honest about being a shape rather than a solution. It does not claim to have measured harm correctly; it claims that once you have some measure of harm, treating it as a constraint and servoing a multiplier toward the boundary beats blending it into the reward at a fixed rate. That is a modular claim, and it composes: improve the cost model and the whole system improves without touching the optimizer. Most alignment methods entangle the measurement and the objective so tightly that a better signal means a redesign. Safe RLHF keeps them separate, which is why I think its constrained-optimization framing will outlast the particular Beaver models it produced.
