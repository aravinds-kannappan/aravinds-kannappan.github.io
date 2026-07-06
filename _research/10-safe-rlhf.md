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

There is a tension at the heart of RLHF that a single reward number papers over: helpfulness and harmlessness pull against each other. A model that refuses everything is safe and useless; a model that answers everything is helpful and dangerous. "Safe RLHF: Safe Reinforcement Learning from Human Feedback" by Dai, Pan, Sun, Ji and colleagues at Peking University (2023), the method behind the Beaver models, is my favorite treatment of this because it stops pretending the two objectives are one and reformulates alignment as *constrained* optimization. The math is clean enough to actually reason about.

---

## Decouple the preferences

The first move is at the annotation stage. Instead of asking crowdworkers for one muddled "which is better," Safe RLHF collects two separate judgments per response pair: which is more **helpful**, and which is more **harmless**, plus a binary safety label. This avoids the confusion where an annotator has to trade off two incommensurable things in their head, and it lets you train two independent models.

The **reward model** $R_\phi$ is trained on the helpfulness data with the standard Bradley-Terry pairwise loss:

$$\mathcal{L}_R(\phi) = -\,\mathbb{E}_{(x,y_w,y_l)}\big[\log \sigma\big(R_\phi(y_w,x) - R_\phi(y_l,x)\big)\big].$$

The **cost model** $C_\psi$ is trained on the harmlessness data, but with an extra classification term so that it not only ranks responses by harm but also learns the sign of harm via a virtual boundary response $y_0$ with $C_\psi(y_0,x)=0$:

$$p(y \succ y_0 \mid x) = \sigma\big(s(y)\cdot C_\psi(y,x)\big), \qquad s(y) = \begin{cases} +1 & y \text{ harmful} \\ -1 & y \text{ harmless.}\end{cases}$$

So $C_\psi(y,x) > 0$ means unsafe and $C_\psi(y,x) \le 0$ means safe. That calibrated zero is what makes the constraint below meaningful.

## Safe RL as a constrained MDP

Now the objective. Rather than maximize a blended reward, Safe RLHF maximizes helpfulness *subject to a harmlessness constraint*, a constrained MDP:

$$\max_\theta \; \mathbb{E}_{x\sim D,\, y\sim\pi_\theta}\big[R_\phi(y,x)\big] \quad \text{s.t.} \quad C_\psi(y,x) \le 0.$$

Constraints on every response are intractable, so they push it into expectation with a margin $d$ and solve the Lagrangian dual, alternately updating the policy $\theta$ and the multiplier $\lambda \ge 0$:

$$\min_\theta \max_{\lambda \ge 0}\; \big[-\mathcal{J}_R(\theta) + \lambda\cdot \mathcal{J}_C(\theta)\big], \qquad \mathcal{J}_C(\theta) = \mathbb{E}[C_\psi(y,x)] + d.$$

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 220" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="220" rx="10" fill="#f8fafc"/>
  <!-- axes -->
  <line x1="70" y1="185" x2="560" y2="185" stroke="#1f2937" stroke-width="1.4"/>
  <text x="315" y="210" font-size="12" text-anchor="middle" fill="#1f2937">training step &#8594;</text>
  <text x="30" y="100" font-size="12" text-anchor="middle" fill="#1f2937" transform="rotate(-90 30 100)">value</text>
  <!-- lambda curve rises then falls -->
  <path d="M70 170 C170 150, 210 40, 300 45 S 470 150, 560 165" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <text x="300" y="35" font-size="12" fill="#2563eb" text-anchor="middle">Lagrange multiplier &#955;</text>
  <!-- cost moving average declines -->
  <path d="M70 90 C200 100, 320 140, 560 158" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5 4"/>
  <line x1="70" y1="150" x2="560" y2="150" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="120" y="145" font-size="11" fill="#555">cost = 0 (safety boundary)</text>
  <text x="440" y="115" font-size="12" fill="#dc2626">cost moving average</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">When the model is unsafe (cost above zero) the multiplier climbs and penalizes harm harder; once safe, it relaxes so helpfulness is not over-sacrificed.</figcaption>
</figure>

The multiplier $\lambda$ is the elegant part: it is not a hyperparameter you guess, it *moves*. When the model's average cost is above zero (unsafe), $\lambda$ rises and the harm penalty bites harder; once the model is safe, $\lambda$ relaxes so you stop over-paying for harmlessness at the expense of usefulness. This is the dynamic that a fixed reward-shaping weight $R_\phi - \nu\, C_\psi$ can never capture: too small a $\nu$ and it stays unsafe, too large and it becomes a refusal machine.

## Results

Three iterations produced Beaver-v1 through v3. The harmful-response rate on their evaluation set dropped from 53.08% for the base Alpaca-7B to 2.45% for Beaver-v3, *while* helpfulness improved, and the method beat static reward shaping across seven tested weights. Decoupling the annotations also raised inter-rater agreement (helpfulness went from 61.65% to 69.00%).

## My take

I like this paper because it takes a fuzzy alignment goal and gives it a control-theoretic backbone. Framing harmlessness as a constraint rather than a term in a weighted sum is the right modeling choice: "be safe" is genuinely a constraint you must satisfy, not a quantity you trade linearly against helpfulness, and the adaptive multiplier operationalizes that intuition. The virtual-boundary trick that gives the cost model a calibrated zero is a small piece of craft I admire, because without a meaningful zero the constraint $C_\psi \le 0$ would be arbitrary.

The honest limits are worth stating. The safety guarantee is only as good as the cost model, so this is really *learned-constraint* satisfaction, and a cost model with blind spots yields a policy with matching blind spots, exactly the failure the agent and reward-hacking papers dramatize. It also operates on single-turn conversations, which is the setting the later agent work shows is the least dangerous one. And "harmful response rate 2.45%" is a distribution-dependent number, not a bound. So I read Safe RLHF as the correct *shape* for the objective, constrained rather than blended, wrapped around learned components that remain the weak link. Get better cost models and this framework scales; keep guessing a fixed safety weight and you will always be either too timid or too reckless.
