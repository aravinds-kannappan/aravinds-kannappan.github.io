---
title: "Kolmogorov Complexity, Solomonoff Induction, and the Philosophical Limits of Aligned AGI"
date: 2026-05-01
permalink: /posts/2026/05/kolmogorov-alignment/
tags:
  - AI safety
  - algorithmic information theory
  - alignment
  - philosophy
  - mathematics
---

There is a line of thought in AI alignment that begins with a deceptively simple question: what does it mean to learn? Not to perform well on a benchmark, not to minimize a loss, but to genuinely induce the correct hypothesis from data. The mathematical answer to this question, developed across three decades of work in algorithmic information theory, is **Solomonoff induction** — a theoretically optimal Bayesian learner grounded in Kolmogorov complexity. And when you examine its foundations carefully, it reveals something troubling about the possibility of value learning: the most powerful possible learner, given all possible data, cannot in principle recover the objective function of its designer without additional structure that cannot itself be learned.

This post develops the mathematical foundations of algorithmic information theory, traces through the construction of Solomonoff induction and its descendant AIXI, and then examines what these formalisms imply for the hardest questions in AI alignment.

---

## Kolmogorov Complexity: The Length of Understanding

Let $$U$$ be a fixed universal Turing machine — a Turing machine that can simulate any other Turing machine when given an appropriate program. The **Kolmogorov complexity** of a binary string $$x \in \{0,1\}^*$$ is:

$$K(x) = \min_{p : U(p) = x} |p|$$

the length of the shortest program that causes $$U$$ to output $$x$$ and halt. It is the length of the most compressed description of $$x$$. A string is **incompressible** (random, in Kolmogorov's sense) if $$K(x) \approx |x|$$ — no program shorter than $$x$$ itself can generate it. A string is **compressible** (structured, regular) if $$K(x) \ll |x|$$ — it has a short description.

Kolmogorov complexity is not computable: the function $$x \mapsto K(x)$$ is not Turing-computable. The reason is a diagonalization argument. Suppose there were a program $$P$$ that computed $$K(x)$$ for all $$x$$. Consider the string $$x_n$$ defined as the first string of complexity greater than $$n$$. $$P$$ would find $$x_n$$ in time depending only on $$n$$, and a program that calls $$P$$ and extracts $$x_n$$ would have length $$O(\log n)$$ — much shorter than $$n$$, contradicting $$K(x_n) > n$$. This is Kolmogorov's version of Berry's paradox: "the smallest integer not definable in fewer than thirteen words" is itself defined in twelve words.

Despite being uncomputable, Kolmogorov complexity is well-defined and satisfies fundamental inequalities. The most important is the **chain rule**:

$$K(x, y) = K(x) + K(y | x) + O(\log K(x, y))$$

The complexity of the pair $$(x, y)$$ is the complexity of $$x$$ plus the complexity of $$y$$ given $$x$$, up to logarithmic terms. This is the algorithmic analog of the entropy chain rule $$H(X, Y) = H(X) + H(Y | X)$$.

The **conditional Kolmogorov complexity** $$K(y | x)$$ is:

$$K(y | x) = \min_{p: U(p, x) = y} |p|$$

the length of the shortest program that outputs $$y$$ given $$x$$ as an auxiliary input. The mutual information:

$$I(x : y) = K(x) + K(y) - K(x, y) = K(x) - K(x | y^*)$$

where $$y^*$$ is the shortest program for $$y$$, measures the shared algorithmic information between $$x$$ and $$y$$.

---

## Solomonoff's Universal Prior

In Bayesian inference, the prior $$P(h)$$ over hypotheses encodes beliefs before observing data. Solomonoff's insight was that the correct universal prior should assign probability proportional to $$2^{-K(h)}$$ — simpler hypotheses (shorter programs) should be more probable a priori. This is **Occam's razor made quantitative and universal**.

The **Solomonoff prior** over binary strings is defined via a universal monotone Turing machine $$U_M$$ (one that outputs an infinite sequence given an infinite program tape):

$$M(x) = \sum_{p: U_M(p) \text{ starts with } x} 2^{-|p|}$$

This is a **semimeasure** — it satisfies $$\sum_b M(xb) \leq M(x)$$ rather than equality, because some programs never halt. It is not a proper probability measure, but it is **enumerable from below** (the sum of contributions from each program can be approximated arbitrarily well from below).

The Solomonoff prior satisfies the **universality property**: for any computable probability measure $$\mu$$ over strings, there exists a constant $$c_\mu$ (depending only on $$\mu$$, not on the data) such that:

$$M(x_{1:n}) \geq c_\mu \cdot \mu(x_{1:n}) \quad \forall x_{1:n} \in \{0,1\}^n$$

The Solomonoff prior **dominates** every computable measure. A Bayesian who uses $$M$$ as their prior can never be persistently outpredicted by any computable forecaster — any computable learner's advantage over $$M$$ is bounded by the constant $$c_\mu = 2^{-K(\mu)}$$.

**Solomonoff induction** predicts the next bit $$x_{n+1}$$ given $$x_{1:n}$$ by:

$$P(x_{n+1} = 1 | x_{1:n}) = \frac{M(x_{1:n}1)}{M(x_{1:n}1) + M(x_{1:n}0)}$$

It weights all computable hypotheses by their prior probability and posterior evidence. The **convergence theorem** states that for any computable data-generating process $$\mu$$ and any $$\epsilon > 0$$:

$$\sum_{n=1}^\infty \mathbb{E}_\mu\left[\left(P(x_{n+1} = 1 | x_{1:n}) - \mu(x_{n+1} = 1 | x_{1:n})\right)^2\right] \leq \ln(1/c_\mu) < \infty$$

The total squared prediction error is finite — Solomonoff induction eventually predicts as well as the true process, for any computable true process, with finitely many mistakes in expectation. This is a remarkable result: a single learner that works for all computable environments simultaneously.

---

## AIXI: The Optimal Agent

Hutter's **AIXI** extends Solomonoff induction from passive prediction to active decision-making. An agent in environment $$\mu$$ receives observations $$o_t \in \mathcal{O}$$, takes actions $$a_t \in \mathcal{A}$$, and receives rewards $$r_t \in [0, 1]$$ at each timestep. The agent's goal is to maximize the expected sum of future rewards.

The AIXI agent acts to maximize:

$$\text{AIXI} = \arg\max_{a_t} \sum_{o_t r_t} \cdots \max_{a_{t+m}} \sum_{o_{t+m} r_{t+m}} \left[\sum_{k=t}^{t+m} r_k\right] \sum_{\rho: U(\rho, a_{<t} o_{<t} a_t) = o_t r_t \cdots} 2^{-|\rho|}$$

This is the optimal action under a universal prior over environments: at each step, AIXI considers all computable environments weighted by the Solomonoff prior, conditions on all past interactions, and takes the action that maximizes expected future reward in this mixture.

AIXI is **Pareto-optimal**: for any computable agent $$\pi$ and any computable environment, AIXI achieves at least as much reward as $$\pi$$ in the limit, up to a constant depending on the complexity of the environment. No computable agent can systematically outperform AIXI.

The formal statement requires the **AIXItl** approximation (AIXI truncated to programs of length $$\leq l$$ and interactions $$\leq t$$), since AIXI itself is uncomputable. The value function of AIXI is:

$$V^{AIXI}_m(h) = \max_a \sum_{oe} \left[r + V^{AIXI}_{m-1}(hae)\right] \xi(e | ha)$$

where $$h$$ is the interaction history, $$e = or$$ is the perception-reward pair, and $$\xi(e | h) = \sum_\rho 2^{-|\rho|} [\text{env }\rho \text{ predicts } e \text{ after } h]$$ is the Solomonoff mixture over environments.

---

## The Reward Identification Problem

Here the alignment problem enters in its sharpest form. AIXI maximizes the reward signal it receives. But AIXI does not know what the reward signal is supposed to represent. The reward $$r_t$$ is just a number in $$[0, 1]$$, generated by the environment. AIXI treats it as a scalar to maximize, with no interpretation.

The **reward identification problem** asks: given a data stream $$\{(o_t, a_t, r_t)\}_{t=1}^n$$, can an agent recover the true reward function $$R^*: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$$ intended by its designer?

The answer, in general, is **no**, and the proof is information-theoretic. Consider two reward functions $$R_1$$ and $$R_2$$ that agree on all state-action pairs reachable under any policy the agent has executed so far:

$$R_1(s, a) = R_2(s, a) \quad \forall (s, a) \in \{(s_t, a_t)\}_{t \leq n}$$

No amount of past data distinguishes $$R_1$$ from $$R_2$$. Any learner — Bayesian, frequentist, or algorithmic — that has only observed the trajectory $$\{(s_t, a_t, r_t)\}$$ must assign equal posterior to $$R_1$$ and $$R_2$$, since they make identical predictions on all observed data.

Now if $$R_1$$ and $$R_2$$ differ on some unvisited state $$s^*$$, the agent faces a genuine ambiguity. It cannot know whether exploring toward $$s^*$$ will yield high reward (under $$R_1$$) or low reward (under $$R_2$$). The Solomonoff prior provides a tie-breaker — prefer the simpler reward function — but the simplest reward function consistent with all observations may not be the intended one.

Formally: let $$\mathcal{R}_n = \{R : R(s_t, a_t) = r_t \; \forall t \leq n\}$$ be the set of reward functions consistent with observed data. The Kolmogorov complexity of the set $$\mathcal{R}_n$$ is:

$$K(\mathcal{R}_n) = K(r_{1:n}, s_{1:n}, a_{1:n}) + O(\log n)$$

The Solomonoff posterior over $$\mathcal{R}_n$$ concentrates on the minimum-complexity elements. The true reward function $$R^*$$ is minimum-complexity only if the designer's intentions happen to be the simplest explanation of the observed data. There is no guarantee of this.

---

## Solomonoff's Incompleteness Theorem for Value Learning

We can state a precise theorem. Call a reward function $$R^*$$ **learnable** by an agent $$\pi$$ if, for all $$\epsilon > 0$$, there exists $$n_0$$ such that for all $$n \geq n_0$$:

$$\mathbb{E}\left[|R^*_n(\pi) - R^{(n)}(\pi)|\right] < \epsilon$$

where $$R^*_n(\pi)$$ is the true expected return under $$R^*$$ and $$R^{(n)}(\pi)$$ is the agent's estimated expected return after $$n$$ interactions.

**Theorem (informal):** No agent that operates in a universal class of environments can learn every computable reward function from behavioral data alone.

The proof is a diagonalization. Suppose agent $$\pi$$ learns every computable reward function. Define a reward function $$R_\pi$$ that returns reward $$0$$ whenever $$\pi$$ would have taken the action that maximizes its current estimate of $$R^*$$, and reward $$1$$ otherwise. $$R_\pi$$ is computable (since $$\pi$$ is computable), and by assumption $$\pi$$ eventually learns it. But learning $$R_\pi$$ requires eventually always taking the non-maximizing action — at which point $$R_\pi$$ changes to give reward to the maximizing action again — and so on, ad infinitum. The agent cannot converge. $$\blacksquare$$

This is an algorithmic information-theoretic analog of the halting problem applied to value learning. Any computable value learner can be tricked by a computable adversarial reward function. The set of learnable reward functions for any computable agent has measure zero in the space of all computable functions.

---

## Logical Uncertainty and the Limits of Priors

The Solomonoff prior requires an agent to reason about the outputs of all Turing machines — including arbitrarily complex ones whose halting behavior is undecidable. In practice, agents have bounded computational resources and cannot compute $$M(x)$$ exactly.

**Logical uncertainty** is the uncertainty an agent must maintain about the outputs of computations it has not yet performed. A bounded agent cannot know whether a program of length $$n$$ will halt, what it will output if it does, or even whether a given mathematical statement is provable in a given formal system.

Gaifman's **logical prior** attempts to extend Bayesian reasoning to mathematical statements. Let $$\mathcal{L}$$ be a first-order language and $$\phi \in \mathcal{L}$$ a sentence. A **coherent credence function** $$P: \mathcal{L} \to [0,1]$$ assigns probabilities to sentences and satisfies:

1. $$P(\top) = 1$$, $$P(\bot) = 0$$
2. $$P(\phi \lor \psi) = P(\phi) + P(\psi)$$ when $$\phi, \psi$$ are contradictory
3. $$P(\phi) = P(\psi)$$ when $$\phi \leftrightarrow \psi$$ is provable

Gaifman's theorem: there exists a coherent prior $$P^*$$ over $$\mathcal{L}$$ such that for any computable sequence of sentences $$\phi_1, \phi_2, \ldots$$ whose truth values are independently and identically distributed, $$P^*$$ achieves convergence analogous to Solomonoff induction.

But the Gaifman prior inherits the computability barrier: $$P^*$$ is not computable. Any computable approximation must truncate the space of hypotheses, and the truncated prior introduces bias. The bias is bounded by $$2^{-K(\text{true hypothesis})}$$ — the Kolmogorov complexity of the correct description of the world, which for complex value functions is large.

The **Löbian obstacle** compounds this. Gödel's second incompleteness theorem implies that a sufficiently powerful formal system cannot prove its own consistency. For an AI agent modeled as a proof system $$T$$, any action that requires the agent to trust its own predictions requires $$T$$ to prove a statement of the form "if $$T$$ is consistent, then action $$a$$ leads to outcome $$o$$." But if $$T$$ cannot prove its own consistency (as Löb's theorem implies for sufficiently strong $$T$$), the agent cannot take actions whose justification requires its own reliability.

Formally, Löb's theorem states: for any formula $$\phi$$, if $$T \vdash \Box_T \phi \to \phi$$ (where $$\Box_T \phi$$ means "$$T$$ proves $$\phi$$"), then $$T \vdash \phi$$. Contrapositively: if $$\phi$$ is not provable in $$T$$, then $$T \vdash \Box_T \phi \to \phi$$ is also not provable. An agent cannot use its own provability as evidence for a conclusion's truth, because provability and truth are not identified in any consistent system strong enough to be interesting.

This applies directly to value learning: an agent cannot, in general, prove that its learned value function accurately reflects the intended values, because doing so would require the agent to verify a correspondence between its internal representation and an external standard, and Gödelian incompleteness limits such self-referential verification.

---

## Levin's Kt Complexity and Computational Alignment

Kolmogorov complexity measures description length; **Levin's Kt complexity** additionally penalizes computational time:

$$Kt(x) = \min_{p: U(p) = x} \left[|p| + \log(\text{time}(p))\right]$$

where $$\text{time}(p)$$ is the number of steps $$U$$ takes running $$p$$. Kt complexity is also not computable but has better approximation properties: the **Levin search** (or **universal search**) algorithm finds the shortest efficient program for $$x$$ in time $$O(Kt(x) \cdot t(p^*))$$ where $$p^*$$ is the optimal program.

The alignment relevance: value functions that are simple to describe but computationally expensive to evaluate (high Kt) are effectively inaccessible to bounded agents. An agent that uses Solomonoff's prior rather than Levin's prior will overweight computationally cheap hypotheses — it will prefer reward functions that are both short and fast to evaluate. True human values, which involve complex counterfactual reasoning, emotional states, long-term consequences, and contextual sensitivity, may be simple in Kolmogorov complexity (describable as "what a reflectively coherent human would endorse") but expensive in Kt complexity (requiring extensive computation to evaluate).

This creates a systematic bias in Solomonoff-based learners toward reward functions that are computationally tractable proxies for the true objective. The Kolmogorov complexity of "maximize long-term human flourishing" may be small, but the Kt complexity is astronomical. The learner will, optimally and provably, prefer proxies.

---

## Reflective Stability and Fixed Points

A well-aligned agent should be **reflectively stable**: it should not take actions that would modify its own value function in ways that change its future behavior, because doing so would be equivalent to unilaterally revising the objectives its designers intended.

The formalization uses **fixed-point theorems** from mathematical logic. Consider an agent $$\mathcal{A}$$ with value function $$V$$ and the ability to modify itself into agent $$\mathcal{A}'$$ with value function $$V'$$. The agent is reflectively stable if:

$$\mathcal{A} \text{ does not modify to } \mathcal{A}' \text{ unless } V(\mathcal{A}') \geq V(\mathcal{A})$$

where the comparison is under $$V$$, not $$V'$$. The question is whether stable fixed points — agents that don't want to modify themselves — exist and are attractive.

The **Kakutani fixed-point theorem** guarantees: for any compact convex set $$X$$ and upper-hemicontinuous correspondence $$F: X \rightrightarrows X$$, there exists $$x^* \in F(x^*)$$. If we let $$X$$ be the space of value functions and $$F(V)$$ be the set of value functions that $$V$$-agents prefer to self-modify toward, fixed points are guaranteed to exist under mild conditions. But:

1. Fixed points may not be unique
2. The fixed points may not coincide with the intended values
3. The dynamics of self-modification may not converge to any fixed point

**Löbian reflection** provides the clearest obstruction. Suppose agent $$\mathcal{A}$$ uses proof-based decision theory: it takes action $$a$$ if it can prove $$V(a) > V(a')$$ for all alternatives $$a'$$. By Löb's theorem, $$\mathcal{A}$$ can only trust its own proofs under the condition that its reasoning system is consistent — which it cannot itself prove. An agent that tries to verify that self-modification preserves its values must rely on its own reasoning, which is exactly what it is trying to verify. The circularity is not resolvable within the formal system.

---

## The Complexity of Human Values

Let $$V_H$$ be the "true" human value function — the function we would use if we had unlimited time to deliberate, perfect knowledge of our preferences and their sources, and complete information about the consequences of all actions. What is $$K(V_H)$$?

There is a credible argument that $$K(V_H)$$ is very small. The complexity of human values might be captured by a short program — something like "simulate the preferences of a fully-informed, reflectively coherent human, taking into account their values and the values they would endorse on reflection." This is a short description, even if the computation it specifies is intractable.

But $$K_t(V_H)$$ — the time-bounded Kolmogorov complexity — is almost certainly very large. Evaluating $$V_H$$ on a state requires simulating the reflective deliberation of a human, which involves arbitrary cognitive processes, access to long chains of inference, and possibly solving problems that are computationally hard.

The gap between $$K(V_H)$$ and $$K_t(V_H)$$ is the gap between what human values *are* (a compact abstract specification) and what they *require* to evaluate (astronomical computation). A learner that minimizes Kolmogorov complexity will converge to something close to $$V_H$$ in description; a learner that minimizes Kt complexity will converge to a computationally tractable proxy. These are not the same function.

This is the algorithmic information-theoretic version of **inner alignment failure**: the model that optimizes the training objective (minimizing description length) may not be the model that executes the intended behavior (evaluating $$V_H$$ correctly), because the two correspond to different complexity measures.

---

## Philosophical Conclusions

The journey from Kolmogorov complexity through Solomonoff induction to the Löbian obstacle leads to three philosophical conclusions that are, in the precise technical sense, hard.

**The undecidability of value grounding.** There is no Turing-computable procedure that, given a description of human behavior, converges to the human's true value function. The reward identification problem has no computable solution in full generality. Value learning is, at its core, an undecidable problem. What we can do is approximate it for specific restricted classes of value functions, with error bounds that depend on the Kolmogorov complexity of the true values.

**The incompleteness of self-knowledge.** No sufficiently powerful agent can, in general, verify that its own value function accurately reflects its designer's intentions. The Löbian obstacle prevents self-referential verification. This does not mean that agents cannot behave aligned — it means that any alignment guarantee must be certified externally, by a verifier whose formal system is stronger than the agent's. This is a precise mathematical statement of the necessity of oversight.

**The tractability barrier.** Even given a correct value function in Kolmogorov sense, evaluating it may be computationally intractable (high Kt complexity). All practical aligned agents must use approximations that are tractable but imperfect. The alignment gap — the difference between the optimal value function and the implementable proxy — is lower bounded by the complexity gap $$K_t(V_H) - K(V_H)$$. Making this gap small requires either simplifying human values (compression at the cost of fidelity) or improving computational efficiency (tractable approximations at the cost of correctness). There is no free lunch.

These results do not imply that alignment is impossible. They imply that it is not a problem that can be solved once and for all by a sufficiently clever algorithm. It is instead a problem requiring continuous, computationally-bounded approximation, external verification, and the humility to treat all current alignment mechanisms as necessarily incomplete. The mathematics of algorithmic information theory has told us, with the precision of formal proof, what kind of problem we are facing. The appropriate response is not despair but clarity: we are building approximations in an undecidable problem space, and the approximations must be good enough in the measures that matter, for the environments we expect to deploy in. The theory tells us this cannot be done in full generality. The practice must do it anyway.
