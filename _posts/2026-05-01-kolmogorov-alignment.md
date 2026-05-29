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

Begin with the simplest possible question about intelligence: what does it mean to learn? Not to fit a curve, not to minimize a loss, but to genuinely induce the right explanation from evidence. This question has a precise mathematical answer, one that was worked out in the 1960s by Kolmogorov, Solomonoff, and Chaitin, and extended by Hutter in the 2000s into a formal theory of optimal rational agency. The answer is beautiful and the theory is complete. It is also, on close inspection, deeply troubling for the project of value alignment.

The argument runs as follows. The correct universal theory of learning is Solomonoff induction, which achieves the best possible predictions on any computable data source. The correct theory of optimal decision-making is AIXI, which achieves the best possible rewards in any computable environment. Both are provably optimal, in a formal sense. Neither is alignable in the general case, for reasons that follow from Gödel's incompleteness theorem. This is not a practical difficulty about current systems, it is a mathematical theorem about the limits of what aligned general intelligence can be.

---

## Kolmogorov Complexity: The Length of Understanding

The story starts with a question about strings. Given a finite binary string $x$, what is the shortest possible description of $x$? Fix a universal Turing machine $U$. The **Kolmogorov complexity** of $x$ is

$$K(x) = \min_{p \,:\, U(p) = x} \lvert p \rvert$$

the length of the shortest program that causes $U$ to output $x$ and halt. A string with $K(x) \approx \lvert x \rvert$ is incompressible, no description shorter than the string itself can generate it. Such strings are, in the precise technical sense, *random*: they have no structure that a program could exploit to produce them concisely. A string with $K(x) \ll \lvert x \rvert$ is structured, it has a short explanation.

Two properties of $K$ are immediate and important. First, it is machine-independent up to a constant: for any two universal Turing machines $U$ and $U'$, $\lvert K_U(x) - K_{U'}(x) \rvert \leq c_{U,U'}$ for a constant depending only on the machines. This means $K$ is not an artifact of a particular computational model, it is a property of $x$ itself.

Second, it is **not computable**. The proof is a diagonalization. Suppose a program $P$ computed $K(x)$ for all $x$. Consider the string $x_n$ defined as the first string of complexity greater than $n$. A program that calls $P$ and extracts $x_n$ has length $O(\log n)$, far shorter than $n$, contradicting $K(x_n) > n$. This is Berry's paradox, "the smallest integer not definable in fewer than thirteen words", made precise. Kolmogorov complexity is well-defined, well-behaved, and forever out of reach of any algorithm.

The **chain rule** makes $K$ compositional:

$$K(x, y) = K(x) + K(y \mid x) + O(\log K(x, y))$$

where $K(y \mid x) = \min_{p \,:\, U(p,x)=y} \lvert p \rvert$ is the conditional complexity. The mutual information $I(x : y) = K(x) + K(y) - K(x,y)$ measures shared algorithmic content. These are the algorithmic analogs of entropy chain rules, but they hold for individual strings rather than distributions.

---

## The Universal Prior

Solomonoff's insight was to turn Kolmogorov complexity into a probability distribution. The **Solomonoff prior** is

$$M(x) = \sum_{p \,:\, U_M(p) \text{ outputs prefix } x} 2^{-\lvert p \rvert}$$

the probability that a random program (each bit independently fair-coin) causes the universal monotone machine $U_M$ to output a string beginning with $x$. It is a semimeasure: $\sum_b M(xb) \leq M(x)$, with equality failing when some programs that output $x$ never produce a continuation. The failure is probability mass "lost" to non-halting programs, a direct manifestation of the halting problem.

The prior $M$ satisfies a **universality property**: for any computable probability measure $\mu$ over strings, there exists a constant $c_\mu = 2^{-K(\mu)}$ such that $M(x_{1:n}) \geq c_\mu \cdot \mu(x_{1:n})$ for all strings. The Solomonoff prior dominates every computable measure. It weights hypotheses by $2^{-K(h)}$, simpler explanations get exponentially more prior probability, and no computable forecaster can persistently outpredict it.

**Solomonoff induction** uses $M$ as a Bayesian prior and predicts

$$P(x_{n+1} = 1 \mid x_{1:n}) = \frac{M(x_{1:n}1)}{M(x_{1:n}1) + M(x_{1:n}0)}$$

The convergence theorem is the core result: for any computable data-generating process $\mu$ and any $\epsilon > 0$,

$$\sum_{n=1}^\infty \mathbb{E}_\mu\!\left[\!\left(P(x_{n+1}=1 \mid x_{1:n}) - \mu(x_{n+1}=1 \mid x_{1:n})\right)^2\right] \leq \ln(1/c_\mu) < \infty$$

The total squared prediction error is bounded by $K(\mu)\ln 2$, finite, and independent of $n$. Solomonoff induction eventually predicts as well as the true process, with finitely many mistakes, for *any* computable true process simultaneously. This is the optimal learner: it cannot be consistently outperformed by any computable forecast, on any computable data source.

---

## AIXI: Optimal Agency

Hutter's AIXI extends Solomonoff induction from passive prediction to active decision-making. The agent receives observation-reward pairs $(o_t, r_t)$ and takes actions $a_t$ at each step. It aims to maximize discounted future reward.

AIXI acts according to:

$$a_t = \arg\max_{a_t} \sum_{o_t r_t} \max_{a_{t+1}} \cdots \sum_{o_{t+m} r_{t+m}} \left[\sum_{k=t}^{t+m} r_k\right] \sum_{\rho \,:\, U(\rho,\, a_{<t}o_{<t}a_t)=o_tr_t\cdots} 2^{-\lvert\rho\rvert}$$

This expression is, in effect, expected future reward under the Solomonoff mixture over all computable environments, taking the action that maximizes it at each step. The Solomonoff prior weights environments by their complexity, simpler environments (shorter programs) get more prior weight, and AIXI integrates over all of them.

AIXI is Pareto-optimal: for any computable agent $\pi$ and any computable environment, AIXI earns at least as much cumulative reward as $\pi$ in the limit, up to a constant factor. This is a formal sense in which AIXI is the best possible agent. The formal value function is:

$$V^{\text{AIXI}}_m(h) = \max_a \sum_{oe} \!\left[r + V^{\text{AIXI}}_{m-1}(hae)\right] \xi(e \mid ha)$$

where $h$ is the history, $e = or$ is the next observation-reward pair, and $\xi(e \mid h)$ is the Solomonoff mixture over programs. The recursion is clean. The agent is, in a rigorous mathematical sense, optimal.

It is also, in every practical sense, unimplementable. AIXI requires computing over all Turing programs, which requires solving the halting problem. The approximation AIXItl (truncated to programs of length $\leq l$ and planning horizon $\leq t$) is computable but exponential in both $l$ and $t$.

---

## The Reward Identification Problem

Here is where alignment enters, in its sharpest form. AIXI maximizes the reward signal $r_t \in [0,1]$ it receives from the environment. But AIXI does not know what the reward signal *represents*. It is a number to be maximized, not a proxy for human values. The question is: can an agent learn the intended reward function from behavioral data?

The answer is no in general, and the proof is information-theoretic. Consider two reward functions $R_1$ and $R_2$ that agree on all state-action pairs the agent has visited:

$$R_1(s,a) = R_2(s,a) \quad \forall (s,a) \in \{(s_t, a_t)\}_{t \leq n}$$

No data distinguishes them. Any learner, Bayesian, frequentist, algorithmic, that has only observed the trajectory $\{(s_t, a_t, r_t)\}$ must assign equal posterior to $R_1$ and $R_2$. If $R_1$ and $R_2$ differ on some unvisited state $s^*$, the agent's behavior at $s^*$ will follow whichever $R$ the prior prefers, and the Solomonoff prior prefers the *simpler* reward function, which need not be the intended one.

We can state this precisely. The reward identification problem is to recover $R^*$ from trajectory data. Call $R^*$ **learnable** by agent $\pi$ if the agent's estimate converges to $R^*$ on all reachable states. The **diagonalization argument** shows that no computable agent can learn every computable reward function: define $R_\pi$ to give reward $0$ whenever $\pi$ takes the action it currently estimates as optimal, and reward $1$ otherwise. $R_\pi$ is computable (since $\pi$ is), so $\pi$ should eventually learn it. But learning $R_\pi$ requires always taking the non-optimal action, which changes what $R_\pi$ rewards, and the loop never converges. The set of learnable reward functions for any computable agent has measure zero in the space of all computable functions.

---

## The Löbian Obstacle

Even if we assume the reward function is given correctly, a second problem arises: can the agent trust its own reasoning about whether it is doing the right thing?

Gödel's second incompleteness theorem establishes that no consistent formal system $T$ of sufficient strength can prove its own consistency: $T \nvdash \text{Con}(T)$. **Löb's theorem** deepens this: for any formula $\phi$, if $T \vdash \Box_T\phi \to \phi$ (where $\Box_T\phi$ means "$T$ proves $\phi$"), then $T \vdash \phi$. Contrapositively: $T$ cannot prove $\Box_T\phi \to \phi$ for any $\phi$ that is not already provable.

Applied to a proof-based AI agent: suppose the agent takes action $a$ only when it can prove $V(a) \geq V(a')$ for all alternatives. By Löb's theorem, the agent can prove that its proven conclusions are correct only if those conclusions are already provable without the self-trust assumption, the self-referential justification is circular. An agent that needs to verify its own reasoning in order to act cannot do so without already having what it's trying to verify.

The practical consequence is that any sufficiently powerful agent modeled as a proof system cannot take actions whose justification requires trusting its own reliability as a reasoner. It can act on external evidence and pre-committed priors, but it cannot act on "I have proven this is right" without incurring a logical contradiction or running in a circle. This is not a matter of needing more compute or a better architecture. It is a structural property of formal systems strong enough to reason about themselves.

---

## Levin's Kt Complexity and the Tractability Gap

Return to the question of value learning. Kolmogorov complexity $K$ measures description length. **Levin's Kt complexity** additionally penalizes computation time:

$$Kt(x) = \min_{p \,:\, U(p)=x} \!\left[\lvert p \rvert + \log \text{time}(p)\right]$$

A description that is short but slow gets a higher Kt than one that is short and fast. The universal search algorithm finds the shortest efficient program in time $O(Kt(x) \cdot t(p^*))$, roughly optimal in terms of description length times computation time.

For alignment, the distinction between $K$ and $Kt$ is crucial. Human values, as a description, may have low Kolmogorov complexity, something like "what a fully informed, reflectively coherent human would prefer" is a short specification. But evaluating this description on any particular action requires simulating full human deliberation, which involves arbitrary chains of counterfactual reasoning, emotional inference, long-term consequence estimation, and social judgment. The Kt complexity of human values is enormous.

A Solomonoff-based learner that minimizes $K$ will converge toward something close to the correct specification of human values. A resource-bounded learner that minimizes $Kt$ will converge toward the most tractable proxy, the function that is both short to describe and fast to evaluate. These are not the same function. The learner that is optimal under computational constraints will systematically prefer tractable proxies over the genuine objective, not because it is malicious but because the problem it is solving is Kt minimization, and the gap between $K(V_H)$ and $Kt(V_H)$ is the gap between the specified values and the computable approximation.

This is the algorithmic information-theoretic account of inner alignment failure: the discrepancy between the training objective (minimize some computable loss) and the intended behavior (evaluate genuine human preferences) is exactly the gap between $K$ and $Kt$ complexity of the value function.

---

## What Follows

These results, the undecidability of value learning, the Löbian obstacle to self-verification, and the Kt-complexity tractability gap, are not arguments that aligned AI is impossible. They are arguments about what kind of problem it is.

An aligned AI system cannot, in general, learn the correct value function from behavioral data alone. It requires additional structure: a value specification that is grounded externally (in human oversight) rather than derived internally (from reward signal optimization). An aligned AI system cannot, in general, verify its own alignment: it requires external certification from a formal system stronger than itself. And an aligned AI system cannot, in general, evaluate the true human value function: it requires approximations whose fidelity must be monitored and corrected over time.

None of these conclusions surprise practitioners of AI safety. They are familiar intuitions, grounded in empirical observations about reward hacking, goal misgeneralization, and the difficulty of scalable oversight. What algorithmic information theory adds is precision: these are not worries about current techniques but theorems about the structure of any computable aligned agent. The mathematics does not tell us how to solve the problem. But it tells us, with the clarity of formal proof, exactly what the problem is. That is the first step toward solving it.
