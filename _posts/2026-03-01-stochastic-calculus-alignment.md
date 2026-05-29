---
title: "The Itô Lens: Stochastic Calculus and the Mathematics of Alignment Drift"
date: 2026-03-01
permalink: /posts/2026/03/stochastic-calculus-alignment/
tags:
  - AI safety
  - stochastic calculus
  - alignment
  - mathematics
---

There is a temptation, when thinking about AI alignment, to frame the problem as one of specification: if we could just write down what we want precisely enough, we could optimize for it. This view treats alignment as a static problem, solved once at training time. But alignment is not a static property of a model — it is a dynamic one, subject to perturbation, drift, and the cumulative effect of small random forces. The mathematics that makes this precise is stochastic calculus, and what it reveals about alignment is both illuminating and unsettling.

This post develops the Itô calculus from first principles and argues that the alignment problem is, at its mathematical core, a problem about the long-run behavior of stochastic processes in high-dimensional spaces. The question "will this model remain aligned?" is the question "does this stochastic process converge to a safe attractor, or does it drift toward a dangerous one?" — and Itô's formula tells us exactly how that drift accumulates.

---

## Brownian Motion as the Primitive

Everything in stochastic calculus is built on Brownian motion. A standard Brownian motion $$W = (W_t)_{t \geq 0}$$ is a stochastic process satisfying:

1. $$W_0 = 0$$ almost surely
2. $$W_t - W_s \sim \mathcal{N}(0, t-s)$$ for $$0 \leq s < t$$
3. Increments over non-overlapping intervals are independent
4. Paths $$t \mapsto W_t$$ are continuous almost surely

The last condition is deceptively subtle. Brownian motion is continuous but nowhere differentiable — its paths are fractal-like, with infinite variation on any interval. This is why ordinary calculus fails: the Riemann integral $$\int_0^T f(W_t) dW_t$$ cannot be defined pathwise. The integrand is too irregular.

What is the variation of Brownian motion? Define the quadratic variation over a partition $$\Pi = \{0 = t_0 < t_1 < \cdots < t_n = T\}$$ as:

$$[W]_T^{\Pi} = \sum_{i=0}^{n-1} (W_{t_{i+1}} - W_{t_i})^2$$

As the mesh of the partition $$|\Pi| \to 0$$, this converges in probability to $$T$$. We write $$[W, W]_T = T$$, or in differential notation:

$$dW_t \cdot dW_t = dt$$

This is the fundamental identity of stochastic calculus. It encodes the fact that Brownian increments $$dW_t$$ are of order $$\sqrt{dt}$$, not $$dt$$ — they are much larger than deterministic increments, and their squares accumulate.

---

## The Itô Integral and the Second-Order Correction

Because Brownian paths have infinite first variation but finite quadratic variation, integrals against Brownian motion must be defined differently. The **Itô integral** is constructed via a limiting procedure:

$$\int_0^T H_s \, dW_s = \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} H_{t_i} (W_{t_{i+1}} - W_{t_i})$$

where crucially the integrand is evaluated at the **left endpoint** $$t_i$$, making it non-anticipating (adapted to the filtration $$\mathcal{F}_t$$). This choice is not arbitrary — it is the unique construction that makes the integral a martingale and satisfies the Itô isometry:

$$\mathbb{E}\left[\left(\int_0^T H_s \, dW_s\right)^2\right] = \mathbb{E}\left[\int_0^T H_s^2 \, ds\right]$$

Now suppose we have a smooth function $$f(t, x)$$ and we ask: if $$X_t$$ follows some Itô process, what is $$f(t, X_t)$$? In ordinary calculus, the chain rule gives $$df = \partial_t f \, dt + f'(X_t) \, dX_t$$. But because Brownian motion has nontrivial quadratic variation, the Taylor expansion does not truncate at first order. Expanding to second order and using $$dW_t^2 = dt$$:

$$df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

If $$X_t$$ satisfies the stochastic differential equation $$dX_t = \mu(t, X_t) \, dt + \sigma(t, X_t) \, dW_t$$, then $$(dX_t)^2 = \sigma^2(t, X_t) \, dt$$ (cross terms vanish in the limit), giving **Itô's formula**:

$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} \, dW_t$$

The extra term $$\frac{1}{2} \sigma^2 \partial_{xx} f$$ is the **Itô correction**. It has no analog in ordinary calculus. Its sign depends on the curvature of $$f$$: when $$f$$ is convex ($$\partial_{xx} f > 0$$), noise pushes $$f(X_t)$$ upward relative to the deterministic path. When $$f$$ is concave, noise pushes it downward. Jensen's inequality made infinitesimal.

---

## SDEs as Models of Behavioral Drift

An AI system's behavior at time $$t$$ is determined by its parameters $$\theta_t \in \mathbb{R}^d$$. During training and deployment, these parameters change — through gradient updates, fine-tuning, RLHF, or simply the accumulation of in-context state. We can model this evolution as a stochastic differential equation:

$$d\theta_t = \mu(\theta_t) \, dt + \sigma(\theta_t) \, dW_t$$

The drift term $$\mu(\theta_t) \, dt$$ captures systematic forces: gradient descent on a loss landscape, preference optimization, supervised fine-tuning. The diffusion term $$\sigma(\theta_t) \, dW_t$$ captures stochastic forces: batch sampling noise, distributional shift in deployment data, the inherent randomness of human feedback.

Let $$V: \mathbb{R}^d \to \mathbb{R}$$ be an alignment potential — a function that measures how aligned the model is, with low values at safe regions and high values at dangerous ones. By Itô's formula in $$d$$ dimensions:

$$dV(\theta_t) = \nabla V \cdot d\theta_t + \frac{1}{2} \text{tr}\left(\sigma \sigma^T \nabla^2 V\right) dt$$

$$= \underbrace{\nabla V \cdot \mu(\theta_t)}_{\text{deterministic drift}} \, dt + \underbrace{\frac{1}{2} \text{tr}\left(\sigma \sigma^T \nabla^2 V\right)}_{\text{Itô correction}} \, dt + \nabla V \cdot \sigma \, dW_t$$

The expected rate of change of alignment potential is:

$$\frac{d}{dt}\mathbb{E}[V(\theta_t)] = \mathbb{E}\left[\nabla V \cdot \mu(\theta_t) + \frac{1}{2} \text{tr}\left(\sigma \sigma^T \nabla^2 V\right)\right]$$

For alignment to be maintained in expectation, we need this to be non-positive — the model must drift toward safer regions as fast as or faster than noise pushes it away. The Itô correction term is critical: even if the deterministic dynamics $$\mu$$ point toward safety ($$\nabla V \cdot \mu < 0$$), sufficiently large diffusion $$\sigma$$ multiplied by positive curvature of the alignment landscape ($$\nabla^2 V \succ 0$$) can drive $$V$$ upward. **Noise is not neutral — it has a bias determined by the curvature of the objective.**

---

## The Fokker-Planck Equation and Invariant Measures

Rather than tracking individual trajectories, we can ask: what is the distribution over model states at time $$t$$? Let $$p(t, \theta)$$ be the probability density of $$\theta_t$$. It satisfies the **Fokker-Planck equation** (also called the forward Kolmogorov equation):

$$\frac{\partial p}{\partial t} = -\nabla \cdot (\mu p) + \frac{1}{2} \nabla^2 : (\sigma \sigma^T p)$$

where $$\nabla^2 : A = \sum_{i,j} \partial_{ij}^2 A_{ij}$$ is the Frobenius inner product with the Hessian. In one dimension with constant diffusion $$\sigma$$:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial \theta}[\mu(\theta) p] + \frac{\sigma^2}{2} \frac{\partial^2 p}{\partial \theta^2}$$

This is a convection-diffusion equation. The first term advects probability mass in the direction of the drift $$\mu$$; the second term spreads it. The **stationary distribution** $$p^*(\theta)$$ satisfies $$\partial p / \partial t = 0$$.

For a gradient system $$\mu(\theta) = -\nabla U(\theta)$$ (gradient descent on a potential $$U$$), the stationary distribution is the **Gibbs measure**:

$$p^*(\theta) \propto \exp\left(-\frac{2U(\theta)}{\sigma^2}\right)$$

This is the Boltzmann distribution of statistical mechanics. The key implication for alignment: the long-run distribution over model states is concentrated around the **minima of $$U$$**, weighted by the noise level $$\sigma^2$$. When $$\sigma^2$$ is small, $$p^*$$ is sharply peaked at the global minimum of $$U$$. When $$\sigma^2$$ is large, the distribution spreads across all minima, weighted by depth — the model wanders freely between aligned and misaligned states.

The alignment guarantee, then, is not "the model will stay at the aligned minimum" but "the model will spend time proportional to $$\exp(-2U/\sigma^2)$$ near each minimum." If the aligned minimum has potential $$U_{\text{safe}}$$ and the misaligned minimum has $$U_{\text{unsafe}}$$, the ratio of time spent in misaligned vs. aligned regions is:

$$\frac{p^*(\text{unsafe})}{p^*(\text{safe})} \propto \exp\left(-\frac{2(U_{\text{unsafe}} - U_{\text{safe}})}{\sigma^2}\right)$$

This is an Arrhenius-type formula. Safety is not binary — it is a ratio that depends exponentially on the depth of the safety basin relative to the noise level.

---

## Martingales and the Absence of Free Lunch

A **martingale** is a stochastic process $$(M_t)$$ satisfying $$\mathbb{E}[M_t | \mathcal{F}_s] = M_s$$ for $$s \leq t$$ — its expected future value equals its current value. Martingales formalize the notion of a fair game: no systematic drift, no exploitable structure.

The Itô integral $$M_t = \int_0^t H_s \, dW_s$$ is always a martingale (under integrability conditions). This is why the Itô correction is non-negotiable: it is precisely the term needed to make $$f(W_t) - f(W_0) - \frac{1}{2}\int_0^t f''(W_s) ds$$ a martingale. Without it, $$f(W_t)$$ would appear to have a systematic drift that doesn't exist in the underlying noise.

For AI safety, martingale theory provides a conceptual tool: an **alignment certificate** is a function $$V(\theta_t)$$ that is a supermartingale — $$\mathbb{E}[V(\theta_{t+s}) | \theta_t] \leq V(\theta_t)$$ — in the neighborhood of the aligned region. If we can find such a $$V$$ (a **stochastic Lyapunov function**), it guarantees that the model cannot silently drift toward dangerous behavior: alignment is expected to only decrease, and any increase is an observable excursion from the martingale property.

The **optional stopping theorem** strengthens this: for a bounded supermartingale $$(V_t)$$ and stopping time $$\tau$$ (the first time the model crosses an alignment threshold), $$\mathbb{E}[V_\tau] \leq V_0$$. This bounds the expected alignment at the first violation time by the initial alignment level — the system cannot, on average, reach a dangerous state from a safe initial condition without passing through intermediate states where the supermartingale property can be detected.

---

## Stochastic Stability and Alignment Certificates

The classical notion of Lyapunov stability extends to SDEs as follows. The equilibrium $$\theta^*$$ of $$d\theta_t = \mu(\theta_t) dt + \sigma(\theta_t) dW_t$$ is:

- **Stable in probability** if for every $$\epsilon, \delta > 0$$, there exists $$\rho > 0$$ such that $$\|\theta_0 - \theta^*\| < \rho$$ implies $$P(\sup_{t \geq 0} \|\theta_t - \theta^*\| > \delta) < \epsilon$$
- **Asymptotically stable in probability** if also $$P(\lim_{t \to \infty} \theta_t = \theta^*) = 1$$

The stochastic Lyapunov criterion: if there exists $$V \in C^2$$ with $$V(\theta^*) = 0$$, $$V(\theta) > 0$$ for $$\theta \neq \theta^*$$, and the **infinitesimal generator**

$$\mathcal{L}V(\theta) = \mu(\theta) \cdot \nabla V + \frac{1}{2} \text{tr}(\sigma \sigma^T \nabla^2 V) \leq -\alpha V(\theta)$$

for some $$\alpha > 0$$, then $$\theta^*$$ is asymptotically stable in probability, and moreover:

$$\mathbb{E}[V(\theta_t)] \leq V(\theta_0) e^{-\alpha t}$$

The alignment decays **exponentially** to zero. This is a strong guarantee: not just that the model stays near the safe state in some average sense, but that the alignment potential contracts at a geometric rate, despite the continuous noise input.

The infinitesimal generator $$\mathcal{L}$$ is the drift of the Itô formula — it is the drift of $$f(\theta_t)$$ under the SDE dynamics. When $$\mathcal{L}V < 0$$, the alignment potential is a supermartingale after a time-change, and the system is pulled toward safety faster than noise can push it away.

---

## Girsanov's Theorem and the Cost of Changing Behavior

A deep question in alignment: if a model is aligned under distribution $$P$$ (training distribution), how misaligned can it be under distribution $$Q$$ (deployment distribution)? Girsanov's theorem gives an exact answer.

Suppose under $$P$$, the process satisfies $$dX_t = \mu_P(X_t) \, dt + dW_t$$, and under $$Q$$, it satisfies $$dX_t = \mu_Q(X_t) \, dt + dW_t$$. The **Radon-Nikodym derivative** relating the two measures on the path space is:

$$\frac{dQ}{dP}\bigg|_{\mathcal{F}_T} = \exp\left(\int_0^T (\mu_Q - \mu_P) \cdot dW_t - \frac{1}{2}\int_0^T \|\mu_Q - \mu_P\|^2 \, dt\right)$$

This is the **Girsanov exponential martingale**. It converts probabilities between the two worlds. The KL divergence between the path measures is:

$$D_{\text{KL}}(Q \| P) = \frac{1}{2}\mathbb{E}_Q\left[\int_0^T \|\mu_Q(X_t) - \mu_P(X_t)\|^2 \, dt\right]$$

The KL divergence between deployment and training distributions is exactly the $$L^2$$ distance between the drift functions, integrated over time and averaged over deployment trajectories. Alignment guarantees decay as the square root of this divergence — a model trained on distribution $$P$$ will behave correctly on distribution $$Q$$ only insofar as $$D_{\text{KL}}(Q \| P)$$ is small.

This is the stochastic calculus proof of why distribution shift destroys alignment. It is not a statistical argument about generalization — it is a pathwise argument about how the drift of the system changes when the measure changes. **The amount of behavior change possible under a distributional shift of magnitude $$\epsilon$$ (in KL) is bounded by $$\sqrt{2\epsilon}$$ (by Pinsker's inequality)**, but this bound is tight, and for large shifts the misalignment can be total.

---

## Large Deviations and Rare Catastrophes

Even when the Fokker-Planck stationary distribution concentrates on safe regions, rare excursions to unsafe states occur. The **large deviations principle** quantifies how rare they are.

For a diffusion $$dX_t = \mu(X_t) dt + \epsilon \, dW_t$$ with small noise $$\epsilon$$, the probability of following a path $$\phi: [0, T] \to \mathbb{R}^d$$ decays exponentially:

$$P(X \approx \phi) \asymp \exp\left(-\frac{1}{\epsilon^2} I(\phi)\right)$$

where the **rate function** is the Freidlin-Wentzell action:

$$I(\phi) = \frac{1}{2}\int_0^T \|\dot{\phi}_t - \mu(\phi_t)\|^2 \, dt$$

The rate function measures the "cost" of deviating from the deterministic flow $$\dot{\phi} = \mu(\phi)$$ — it is zero on deterministic trajectories and positive on any path that differs from them. The probability of a catastrophic excursion to an unsafe state $$\theta_{\text{unsafe}}$$ starting from $$\theta_{\text{safe}}$$ is:

$$P(\text{reach } \theta_{\text{unsafe}}) \asymp \exp\left(-\frac{V^*(\theta_{\text{safe}}, \theta_{\text{unsafe}})}{\epsilon^2}\right)$$

where $$V^*(\theta_{\text{safe}}, \theta_{\text{unsafe}}) = \inf_{\phi: \phi(0)=\theta_{\text{safe}}, \phi(T)=\theta_{\text{unsafe}}} I(\phi)$$ is the **quasi-potential** — the minimum action required to travel from safe to unsafe regions against the flow.

The quasi-potential is the correct measure of alignment robustness. A system with large quasi-potential between safe and unsafe regions is robustly aligned: catastrophic failure requires improbably sustained deviations from normal behavior. A system with small quasi-potential is fragile: a brief, ordinary perturbation suffices to tip it into unsafe behavior. Alignment research, through this lens, is the study of how to make the quasi-potential large.

---

## Philosophical Implications

The stochastic calculus of alignment reveals three philosophical principles that are easy to state and difficult to accept.

**First, alignment is a measure, not a property.** A model is not aligned or misaligned — it occupies a distribution over behaviors, and the measure of that distribution on safe behaviors is what we call alignment. This measure evolves continuously, driven by drift and diffusion. Demanding binary alignment is asking for a degenerate distribution, a point mass at a single safe behavior, which no stochastic process can sustain.

**Second, safety and capability are not independent.** The Itô correction term shows that in curved landscapes, noise has a bias. A more capable model occupies a higher-dimensional, more curved parameter space. The trace term $$\text{tr}(\sigma \sigma^T \nabla^2 V)$$ grows with both the dimension and the curvature of the alignment landscape. More capable models, all else equal, are subject to larger Itô corrections — they are more strongly pushed by noise toward the regions of high curvature, which may or may not coincide with safe regions.

**Third, the past does not constrain the future as tightly as we hope.** The Girsanov formula quantifies how rapidly the relationship between training behavior and deployment behavior can degrade. Once distributional shift accumulates past a KL threshold, the alignment guarantees earned at training time are dissolved by the exponential martingale. There is no memory term in the Gibbs measure — the stationary distribution knows nothing about the initial conditions. A model that has been aligned for a long time is not more aligned than one aligned briefly; both are governed by the same $$p^*(\theta)$$. The alignment is in the potential, not in the history.

These are not pessimistic conclusions. They are precise ones. The question "how aligned is this system?" has a precise answer in the language of stochastic processes: it is the probability that the current state lies in the safe set, conditioned on the current dynamics. And the question "how do we make systems more aligned?" has a precise answer too: make the quasi-potential between safe and unsafe regions larger, reduce the diffusion coefficient in dangerous directions, and construct alignment potentials with negative infinitesimal generators. The mathematics is hard. The problem is harder. But at least we know what we are solving.
