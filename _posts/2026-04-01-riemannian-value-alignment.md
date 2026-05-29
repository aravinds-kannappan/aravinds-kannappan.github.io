---
title: "The Geometry of Value: Riemannian Manifolds, Geodesics, and Why Objectives Diverge"
date: 2026-04-01
permalink: /posts/2026/04/riemannian-value-alignment/
tags:
  - AI safety
  - differential geometry
  - alignment
  - mathematics
---

When we say a model "optimizes for the wrong objective," we usually mean something informal — it learned to approximate the training signal in ways that fail to generalize to what we actually wanted. But there is a precise mathematical statement hiding behind that informal one. The model and the human each navigate a space of values, and those spaces are not flat. They are curved. And on curved spaces, the shortest path between two points is not a straight line — it is a **geodesic**, and geodesics diverge. Understanding alignment through differential geometry transforms vague worries about specification gaming into theorems about the curvature of objective manifolds.

---

## Smooth Manifolds and the Space of Behaviors

Let $$\mathcal{M}$$ be the space of possible behavioral policies — the set of all measurable functions $$\pi: \mathcal{S} \to \Delta(\mathcal{A})$$ from states to probability distributions over actions. This space is infinite-dimensional in general, but for practical purposes we consider it as parameterized by model weights $$\theta \in \Theta \subseteq \mathbb{R}^d$$, giving a finite-dimensional manifold $$\mathcal{M}_\Theta$$.

A **smooth manifold** of dimension $$n$$ is a topological space $$\mathcal{M}$$ that is locally homeomorphic to $$\mathbb{R}^n$$, with smooth transition maps between coordinate charts. The parameter space $$\Theta$$ is itself a manifold (generically $$\mathbb{R}^d$$), but the image of $$\Theta$$ under the map $$\theta \mapsto \pi_\theta$$ in the space of policies is curved — it is a submanifold of the infinite-dimensional policy space whose geometry is determined by the architecture of the model and the geometry of the task distribution.

At each point $$\theta \in \mathcal{M}$$, the **tangent space** $$T_\theta \mathcal{M}$$ is the set of all infinitesimal displacements from $$\theta$$ — formally, equivalence classes of smooth curves through $$\theta$$. A vector $$v \in T_\theta \mathcal{M}$$ represents a direction in which parameters can be moved. The disjoint union $$T\mathcal{M} = \bigsqcup_{\theta} T_\theta \mathcal{M}$$ is the **tangent bundle**.

A **Riemannian metric** is a smooth assignment of inner products $$g_\theta: T_\theta \mathcal{M} \times T_\theta \mathcal{M} \to \mathbb{R}$$, varying smoothly with $$\theta$$. In local coordinates, it is given by a positive definite matrix $$G(\theta) = (g_{ij}(\theta))$$ where $$g_{ij} = g(\partial_i, \partial_j)$$. The metric defines lengths of curves: for a curve $$\gamma: [0,1] \to \mathcal{M}$$,

$$\text{Length}(\gamma) = \int_0^1 \sqrt{g_{\gamma(t)}(\dot\gamma(t), \dot\gamma(t))} \, dt = \int_0^1 \|\dot\gamma(t)\|_{G(\gamma(t))} \, dt$$

and distances: $$d(\theta_1, \theta_2) = \inf_\gamma \text{Length}(\gamma)$$ over all smooth curves connecting $$\theta_1$$ to $$\theta_2$$.

---

## The Fisher Information Metric

For statistical models — and language models and RL policies are statistical models — there is a canonical Riemannian metric: the **Fisher information metric**. For a parametric family $$\{p_\theta\}$$, the Fisher information matrix is:

$$F_{ij}(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\frac{\partial \log p_\theta(x)}{\partial \theta_i} \frac{\partial \log p_\theta(x)}{\partial \theta_j}\right] = -\mathbb{E}_{x \sim p_\theta}\left[\frac{\partial^2 \log p_\theta(x)}{\partial \theta_i \partial \theta_j}\right]$$

The Fisher metric is the unique Riemannian metric (up to a constant) that is invariant under reparameterization of the model — it measures distance in terms of distinguishability of distributions, not in terms of Euclidean parameter distance.

Two parameter vectors $$\theta_1, \theta_2$$ that are close in Euclidean distance but far in Fisher distance define distributions $$p_{\theta_1}, p_{\theta_2}$$ that are easily distinguishable — the model has learned to be very sensitive to changes in those parameters. Conversely, two parameter vectors far in Euclidean distance but close in Fisher distance define nearly identical distributions — the model is insensitive to those parameters, and they encode redundant information.

The Fisher metric is related to the KL divergence by:

$$D_{\text{KL}}(p_\theta \| p_{\theta + d\theta}) = \frac{1}{2} d\theta^T F(\theta) \, d\theta + O(\|d\theta\|^3)$$

The Fisher metric is the local second-order approximation to KL divergence. The curvature of the statistical manifold is the curvature of the KL divergence. This has a direct alignment interpretation: two behavioral policies that are nearby in Fisher distance are similarly distinguishable by any observer, and thus similarly aligned from the perspective of any evaluator who judges by observed behavior.

---

## Geodesics and the Exponential Map

A **geodesic** on a Riemannian manifold is the curve of shortest length between two points — the generalization of a straight line to curved space. Geodesics satisfy the geodesic equation:

$$\ddot\gamma^k + \Gamma^k_{ij} \dot\gamma^i \dot\gamma^j = 0$$

where $$\Gamma^k_{ij}$$ are the **Christoffel symbols** of the metric:

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(\frac{\partial g_{jl}}{\partial \theta^i} + \frac{\partial g_{il}}{\partial \theta^j} - \frac{\partial g_{ij}}{\partial \theta^l}\right)$$

The Christoffel symbols encode the curvature of the space — how the coordinate basis vectors change as you move along the manifold. On flat Euclidean space, $$\Gamma^k_{ij} = 0$$ everywhere, and geodesics are straight lines. On a sphere, geodesics are great circles. On the statistical manifold of a language model, geodesics are the paths that interpolate between two policies in the most efficient way — changing the model's behavior as rapidly as possible per unit of parameter change, measured in the information-geometric sense.

The **exponential map** $$\exp_\theta: T_\theta \mathcal{M} \to \mathcal{M}$$ sends a tangent vector $$v \in T_\theta \mathcal{M}$$ to the point reached by following the geodesic starting at $$\theta$$ in direction $$v$$ for unit time. It satisfies $$\exp_\theta(0) = \theta$$ and $$\frac{d}{dt}\exp_\theta(tv)\big|_{t=0} = v$$. For small $$v$$, it approximates the Euclidean $$\theta + v$$.

The inverse, the **logarithmic map** $$\log_\theta: \mathcal{M} \to T_\theta \mathcal{M}$$, sends a nearby point $$\theta'$$ to the initial tangent vector of the geodesic from $$\theta$$ to $$\theta'$$. Together, $$\exp$$ and $$\log$$ allow us to do calculus on the manifold by converting between points and tangent vectors.

---

## Curvature and the Divergence of Geodesics

The **Riemann curvature tensor** measures the failure of parallel transport to be path-independent. For vector fields $$X, Y, Z$$, it is defined by:

$$R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

where $$\nabla$$ is the Levi-Civita connection (the unique torsion-free connection compatible with the metric). In coordinates:

$$R^l{}_{kij} = \partial_i \Gamma^l_{jk} - \partial_j \Gamma^l_{ik} + \Gamma^l_{im}\Gamma^m_{jk} - \Gamma^l_{jm}\Gamma^m_{ik}$$

The sectional curvature $$K(\sigma)$$ of a two-dimensional plane $$\sigma \subset T_\theta \mathcal{M}$$ spanned by orthonormal vectors $$u, v$$ is:

$$K(\sigma) = g(R(u,v)v, u)$$

Sectional curvature governs the **Jacobi equation**, which describes how nearby geodesics diverge. If $$J(t)$$ is a Jacobi field along a geodesic $$\gamma$$, measuring the separation between nearby geodesics, it satisfies:

$$\frac{D^2 J}{dt^2} + R(\dot\gamma, J)\dot\gamma = 0$$

In a space of constant sectional curvature $$K$$:
- If $$K > 0$$ (positive curvature, like a sphere): $$J(t) = J(0) \cos(\sqrt{K}\,t)$$ — geodesics converge, then refocus. Space is compact-like.
- If $$K = 0$$ (flat): $$J(t) = J(0) + t J'(0)$$ — geodesics diverge linearly, like Euclidean space.
- If $$K < 0$$ (negative curvature, like a hyperbolic space): $$J(t) = J(0) \cosh(\sqrt{|K|}\,t)$$ — geodesics diverge **exponentially**.

The alignment implication is stark. In a parameter space with negative sectional curvature, two optimization trajectories that begin nearby will diverge exponentially fast. A model trained to approximate a human value function and another trained to approximate a slightly different proxy value function will, under gradient descent (which follows geodesics of the loss landscape), diverge exponentially in their behavioral policies. The curvature is not a perturbation — it is the dominant term at even modest distances.

---

## Parallel Transport and the Holonomy of Value Drift

**Parallel transport** is the operation of moving a vector along a curve on a manifold while keeping it "as constant as possible" — formally, keeping it horizontal with respect to the Levi-Civita connection. If $$\gamma: [0,1] \to \mathcal{M}$$ is a curve and $$v_0 \in T_{\gamma(0)}\mathcal{M}$$, the parallel transport of $$v_0$$ along $$\gamma$$ is the vector field $$V(t) \in T_{\gamma(t)}\mathcal{M}$$ satisfying:

$$\nabla_{\dot\gamma} V = 0, \quad V(0) = v_0$$

On flat spaces, parallel transport is trivial — you simply translate the vector. On curved spaces, the transported vector rotates relative to its local frame, and this rotation depends on the path taken.

**Holonomy** is what happens when you parallel-transport a vector around a closed loop $$\gamma: [0,1] \to \mathcal{M}$$ with $$\gamma(0) = \gamma(1) = \theta$$. On flat space, the vector returns to itself. On curved space, it returns rotated by a holonomy transformation $$\text{Hol}(\gamma) \in SO(d)$$ that depends on the curvature enclosed by the loop.

For AI alignment, holonomy provides a metaphor that is also a theorem. Suppose a model undergoes a sequence of value updates — fine-tuning on dataset A, then B, then C, then back to A. In flat parameter space, this would return the model to its original value function. In a curved parameter space (with nontrivial holonomy), the model returns to the same parameter value but with a **rotated value gradient** — it now ascends the value function in a different direction than it did before the update sequence. The model has drifted in its implicit valuation of trade-offs without any single update having caused misalignment.

This is not a metaphor. The **Ambrose-Singer theorem** states that the holonomy group at a point $$\theta$$ is generated by the curvature tensors $$R(u,v)$$ at all points reachable from $$\theta$$. The curvature of the Fisher information manifold generates the holonomy of value drift under iterated fine-tuning. Alignment is not preserved under arbitrary loops of updates, and the failure is geometric — it is measured by the integrated curvature of the path through parameter space.

---

## The Geometry of Goodhart's Law

Goodhart's law states: "When a measure becomes a target, it ceases to be a good measure." In its canonical form this is sociological. In differential geometry, it is a theorem about the relationship between submanifolds.

Let $$\mathcal{M}_V$$ be the submanifold of policies that maximize the true value function $$V$$, and $$\mathcal{M}_\phi$$ be the submanifold that maximizes the proxy $$\phi$$. Goodhart's law says that optimizing over $$\mathcal{M}_\phi$$ does not in general take you to $$\mathcal{M}_V$$.

The precise statement involves the **second fundamental form** $$\mathrm{II}$$ of $$\mathcal{M}_\phi$$ inside $$\mathcal{M}$$. For a submanifold $$\mathcal{M}_\phi \hookrightarrow \mathcal{M}$$, the second fundamental form measures how $$\mathcal{M}_\phi$$ curves inside $$\mathcal{M}$$:

$$\mathrm{II}(X, Y) = (\nabla_X Y)^\perp$$

the component of the connection normal to $$\mathcal{M}_\phi$$. The mean curvature vector is $$H = \text{tr}(\mathrm{II})$$.

The divergence between $$\mathcal{M}_V$$ and $$\mathcal{M}_\phi$$ under optimization can be bounded by the **distortion** of the embedding:

$$d_\mathcal{M}(\mathcal{M}_V, \mathcal{M}_\phi) \leq \int_0^T \|\mathrm{II}(\dot\gamma(t), \dot\gamma(t))\| \, dt$$

where $$\gamma$$ is the optimization trajectory. The second fundamental form quantifies how much the proxy-optimal submanifold curves away from the value-optimal submanifold. Large curvature means optimization on $$\phi$$ leads rapidly away from $$V$$, even from a good starting point. Small curvature (nearly flat embedding) means the proxy is a good local surrogate.

Concretely: if the proxy $$\phi$$ is a projection of $$V$$ onto a lower-dimensional subspace, the second fundamental form of $$\mathcal{M}_\phi$$ is determined by the curvature of the level sets of the projection. For linear projections, the second fundamental form vanishes and Goodhart's law fails in its strong form — the proxy is a good surrogate. For nonlinear projections (human feedback through a neural evaluator, reward from a vision model), the second fundamental form can be large, and Goodhart's law is a theorem.

---

## Natural Gradient Descent as Geodesic Flow

The **natural gradient** of a loss function $$\mathcal{L}$$ on the statistical manifold is:

$$\tilde\nabla \mathcal{L}(\theta) = F(\theta)^{-1} \nabla \mathcal{L}(\theta)$$

where $$F(\theta)$$ is the Fisher information matrix. Natural gradient descent follows:

$$\theta_{t+1} = \theta_t - \eta \, F(\theta_t)^{-1} \nabla \mathcal{L}(\theta_t)$$

The geometric interpretation is exact: natural gradient descent follows the **steepest descent direction in the Fisher metric**, not in the Euclidean metric. Equivalently, it follows a discrete approximation to the geodesic on the statistical manifold induced by the KL divergence.

The difference between gradient descent and natural gradient descent is the difference between following a straight line in parameter space and following a geodesic in policy space. On the Fisher manifold, these diverge whenever the curvature is nonzero — i.e., always, for nontrivially parameterized models.

The **Amari-Chentsov theorem** identifies the Fisher metric as the unique metric (up to scale) invariant under sufficient statistics — it is the canonical metric for statistical inference, derived from first principles rather than imposed. Natural gradient descent in this metric is therefore the canonical optimization method on the statistical manifold, and standard gradient descent is a distorted approximation to it.

For alignment, this matters because RLHF and most fine-tuning methods use Euclidean gradient descent in parameter space. The geodesics they follow are not the shortest paths in policy space — they curve through parameter space in ways determined by the Fisher metric, and the resulting behavioral trajectories are harder to predict and control than they would be under natural gradient methods.

---

## Curvature Bounds and PAC-Bayesian Alignment

The **Bonnet-Myers theorem** states that if a complete Riemannian manifold $$(\mathcal{M}, g)$$ has Ricci curvature bounded below by $$K > 0$$:

$$\text{Ric}(v, v) \geq K \|v\|^2 \quad \forall v \in T\mathcal{M}$$

then $$\mathcal{M}$$ is compact with diameter at most $$\pi / \sqrt{K}$$. Applied to the statistical manifold of a model's policies: if the Fisher curvature is bounded below by a positive constant, the space of policies is compact — there is a maximum behavioral distance between any two policies in the family.

For alignment, this is a uniformity result: a model family with positive Ricci curvature bounded below has a finite diameter in policy space. The worst possible misalignment is bounded:

$$\sup_{\theta_1, \theta_2 \in \Theta} d_F(\pi_{\theta_1}, \pi_{\theta_2}) \leq \frac{\pi}{\sqrt{K}}$$

where $$d_F$$ is the Fisher distance. Increasing the curvature lower bound $$K$$ — which corresponds to constraining the model architecture to have more "curvature" in its policy parameterization — directly caps the maximum possible behavioral divergence. This connects geometric regularization (curvature bounds on the Fisher manifold) to PAC-Bayesian alignment guarantees.

The **Lichnerowicz theorem** further states that under positive Ricci curvature bounds, the first nonzero eigenvalue of the Laplace-Beltrami operator satisfies $$\lambda_1 \geq nK/(n-1)$$, where $$n = \dim \mathcal{M}$$. The Laplacian governs diffusion on the manifold, and its spectral gap controls the mixing time of optimization — how quickly gradient flows explore the policy space. A large spectral gap means optimization is fast and the model's behavioral space is well-connected; a small spectral gap means slow exploration and possible trapping near local optima.

---

## The Atiyah-Singer Index Theorem and Obstructions to Alignment

The most structurally deep result in differential geometry relevant to alignment is the **Atiyah-Singer index theorem**, which connects the topology of a manifold to the analysis of differential operators on it. For a Dirac operator $$D$$ on a compact Riemannian manifold $$\mathcal{M}$$:

$$\text{ind}(D) = \dim \ker D - \dim \ker D^* = \hat{A}(\mathcal{M})$$

where $$\hat{A}(\mathcal{M})$$ is the $$\hat{A}$$-genus, a topological invariant computable from the Pontryagin classes of $$\mathcal{M}$$.

The index theorem says that certain operators on manifolds must have nontrivial kernels — solutions that cannot be eliminated by smooth deformation. Applied to the policy manifold: if the space of policies has nontrivial topology (measured by its characteristic classes), then certain alignment constraints — formalized as differential equations on the policy manifold — must have solutions even when we would prefer they didn't. There are **topological obstructions** to eliminating certain failure modes.

While this application to alignment is speculative, it points toward a research agenda: use the topology of the policy manifold to identify which failure modes are topologically inevitable — not eliminated by any training procedure that preserves the manifold structure — versus which are contingent artifacts of the specific optimization trajectory taken.

The relevance is philosophical as much as technical. Differential geometry tells us that the spaces in which AI systems operate have intrinsic structure — curvature, topology, holonomy — that constrains their behavior independently of any specific training objective. Alignment failures that look like optimization failures may be geometric necessities: consequences of the curvature of the value landscape, not errors in the training pipeline.

---

## Philosophical Synthesis: Alignment as Isometry

What would a perfectly aligned model look like, from a differential geometric perspective? It would be one whose policy parameterization $$\theta \mapsto \pi_\theta$$ is an **isometry** from the parameter manifold $$(\Theta, G_\Theta)$$ (with some appropriate metric, such as the Euclidean metric on gradient steps) to the value manifold $$(\mathcal{M}_V, G_V)$$ (the submanifold of policies consistent with human values, with the Fisher metric).

An isometry preserves distances: parameter changes translate into policy changes of proportional magnitude, in the same direction, at every point. There is no curvature-induced divergence between what the optimizer does (moves through parameter space) and what matters (moves through value space). The model faithfully represents the geometry of human values in its own geometry.

This is an impossibly strong demand. It requires that the model's architecture encode the exact curvature of the human value function — that the Fisher metric of the model policy class be isometric to the metric on human value space. We have no access to the latter, and the former is determined by architecture choices made without alignment in mind.

But the isometric ideal is useful as a target. Techniques that reduce the distortion of the embedding $$\Theta \hookrightarrow \mathcal{M}_V$$ — that make the policy manifold "more isometric" to the value manifold — are, in this geometric sense, alignment techniques. Natural gradient methods reduce distortion by respecting the Fisher geometry. Curvature regularization reduces distortion by controlling the second fundamental form. Representation alignment — ensuring that the model's internal geometry mirrors the geometry of the target domain — reduces distortion by construction.

The alignment problem is not, fundamentally, a problem about objectives. It is a problem about geometry. We are trying to embed a high-dimensional optimization process into a space whose curvature we do not know, and asking that the embedding be nearly isometric to a submanifold we cannot directly observe. Differential geometry does not solve this problem. But it tells us, with precision, what kind of problem it is.
