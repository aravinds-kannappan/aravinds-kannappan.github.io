---
title: "Singular Learning Theory and the Geometry of Neural Network Interpretability"
date: 2026-03-01
permalink: /posts/2026/03/singular-learning-interpretability/
tags:
  - interpretability
  - singular learning theory
  - mathematics
  - AI
---

Interpretability research often proceeds empirically: find a circuit, patch an activation, ablate a head, observe the behavioral change. This is valuable work, but it lacks a theory of *why* neural networks form the internal structures they do — why circuits emerge at all, why features are represented in superposition, why some behaviors are localized and others distributed. Singular Learning Theory (SLT), developed by Sumio Watanabe over two decades of statistical mathematics, provides the deepest available mathematical account of these questions. It begins with an observation that the field of machine learning spent decades overlooking: the models we train are not regular statistical models, and their irregularity is not a nuisance — it is the source of their generalization and the key to understanding their internal structure.

---

## The Failure of Regular Statistical Theory

Classical statistical learning theory assumes models are **regular**: that the Fisher information matrix $$F(\theta)$$ is non-singular at all parameter values, that maximum likelihood estimators are asymptotically normal, and that the model's true complexity is well-approximated by its parameter count. These assumptions fail catastrophically for neural networks.

At a parameter value $$\theta^*$$ where the network implements the true function $$f^*$$, the Fisher information matrix is generically **singular**. The null space of $$F(\theta^*)$$ — the set of parameter perturbations that do not change the network's input-output function — is non-trivial. This null space has dimension equal to the number of **symmetries** of the parameterization: permutation of hidden units, rescaling of weights between layers, and more subtle symmetries in deep networks.

The consequence is that the loss landscape near $$\theta^*$$ is not quadratic — it is a polynomial of degree higher than two, with a complicated zero set. This zero set $$W_0 = \{\theta : L(\theta) = L^*\}$$, the set of globally optimal parameters, is not an isolated point but a **real algebraic variety** — the intersection of polynomial equations in parameter space.

**Definition (Singularity):** A parameter $$\theta^* \in W_0$$ is a **singular point** of the loss landscape if the Hessian $$\nabla^2 L(\theta^*)$$ is singular. At singular points, the normal approximation to the posterior breaks down, and the standard Bayesian information criterion (BIC) — which assumes the Hessian determines model complexity — gives the wrong answer.

---

## Real Log Canonical Thresholds

The correct measure of effective model complexity at a singular point is the **real log canonical threshold (RLCT)**, also called the learning coefficient $$\lambda$$.

Let $$L(\theta) = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(x_i)$$ be the empirical loss, and $$K(\theta) = \mathbb{E}[\log p_{\theta^*}/p_\theta]$$ be the true KL divergence from the optimal model. Define the **zeta function** of the loss:

$$\zeta(z) = \int_\Theta K(\theta)^z \varphi(\theta) \, d\theta$$

where $$\varphi(\theta)$$ is a prior density. This integral is holomorphic for $$\text{Re}(z) > 0$$ but has a meromorphic continuation to the entire complex plane. The **learning coefficient** $$\lambda$$ is minus the largest pole of $$\zeta(z)$$:

$$\lambda = -\max\{z \in \mathbb{R} : \zeta(z) \text{ has a pole at } z\}$$

For regular models, $$\lambda = d/2$$ where $$d$$ is the parameter dimension — this recovers the classical BIC. For singular models, $$\lambda < d/2$$, reflecting the fact that the model's effective complexity is less than its parameter count. The **multiplicity** $$m$$ of the pole determines the logarithmic corrections.

Watanabe's **free energy formula** (his central theorem) states that the Bayesian generalization error satisfies:

$$\mathbb{E}[G_n] = \frac{\lambda}{n} - \frac{m-1}{n}\log\log n + O\left(\frac{1}{n}\right)$$

and the Bayesian free energy (negative log marginal likelihood) satisfies:

$$F_n = nL_n + \lambda \log n - (m-1)\log\log n + O_p(1)$$

where $$L_n$$ is the empirical training loss. The term $$\lambda \log n$$ replaces the $$\frac{d}{2}\log n$$ term of the classical BIC. Because $$\lambda \leq d/2$$ for singular models, singular models are *penalized less* for complexity — they generalize better than their parameter count suggests.

---

## Computing $$\lambda$$ for Neural Networks

The RLCT is a birational invariant from algebraic geometry. For a polynomial map $$f: \mathbb{R}^d \to \mathbb{R}^k$$ representing the KL divergence $$K(\theta) = \|f(\theta)\|^2$$ near a singular point, the RLCT is computed via **resolution of singularities** — a classical result of Hironaka guaranteeing that any algebraic variety can be desingularized by a finite sequence of blowups.

For a 3-layer network with $$H$$ hidden units mapping $$\mathbb{R}^m \to \mathbb{R}^H \to \mathbb{R}^n$$ (tanh activations), Watanabe and collaborators computed:

$$\lambda = \frac{1}{2}\min_{k_1 + k_2 \leq H} \left(\frac{m \cdot k_1 - k_1^2}{2} + \frac{k_2 \cdot n - k_2^2}{2} + k_1 k_2\right)$$

subject to constraints on the network's rank. The formula involves minimizing over the decomposition of the hidden layer into two ranks $$k_1, k_2 \leq H$$ — the RLCT is determined by the most singular decomposition of the weight matrices.

For ReLU networks, the situation is more complex because the piecewise-linear activation creates a **stratified** parameter space — different activation patterns correspond to different polynomial maps, and the RLCT must be computed separately on each stratum and then assembled via the minimum:

$$\lambda = \min_{\sigma \in \{0,1\}^H} \lambda_\sigma$$

where $$\sigma$$ indexes the activation pattern and $$\lambda_\sigma$$ is the RLCT on the corresponding stratum.

The key result: **smaller $$\lambda$$ corresponds to a more singular loss landscape, which corresponds to more parameter symmetry, which corresponds to more regularized and interpretable representations.** The RLCT is a formal measure of how "overparameterized" a model is — how many of its degrees of freedom are redundant — and this redundancy is precisely the condition under which simple, human-interpretable features can emerge.

---

## Phase Transitions and Developmental Interpretability

Singular learning theory predicts that as a model trains, it undergoes **phase transitions** — abrupt changes in the structure of its parameter distribution. These transitions occur at critical sample sizes where the posterior distribution suddenly concentrates on a lower-dimensional stratum of the parameter space.

Define the **Bayesian posterior** $$\varphi_n(\theta) \propto \exp(-nL_n(\theta))\varphi(\theta)$$. As $$n$$ increases from 0, the posterior initially spreads over the entire parameter space (underfitting regime), then concentrates on the variety $$W_0$$ (interpolation regime), then concentrates on specific connected components of $$W_0$$ determined by the data structure (representation learning regime).

The transitions between these regimes are characterized by changes in $$\lambda$$. When the posterior crosses from a higher-$$\lambda$$ to a lower-$$\lambda$$ stratum, the model undergoes a phase transition in which:

1. The free energy decreases discontinuously relative to the BIC prediction
2. The model's internal representations reorganize
3. New interpretable features appear

This is the SLT account of **grokking** — the phenomenon where models suddenly generalize after extended training. Grokking is a phase transition in the posterior from a high-$$\lambda$$ stratum (memorization, poor generalization) to a low-$$\lambda$$ stratum (structured representation, good generalization). The transition is delayed because the posterior must accumulate sufficient evidence to escape the higher-$$\lambda$$ stratum, even though the lower-$$\lambda$$ stratum has better free energy.

---

## Superposition as a Compressed Sensing Problem

One of the most studied phenomena in interpretability is **superposition**: the observation that neural networks represent more features than they have neurons, by storing multiple features in overlapping directions in activation space. SLT provides a principled mathematical account of when and why this occurs.

Let $$\mathbf{h} \in \mathbb{R}^d$$ be an activation vector and suppose there are $$n \gg d$$ features $$f_1, \ldots, f_n \in \mathbb{R}$$ that the network wants to represent. The network encodes them as:

$$\mathbf{h} = W\mathbf{f} + \epsilon$$

where $$W \in \mathbb{R}^{d \times n}$$ is the encoding matrix and $$\epsilon$$ is noise. Recovering $$\mathbf{f}$$ from $$\mathbf{h}$$ requires solving an underdetermined system — a compressed sensing problem.

The **Restricted Isometry Property (RIP)** states that a matrix $$W$$ satisfies RIP of order $$s$$ with constant $$\delta_s$$ if for all $$s$$-sparse vectors $$\mathbf{f}$$ (at most $$s$$ nonzero entries):

$$(1-\delta_s)\|\mathbf{f}\|^2 \leq \|W\mathbf{f}\|^2 \leq (1+\delta_s)\|\mathbf{f}\|^2$$

If $$W$$ satisfies RIP with $$\delta_{2s} < \sqrt{2} - 1$$, then the $$\ell_1$$ minimization program:

$$\hat{\mathbf{f}} = \arg\min_{\mathbf{f}'} \|\mathbf{f}'\|_1 \quad \text{subject to} \quad W\mathbf{f}' = \mathbf{h}$$

recovers $$\mathbf{f}$$ exactly when it is $$s$$-sparse, with error at most $$C\|\mathbf{f} - \mathbf{f}_s\|_1 / \sqrt{s}$$ otherwise (where $$\mathbf{f}_s$$ is the best $$s$$-sparse approximation).

Superposition is, precisely, a network that stores sparse features in a low-dimensional space by exploiting the RIP: the weight matrix $$W$$ encodes $$n$$ features into $$d \ll n$$ neurons while approximately preserving inner products between sparse feature combinations. The features the network chooses to represent are those that are **both important and sparse in the natural data distribution** — because RIP guarantees reliable recovery only for sparse feature patterns.

The number of features representable in superposition scales as $$n = O(d^2)$$ for random unit-norm frames (achieving $$d(d+1)/2$$ equiangular tight frame bound) — quadratically in the number of neurons, not linearly. This is why interpretability researchers find "more features than neurons": the correct scaling is polynomial.

---

## Causal Abstraction and Mechanistic Circuits

Mechanistic interpretability seeks to identify **circuits**: subgraphs of the computational graph that implement specific algorithmic functions. The mathematical foundation is **causal abstraction** theory, developed by Geiger, Lu, and Potts.

A **causal model** is a tuple $$\mathcal{M} = (\mathbf{V}, \mathbf{E}, \mathbf{U}, \mathcal{F})$$ where $$\mathbf{V}$$ are endogenous variables, $$\mathbf{E}$$ are edges, $$\mathbf{U}$$ are exogenous noise variables, and $$\mathcal{F} = \{f_V : V \in \mathbf{V}\}$$ are structural equations $$V = f_V(\text{Pa}(V), U_V)$$.

An **intervention** $$\text{do}(V = v)$$ sets $$V$$ to $$v$$ and removes its incoming edges, yielding the **interventional distribution** $$P(\mathbf{W} | \text{do}(V = v))$$ for any downstream variables $$\mathbf{W}$$.

Two causal models $$\mathcal{M}_{\text{high}}$$ (high-level, abstract) and $$\mathcal{M}_{\text{low}}$$ (low-level, neural network) are **causally abstracted** if there exists an **abstraction map** $$\alpha: \mathcal{S}_{\text{low}} \to \mathcal{S}_{\text{high}}$$ (from neural network states to high-level model states) such that for all interventions $$\mathcal{I}$$ in the high-level model and their corresponding **realizations** $$\mathcal{I}_\alpha$$ in the low-level model:

$$\alpha(\mathcal{M}_{\text{low}}^{\mathcal{I}_\alpha}(\mathbf{s})) = \mathcal{M}_{\text{high}}^{\mathcal{I}}(\alpha(\mathbf{s}))$$

Diagrammatically: the interventional map commutes with the abstraction map. The neural network implements the high-level algorithm if and only if this commutativity holds for all interventions.

**Interchange interventions** (setting an activation to the value it would have had under a different input) test this commutativity empirically. If the commutativity holds up to approximation $$\epsilon$$ for all tested interventions, we say the network $$\epsilon$$-approximately implements the high-level algorithm. The set of interventions needed to certify a circuit is at most exponential in the circuit's depth, and polynomial in its width for sufficiently structured algorithms.

---

## Representation Theory and Equivariance

Deep networks trained on structured data — language, images, graphs — develop representations that reflect the symmetry structure of the data. This is not accidental: it is a consequence of representation theory applied to the hypothesis class.

Let $$G$$ be a group acting on input space $$\mathcal{X}$$ by $$\rho: G \to GL(\mathcal{X})$$ (a representation). A function $$f: \mathcal{X} \to \mathcal{Y}$$ is **equivariant** with respect to $$\rho$$ if:

$$f(\rho(g) \cdot x) = \rho'(g) \cdot f(x) \quad \forall g \in G, x \in \mathcal{X}$$

for some representation $$\rho'$$ of $$G$$ on $$\mathcal{Y}$$. Equivariant functions respect the group symmetry: transforming the input is equivalent to transforming the output.

**Schur's lemma** constrains the structure of equivariant linear maps. If $$\rho$$ and $$\rho'$$ are irreducible representations of $$G$$ and $$f: V \to W$$ is an equivariant linear map, then:

- If $$\rho \not\cong \rho'$$, then $$f = 0$$
- If $$\rho \cong \rho'$$ (over $$\mathbb{C}$$), then $$f = \lambda I$$ for some scalar $$\lambda$$

This means that equivariant linear layers can only mix components within the same irreducible representation — the weight matrix block-diagonalizes along irreducible components. The internal representation of a perfectly equivariant network is a direct sum of irreducible representations of $$G$$, and each irrep corresponds to an interpretable feature type.

For cyclic groups $$\mathbb{Z}_n$$ (relevant to modular arithmetic tasks studied in grokking), the irreducible representations are Fourier modes $$e^{2\pi i k / n}$$ for $$k = 0, 1, \ldots, n-1$$. Networks trained on modular arithmetic tasks learn Fourier representations — not because Fourier analysis was built in, but because Fourier modes are the unique basis that diagonalizes the cyclic symmetry group, and diagonalizing the symmetry is optimal for weight sharing. Schur's lemma makes the Fourier basis inevitable.

---

## Philosophical Implications: What Interpretability Is Measuring

SLT tells us that interpretability is fundamentally about the geometry of the loss landscape near the zero set $$W_0$$. When a model is interpretable — when its computations can be decomposed into human-readable circuits — it is because the parameters lie near a singularity of low RLCT: a highly degenerate point where many parameter directions do not affect the function. Interpretable features are the natural coordinates near these singularities.

This reframes the interpretability research agenda. The question is not "how do we find the features?" but "what determines the RLCT of the region where the model converges?" The answer is the data distribution, the architecture, and the training dynamics. Architectures that create more symmetry — more parameter degeneracy — have smaller RLCT and more interpretable convergence points. Training procedures that push models toward lower-$$\lambda$$ strata produce more interpretable representations.

The hard problem of interpretability, then, is not a problem about visualization or measurement — it is a problem about algebraic geometry. Understanding why a model is interpretable requires resolving the singularities of its loss landscape, computing the RLCT at its convergence point, and identifying the irreducible representations of the symmetry groups preserved by the architecture. This is hard mathematics, not just empirical circuit-hunting. But it is the right mathematics for the right question.
