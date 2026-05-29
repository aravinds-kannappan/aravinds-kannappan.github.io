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

Ask a mechanistic interpretability researcher why neural networks form the circuits they do and you will get an honest answer: we observe them, name them, and ablate them, but we lack a theory of why they emerge. This is not a complaint about the field — the empirical discoveries are real and important. It is a statement about what is missing. What we need is a mathematical account of why, given a data distribution and an architecture, gradient descent converges to representations with specific structural properties rather than others.

Singular Learning Theory (SLT), developed by Sumio Watanabe across a series of papers and a 2009 monograph, provides that account. It begins with an observation that is easy to state and took decades to fully appreciate: neural networks are not regular statistical models, and their irregularity is not a nuisance to be engineered around. It is the source of their generalization, and it is the key to understanding what their representations mean.

---

## The Problem With Regular Models

Classical statistical theory rests on an assumption so basic it often goes unstated: the model is *regular*. Formally, this means the Fisher information matrix $F(\theta)$ is nonsingular at the true parameter $\theta^*$, so the log-likelihood landscape near the optimum is well-approximated by a bowl-shaped quadratic. Under regularity, maximum likelihood estimators are asymptotically normal, model complexity is well-captured by parameter count, and the Bayesian information criterion

$$\text{BIC} = -2\log p(\text{data} \mid \hat\theta) + d\log n$$

gives a reliable estimate of generalization, where $d$ is the number of parameters and $n$ is the sample size.

Neural networks violate regularity comprehensively. At any parameter $\theta^*$ that implements the true function, the Fisher matrix is generically singular. Its null space — the set of parameter perturbations that leave the network's input-output mapping unchanged — is large. It includes permuting hidden units, rescaling weights between layers, and more subtle symmetries that grow with depth. The consequence is that the loss landscape near $\theta^*$ is not a bowl. It is a polynomial of degree higher than two, and its zero set $W_0 = \{\theta : L(\theta) = L^*\}$ is not an isolated point but a real algebraic variety — a high-dimensional surface of equivalent solutions, with a complex topology that depends on the network architecture and the data distribution.

This matters because the BIC uses $d\log n$ as its complexity penalty, implicitly assuming all $d$ parameters do independent work. For a singular model, many parameters are redundant — the effective complexity is lower, the model generalizes better than BIC predicts, and a different mathematical object is needed to describe what is really happening.

---

## The Real Log Canonical Threshold

That object is the **real log canonical threshold (RLCT)**, also called the learning coefficient $\lambda$. To define it, let $K(\theta) = \mathbb{E}[\log p_{\theta^*}/p_\theta]$ be the KL divergence from the optimal model — it measures how far a parameter $\theta$ is from the optimal function in an information-theoretic sense. Near the optimal set $W_0$, $K(\theta)$ vanishes on $W_0$ and grows as a polynomial as you move away from it. The RLCT is extracted from the **zeta function** of the loss:

$$\zeta(z) = \int_\Theta K(\theta)^z \, \varphi(\theta) \, d\theta$$

where $\varphi(\theta)$ is a prior density. This integral is holomorphic for $\text{Re}(z) > 0$ and has a meromorphic continuation to the complex plane. The RLCT $\lambda$ is minus the largest pole:

$$\lambda = -\max\{z \in \mathbb{R} : \zeta(z) \text{ has a pole at } z\}$$

The pole's multiplicity $m$ captures logarithmic correction terms. Together, $\lambda$ and $m$ appear in Watanabe's central theorem: the Bayesian free energy (negative log marginal likelihood) satisfies

$$F_n = nL_n + \lambda \log n - (m-1)\log\log n + O_p(1)$$

and the expected generalization error satisfies

$$\mathbb{E}[G_n] = \frac{\lambda}{n} - \frac{m-1}{n}\log\log n + O\!\left(\frac{1}{n}\right)$$

For regular models, $\lambda = d/2$ and $m = 1$, exactly recovering the BIC. For singular models, $\lambda < d/2$ — the model is penalized *less* for complexity, and this reduced penalty directly explains why overparameterized networks generalize: their effective complexity, as measured by $\lambda$, is much smaller than their parameter count.

The computation of $\lambda$ for a specific architecture requires **resolution of singularities** — a classical algebraic geometry result (Hironaka, 1964) guaranteeing that any algebraic variety can be desingularized by a finite sequence of coordinate changes called blowups. After blowing up the singular points of $K(\theta)$ into normal crossing divisors, the zeta function becomes a standard multidimensional integral whose poles can be read off directly.

For a three-layer network with $H$ hidden units mapping $\mathbb{R}^m \to \mathbb{R}^H \to \mathbb{R}^n$ with tanh activations, the result is:

$$\lambda = \frac{1}{2}\min_{k_1 + k_2 \leq H} \left[\frac{m k_1 - k_1^2}{2} + \frac{k_2 n - k_2^2}{2} + k_1 k_2\right]$$

minimized over integer decompositions of the hidden layer. This formula is structurally revealing: $\lambda$ depends not on the total parameter count but on how the hidden layer's width relates to the input and output dimensions. Wider-than-necessary hidden layers do not increase $\lambda$ proportionally — they increase it sublinearly, because the additional units contribute to the null space of $F(\theta^*)$ rather than to genuinely independent parameters.

---

## Phase Transitions and Grokking

SLT's second major insight concerns how a model's internal representations evolve during training. As the sample size $n$ grows, the Bayesian posterior

$$\varphi_n(\theta) \propto \exp(-nL_n(\theta))\,\varphi(\theta)$$

undergoes a sequence of abrupt reorganizations. Initially the posterior spreads across parameter space (underfitting). As $n$ increases, it concentrates on the optimal set $W_0$. But $W_0$ is not a connected manifold in general — it has multiple connected components with different values of $\lambda$. The posterior concentrates first on the component with the highest $\lambda$ (least singular), then, as evidence accumulates, undergoes a first-order phase transition to the component with lower $\lambda$ (more singular, better generalization).

This is the SLT account of **grokking** — the striking phenomenon where a model trained on a small dataset first memorizes it (high training accuracy, low test accuracy), then suddenly generalizes far later in training. The memorization regime corresponds to the posterior sitting on a high-$\lambda$ component of $W_0$. The generalization transition is the phase transition to a lower-$\lambda$ component. The delay between memorization and generalization reflects the time needed to accumulate enough evidence to overcome the free energy barrier between components.

The phase transition is not smooth. In experiments on modular addition with small transformers, the transition from memorization to generalization occurs over a narrow range of training steps, the test loss drops sharply (not gradually), and the internal representations reorganize simultaneously — a Fourier-mode structure appears in the embedding weights within hundreds of steps of the transition. SLT predicts exactly this: because the transition is between discrete components of $W_0$ with different $\lambda$, it is necessarily discontinuous.

---

## Superposition as a Compressed Sensing Problem

One of the most studied phenomena in interpretability is **superposition**: networks represent more features than they have neurons by encoding multiple features in overlapping, nonorthogonal directions in activation space. The naive expectation would be one feature per neuron; the observation is a feature count that scales superlinearly with neuron count. Why?

The compressed sensing framework makes this precise. Let $\mathbf{h} \in \mathbb{R}^d$ be an activation vector, and suppose the network wants to represent $n \gg d$ sparse features $\mathbf{f} \in \mathbb{R}^n$ (sparse meaning most entries are near zero at any given time). The encoding is $\mathbf{h} = W\mathbf{f}$, with $W \in \mathbb{R}^{d \times n}$. This is underdetermined — more features than dimensions — and recovery requires that the encoding matrix $W$ satisfy the **Restricted Isometry Property (RIP)**:

$$(1-\delta_s)\lVert\mathbf{f}\rVert^2 \leq \lVert W\mathbf{f}\rVert^2 \leq (1+\delta_s)\lVert\mathbf{f}\rVert^2$$

for all $s$-sparse vectors $\mathbf{f}$, with $\delta_s < \sqrt{2}-1$. When RIP holds, $\ell_1$-minimization exactly recovers sparse features from the compressed representation.

The fundamental result of compressed sensing is that random $d \times n$ matrices satisfy RIP with high probability when $n = O(d^2)$ — the number of recoverable sparse features scales quadratically with the number of neurons, not linearly. This is precisely the empirical observation: networks trained on tasks with many sparse features organize their weights into near-equiangular tight frames, which achieve the $d(d+1)/2$ Welch bound on the number of equiangular directions in $\mathbb{R}^d$.

Which features end up in superposition? The features that are most common and most sparse in the training distribution. Common features are worth representing; sparse features can be represented without much interference because they rarely co-activate. SLT connects this to the RLCT: the low-$\lambda$ region of parameter space (which gradient descent converges to) is precisely the region where the weight matrices implement near-optimal compressed sensing — RIP-satisfying frames that maximize recoverable feature count per neuron.

---

## Causal Abstraction

The empirical side of interpretability — circuit finding, activation patching, probing — is unified by **causal abstraction** theory. A circuit is not just a subgraph of the network's computation — it is a claim that the network implements a high-level algorithm, where "implements" has a precise meaning in terms of interventions.

Two causal models are **causally abstracted** if there exists a map $\alpha$ from low-level states (activations) to high-level states (algorithm variables) such that intervening on the high-level model corresponds, through $\alpha$, to intervening on the low-level model:

$$\alpha\!\left(\mathcal{M}_{\text{low}}^{\,\alpha^{-1}(\mathcal{I})}(\mathbf{s})\right) = \mathcal{M}_{\text{high}}^{\,\mathcal{I}}(\alpha(\mathbf{s}))$$

The diagram commutes: abstract and concrete interventions are related by $\alpha$. **Interchange interventions** (setting an activation to the value it would take under a different input) test this commutativity empirically. A circuit is validated when interchange interventions on the claimed algorithmic variables produce the same output changes as the corresponding high-level interventions.

What SLT adds to this picture: the structure of the low-level model $\mathcal{M}_{\text{low}}$ at a convergence point with small $\lambda$ is not arbitrary. The RLCT measures the degree of parameter redundancy, and highly redundant parameterizations (low $\lambda$) are precisely those where simple causal abstractions exist. A model with many equivalent parameterizations has, by definition, a large null space of parameter changes that don't affect the function — and any basis of this null space defines a natural set of "irrelevant parameters" that can be abstracted away. The abstraction map $\alpha$ is most cleanly defined at highly singular convergence points, which is why circuit structure appears most clearly in well-trained models and not in randomly initialized ones.

---

## Representation Theory and the Inevitability of Fourier Structure

Perhaps the most striking concrete prediction of SLT — actually its representation-theoretic analogue — is the inevitability of Fourier representations in models trained on cyclic-symmetry tasks. **Schur's lemma** states that if $\rho$ and $\rho'$ are irreducible representations of a group $G$ and $f$ is an equivariant linear map between them, then $f = 0$ if $\rho \not\cong \rho'$ and $f = \lambda I$ if $\rho \cong \rho'$ over $\mathbb{C}$.

This means that any linear layer that respects a symmetry group $G$ must block-diagonalize along irreducible representations of $G$ — it cannot mix different irreps. For cyclic groups $\mathbb{Z}_n$ (the symmetry of modular arithmetic tasks), the irreducible representations are exactly the Fourier modes $e^{2\pi i k / n}$ for $k = 0, \ldots, n-1$.

Networks trained on modular arithmetic tasks learn Fourier representations not because we built Fourier structure in, but because the task symmetry group is $\mathbb{Z}_n$, Schur's lemma forces any equivariant linear layer to be diagonal in the Fourier basis, and gradient descent on the cross-entropy loss preserves the symmetry because the loss is itself $\mathbb{Z}_n$-invariant. The Fourier basis is the unique basis that makes the equivariant constraint compatible with efficient learning. It is, in a precise algebraic sense, the only basis that works.

---

## What This Means for Interpretability Research

The SLT picture synthesizes the empirical interpretability findings into a coherent theoretical frame. Circuits emerge because training converges to low-$\lambda$ points of $W_0$, where the functional structure is simple and the causal abstraction map $\alpha$ is cleanly defined. Grokking is a phase transition between components of $W_0$. Superposition is the compressed sensing solution that gradient descent finds at those low-$\lambda$ points. Fourier representations are the algebraically inevitable outcome of training on symmetric tasks.

What SLT cannot yet do is predict in advance which features will appear in which circuits for a given architecture and dataset. Computing $\lambda$ for real neural networks requires resolving the singularities of high-dimensional polynomial systems — a problem that is algebraically well-posed but computationally very hard. The practical path forward is to use SLT as a diagnostic: measure $\lambda$ empirically (via the Bayesian free energy on held-out data), track its changes during training, and use phase transition predictions to identify the training checkpoints where representational reorganization is most likely to occur.

The hard problem of interpretability is not a problem of visualization or measurement. It is a problem of algebraic geometry. We are trying to understand the structure of a real algebraic variety in a billion-dimensional space, and the features we see in circuits are the natural coordinates near its most degenerate points.
