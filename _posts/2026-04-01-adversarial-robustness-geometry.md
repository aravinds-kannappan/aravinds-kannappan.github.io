---
title: "The Mathematics of Adversarial Robustness: Certified Defenses, Lipschitz Geometry, and the Limits of Perturbation Sets"
date: 2026-04-01
permalink: /posts/2026/04/adversarial-robustness-geometry/
tags:
  - robustness
  - adversarial examples
  - mathematics
  - AI safety
---

In 2013 Szegedy et al. showed that a GoogLeNet classifier, trained to near-human accuracy on ImageNet, could be fooled by adding imperceptibly small perturbations to any input image. The perturbations were invisible to human eyes, no larger than the noise in a compressed JPEG, but they caused confident, catastrophically wrong predictions. The model saw a school bus and called it an ostrich. A decade later, after thousands of papers on attacks and defenses, the phenomenon is still not fully understood. State-of-the-art models remain vulnerable in ways that defy intuitive explanation.

The reason adversarial examples are hard to explain is that they are not, primarily, an engineering failure. They are a mathematical consequence of high-dimensional geometry, and understanding them requires moving from empirical observations about specific models to theorems about the geometry of decision boundaries and the measure-theoretic structure of high-dimensional probability spaces.

---

## What We Are Trying to Prove

The goal of a certified defense is to produce a classifier $f$ together with a guarantee: for every test point $x$ in some distribution, and every perturbation $\delta$ with $\lVert\delta\rVert \leq \epsilon$, the classifier makes the same prediction on $x + \delta$ as on $x$. The guarantee is not probabilistic, it is a proof that holds for all perturbations in the threat model simultaneously.

Define the **robust accuracy** at radius $\epsilon$ as:

$$\text{RA}(\epsilon) = P_{(x,y) \sim \mathcal{D}}\!\left(\min_{\lVert\delta\rVert \leq \epsilon} f(x + \delta) = y\right)$$

For most models trained with standard cross-entropy loss, $\text{RA}(\epsilon) \approx 0$ for $\ell_\infty$ perturbations with $\epsilon = 8/255 \approx 0.03$ on ImageNet. The **robustness gap** $\text{RA}(0) - \text{RA}(\epsilon)$ is large, and understanding its size, and whether it can be closed, requires understanding the geometry of the decision boundary.

---

## Why Perturbations Are So Small: The Linear Hypothesis

The first explanation for adversarial vulnerability is simple and important. For a linear classifier $f(x) = \text{sign}(w \cdot x + b)$, the $\ell_\infty$ distance from $x$ to the decision boundary is

$$\epsilon^* = \frac{\lvert w \cdot x + b \rvert}{\lVert w \rVert_1}$$

In $d$ dimensions with $d = 10^6$ (ImageNet), the $\ell_1$ norm $\lVert w \rVert_1$ can be enormous even when $\lVert w \rVert_2$ is moderate, because $\lVert w \rVert_1 \leq \sqrt{d} \lVert w \rVert_2$. Perturbing each coordinate by $\epsilon$ in the direction of $\text{sign}(w_i)$ changes the dot product by $\epsilon \lVert w \rVert_1$, which grows as $O(\epsilon \sqrt{d})$ for a unit-norm weight vector. In $d = 10^6$, a perturbation of magnitude $\epsilon = 0.03$ in every coordinate changes the dot product by roughly $0.03 \times 1000 = 30$, far larger than the margin $\lvert w \cdot x + b \rvert$ for most inputs.

The linear classifier is fragile not despite being a good classifier, but *because* it is a good classifier that uses all available dimensions. Good linear classification requires that many small features combine into a decisive prediction, and adversarial attacks exploit exactly those many small features. The same model properties that make a classifier accurate make it adversarially fragile, and this is not a coincidence. It is the linear hypothesis, and it applies to neural networks approximately, because deep networks in their bottom layers perform operations that are approximately linear in the input.

---

## Randomized Smoothing: The Certified Defense

The most practically scalable certified defense is **randomized smoothing**. The idea is elegant: rather than trying to make a given classifier $f$ robust, build a *new* classifier $g$ by averaging $f$ over random noise:

$$g(x) = \arg\max_{c} \; P\!\left(f(x + \delta) = c\right), \quad \delta \sim \mathcal{N}(0, \sigma^2 I)$$

The smoothed classifier $g$ predicts whichever class $f$ outputs most often when the input is perturbed by Gaussian noise. This sounds like it would reduce accuracy, and it does, but the noise also creates a certificate.

The certificate comes from the following argument. Let $c_A$ be the most probable class under $g$ at input $x$, with probability $p_A = P(f(x+\delta) = c_A)$, and let $p_B = \max_{c \neq c_A} P(f(x+\delta) = c)$. **Cohen et al. (2019)** proved that $g$ is certifiably robust within $\ell_2$ radius

$$R = \frac{\sigma}{2}\!\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)$$

where $\Phi^{-1}$ is the inverse normal CDF. The proof uses the Neyman-Pearson lemma: among all sets of measure $p_A$ under $\mathcal{N}(x, \sigma^2 I)$, the one that is hardest to shift to the second class under a translation $\delta$ with $\lVert\delta\rVert_2 \leq R$ is a half-space, and the certificate is the radius at which this half-space's probability falls below $p_B$.

The table below shows the tradeoff between noise level $\sigma$, clean accuracy, and certified accuracy at radius $R = 0.5$ on CIFAR-10, from Cohen et al.'s original experiments:

| $\sigma$ | Clean accuracy | Certified acc. ($R = 0.5$) |
|---------|---------------|---------------------------|
| 0.12    | 77%           | 0%                        |
| 0.25    | 74%           | 49%                       |
| 0.50    | 67%           | 38%                       |
| 1.00    | 57%           | 22%                       |

The tradeoff is fundamental, not incidental: more noise creates larger certificates but worse clean predictions. The optimal $\sigma$ for a given target radius $R$ scales as $\sigma^* \propto R$. For $\ell_\infty$ threat models (the more common practical concern), the certificate shrinks by a factor of $\sqrt{d}$, making randomized smoothing essentially useless for high-dimensional $\ell_\infty$ robustness.

---

## Lipschitz Networks and What They Trade Away

A classifier with global Lipschitz constant $L$, meaning $\lVert f(x) - f(x') \rVert_2 \leq L \lVert x - x' \rVert_2$ for all $x, x'$, satisfies a robustness certificate by construction: if the margin at $x$ is $\gamma(x) = f(x)_{y} - \max_{y'\neq y} f(x)_{y'}$, then the certified $\ell_2$ radius is $\gamma(x)/(2L)$.

For a deep network with layers $f = f_K \circ \cdots \circ f_1$, the Lipschitz constant satisfies $\text{Lip}(f) \leq \prod_k \sigma_{\max}(W_k)$ where $\sigma_{\max}(W_k)$ is the spectral norm of the $k$-th weight matrix. **Spectral normalization** constrains each $\sigma_{\max}(W_k) \leq 1$, ensuring $\text{Lip}(f) \leq 1$. The problem is that this is extremely aggressive: a network with unit Lipschitz constant can only assign confidence proportional to the $\ell_2$ distance to the decision boundary, and natural data distributions do not organize themselves in $\ell_2$-separated clusters.

Empirically, spectrally-normalized networks on CIFAR-10 achieve roughly 55% clean accuracy with $R = 0.14$ certificates, compared to 67% clean accuracy with $R = 0.5$ certificates from randomized smoothing. The Lipschitz approach gives tighter guarantees per unit accuracy loss at small radii, but does not scale to larger radii as gracefully. Orthogonal networks (where $W_k^T W_k = I$ at every layer) give exactly unit spectral norm while preserving more representational power, and recent work on Cayley parameterizations achieves 69% clean accuracy with $R = 0.36$ certificates.

The fundamental tension is this: a classifier cannot be simultaneously accurate and robust under arbitrary perturbations if the threat model allows perturbations large enough to reach other-class examples. This is not a statement about architectures or training procedures. It is a theorem.

---

## The Accuracy-Robustness Tradeoff Is a Theorem

**Zhang et al. (2019)** formalized the tradeoff. For any classifier and any distribution $\mathcal{D}$:

$$R_{\text{adv}}(f, \epsilon) \geq R(f) + \underbrace{\mathbb{E}_x\!\left[\max_{c \neq f(x)} P\!\left(\mathcal{B}(x,\epsilon) \cap \{x' : y(x') = c\}\right)\right]}_{\text{boundary overlap}} - \eta^*$$

where $\eta^*$ is the Bayes error and $\mathcal{B}(x,\epsilon)$ is the perturbation ball. The middle term, **boundary overlap**, measures how much probability mass from other classes lies within distance $\epsilon$ of each test point. If two classes have examples that are $\epsilon$-close, any classifier must either make errors on clean inputs (putting the boundary through that region) or make errors on adversarial inputs (being fooled by the perturbation across the boundary).

For CIFAR-10 with $\epsilon = 8/255$ in $\ell_\infty$, a theoretical estimate of the overlap term gives a minimum achievable adversarial error of roughly 5%, compared to a Bayes clean error below 0.5%. The gap between those numbers, roughly a factor of 10 in irreducible error, is an inherent property of the data distribution at this perturbation radius. No training procedure can close it. The ceiling on robustness is set by geometry, not by model capacity.

---

## Concentration of Measure and the Inevitability of Adversarial Examples

The deepest explanation for adversarial vulnerability is a pure probability theorem with no reference to learning at all. **Lévy's theorem** on the concentration of measure on the sphere states: for any 1-Lipschitz function $f: S^{d-1} \to \mathbb{R}$ with median $M$,

$$\mu\!\left(\{\lvert f(x) - M \rvert \geq t\}\right) \leq 2\exp\!\left(-\frac{(d-2)t^2}{2}\right)$$

The measure concentrates around the median with Gaussian tails, and the width of the concentration band shrinks as $O(1/\sqrt{d})$. For a binary classifier, the median is 0 or 1, and most of the sphere lies within $O(1/\sqrt{d})$ of the equator, the decision boundary.

In $d = 10^6$ dimensions, the typical distance from a uniformly random point to the decision boundary of any continuous binary function is of order $10^{-3}$. This is independent of how the classifier was trained. Any continuous decision function on a high-dimensional sphere has adversarial examples within $O(1/\sqrt{d})$ of most correctly classified points, not because the model is poorly trained, but because the sphere has that shape.

The practical consequence is that adversarial vulnerability in high dimensions is not evidence of poor generalization or insufficient training. It is a consequence of the same high dimensionality that makes these spaces so expressive. The only models that escape concentration of measure are those that rely on very few features (sparse classifiers) or those that operate in a threat model small enough to stay in the concentrated band. Both approaches sacrifice something: sparse classifiers ignore available features, and small threat models do not defend against the perturbations that matter empirically.

---

## Complete Verification and Its Limits

Empirical defenses (adversarial training, data augmentation) can raise the cost of an attack but cannot prove robustness. The alternative, **complete verification**, is to check, for every test point, that no perturbation within the threat model changes the prediction.

For ReLU networks, this can be cast as a mixed-integer linear program (MILP). Each ReLU unit has an activation pattern (on/off), and the $2^N$ possible patterns for $N$ hidden neurons define $2^N$ linear regions. Verifying robustness within each region is a linear program (LP); verifying across all regions is the exponentially hard MILP. Branch-and-bound with LP relaxations (the LP relaxes each ReLU to an interval constraint, giving an outer approximation of the reachable output set) gives the best practical performance.

Practically, complete verification is feasible for networks with up to a few thousand neurons. For the networks used in practice, ResNets with millions of parameters, complete verification is computationally intractable. **Interval Bound Propagation (IBP)**, which propagates simple interval bounds through each layer, scales to large networks but is conservative: the verified radius it computes is typically far smaller than the true robust radius because interval arithmetic ignores correlations between neurons. An IBP-verified ResNet-50 achieves roughly 36% certified accuracy at $\epsilon = 2/255$ on CIFAR-10, while adversarially trained models without certificates achieve 67% empirical robustness at the same radius.

The gap between certified and empirical robustness reflects the looseness of the certificate, not fraud, but proof theory: IBP is sound (certified inputs are truly robust) but incomplete (non-certified inputs may also be robust). Tightening the gap is an active research area with a clear character: it requires tighter approximations of the reachable set of activations, which in turn requires more computation per verification query.

---

## The Shape of the Problem

Putting the pieces together, adversarial robustness resolves into three distinct sub-problems that live at different levels of abstraction.

The first is **geometric**: the decision boundary of any accurate classifier on natural data is close to most test points, because natural data distributions of different classes overlap at the scale of humanly-imperceptible perturbations. This is a property of the data, not the model.

The second is **computational**: even if we had a perfect classifier with the maximum achievable robust radius, verifying its robustness requires solving problems that are NP-hard in general. The best we can do in practice is outer approximations that trade completeness for scalability.

The third is **statistical**: there is an inherent tradeoff between standard accuracy and adversarial robustness, because robustness requires separating classes in the perturbation metric, and class overlap makes this separation imperfect. The tradeoff is not an artifact of current methods. It is a lower bound derived from the data distribution.

This is not a pessimistic picture. It tells us precisely what we can and cannot improve. We cannot change the geometry of the data, but we can choose perturbation metrics that better reflect human perception, reducing the class overlap at the relevant scale. We cannot avoid NP-hardness in general, but we can design architectures that are easier to verify (Lipschitz, orthogonal, interval-analyzable). And we cannot close the accuracy-robustness gap below its fundamental lower bound, but we can get closer to it.

Understanding what is mathematically inevitable is the first step toward knowing where the remaining room for progress lies.
