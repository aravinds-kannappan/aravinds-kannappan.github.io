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

The existence of adversarial examples — inputs imperceptibly different from natural data that cause dramatic mispredictions — is one of the most empirically robust and theoretically puzzling facts in deep learning. Since Szegedy et al. first demonstrated them in 2013, thousands of attack and defense papers have been published, and the central problem remains open: why do adversarial examples exist, and how can we certifiably eliminate them? This post develops the mathematical foundations of adversarial robustness from first principles — through the geometry of perturbation sets, the theory of certified defenses, Lipschitz constraints, and PAC-Bayesian robustness bounds — and argues that the phenomenon of adversarial vulnerability is not an engineering failure but a consequence of the geometry of high-dimensional spaces that admits only partial remedies.

---

## Formalizing Robustness

Let $$f: \mathcal{X} \to \mathcal{Y}$$ be a classifier mapping inputs $$x \in \mathcal{X} \subseteq \mathbb{R}^d$$ to labels $$y \in \mathcal{Y}$$. Define the **threat model** as a perturbation set $$\mathcal{B}(x, \epsilon) \subseteq \mathcal{X}$$ — the set of inputs considered indistinguishable from $$x$$ by an adversary with budget $$\epsilon$$. The most common choices are $$\ell_p$$ balls:

$$\mathcal{B}_p(x, \epsilon) = \{x' \in \mathcal{X} : \|x' - x\|_p \leq \epsilon\}$$

A classifier $$f$$ is **$$(\epsilon, p)$$-robust** at $$x$$ if:

$$f(x') = f(x) \quad \forall x' \in \mathcal{B}_p(x, \epsilon)$$

The **robust accuracy** at level $$\epsilon$$ is:

$$\text{RA}(\epsilon) = P_{(x,y) \sim \mathcal{D}}\left(\min_{x' \in \mathcal{B}(x,\epsilon)} f(x') = y\right)$$

the probability that the classifier is correct on the worst-case perturbation of each test point. This is the quantity that certified defenses seek to bound from below.

The **adversarial risk** is:

$$R_{\text{adv}}(f) = \mathbb{E}_{(x,y)}\left[\max_{x' \in \mathcal{B}(x,\epsilon)} \mathbf{1}[f(x') \neq y]\right]$$

and the **robustness gap** is $$R_{\text{adv}}(f) - R(f)$$, the difference between adversarial risk and standard risk. Most current classifiers have large robustness gaps — they have near-zero standard error but high adversarial error.

---

## The Geometry of Adversarial Examples in High Dimensions

A fundamental question: why are adversarial perturbations so small? An $$\ell_\infty$$ perturbation of magnitude $$\epsilon = 8/255 \approx 0.031$$ (the standard ImageNet threat model) is invisible to humans but sufficient to fool state-of-the-art models with near-100% success rate.

The answer is geometric. Consider a linear classifier $$f(x) = \text{sign}(w \cdot x + b)$$ in $$\mathbb{R}^d$$. The distance from a correctly classified point $$x$$ to the decision boundary is:

$$\text{dist}(x, \partial f) = \frac{|w \cdot x + b|}{\|w\|_2}$$

For an $$\ell_\infty$$ adversary, the required perturbation is:

$$\epsilon^* = \frac{|w \cdot x + b|}{\|w\|_\infty}$$

Since $$\|w\|_\infty \leq \|w\|_2 \leq \sqrt{d} \|w\|_\infty$$, we have:

$$\frac{|w \cdot x + b|}{\sqrt{d}\|w\|_\infty} \leq \epsilon^* \leq \frac{|w \cdot x + b|}{\|w\|_\infty}$$

For $$d = 10^6$$ (ImageNet dimensions), $$\epsilon^*$$ in the $$\ell_\infty$$ norm can be $$\sqrt{d} \approx 10^3$$ times smaller than the $$\ell_2$$ distance to the boundary. The model is vulnerable in $$\ell_\infty$$ because it relies on the cumulative effect of many small features — the dot product $$w \cdot x$$ aggregates contributions from all $$d$$ dimensions, and each dimension can be perturbed slightly while the sum changes substantially.

More precisely, Goodfellow et al.'s **linear hypothesis** states that adversarial examples in high dimensions are inevitable for any classifier that uses linear features: perturbing each input coordinate by $$\epsilon$$ in the direction of $$\text{sign}(w_i)$$ changes the output by $$\epsilon \|w\|_1 \geq \epsilon \sqrt{d} \|w\|_2 / \sqrt{d}$$ — growing as $$O(\epsilon d)$$ for classifiers that use all dimensions. This is unavoidable unless the classifier uses only a small number of features (sparse $$w$$) or actively ignores the remaining dimensions.

---

## Randomized Smoothing and Certified $$\ell_2$$ Defenses

**Randomized smoothing** (Cohen, Rosenfeld, Kolter, 2019) is the only method that provides scalable certified $$\ell_2$$ defenses for deep networks. The construction is elegant.

Given any base classifier $$f: \mathbb{R}^d \to \mathcal{Y}$$ (potentially non-robust), define the **smoothed classifier**:

$$g(x) = \arg\max_{c \in \mathcal{Y}} P(f(x + \delta) = c) \quad \text{where} \quad \delta \sim \mathcal{N}(0, \sigma^2 I)$$

The smoothed classifier predicts the most probable class when the input is corrupted by isotropic Gaussian noise of standard deviation $$\sigma$$.

**Theorem (Cohen et al.):** If $$g(x) = c_A$$ (the most probable class) with probability $$p_A = P(f(x+\delta)=c_A)$$, and the second-most probable class has probability $$p_B = \max_{c \neq c_A} P(f(x+\delta)=c)$$, then $$g$$ is certifiably robust within:

$$R = \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)$$

where $$\Phi^{-1}$$ is the inverse standard normal CDF. For $$x' \in \mathcal{B}_2(x, R)$$, $$g(x') = g(x) = c_A$$.

*Proof sketch:* For any $$x'$$ with $$\|x'-x\|_2 \leq R$$, the noise-corrupted distributions $$x + \delta$$ and $$x' + \delta$$ (both Gaussian) satisfy:

$$D_{\text{KL}}(\mathcal{N}(x, \sigma^2 I) \| \mathcal{N}(x', \sigma^2 I)) = \frac{\|x-x'\|_2^2}{2\sigma^2} \leq \frac{R^2}{2\sigma^2}$$

By the **Neyman-Pearson lemma**, the optimal adversarial perturbation is in the direction maximizing the log-likelihood ratio, and the robustness radius is determined by how much the probability assigned to $$c_A$$ can decrease under the KL-bounded shift. The Gaussian tail bound converts the KL constraint into the $$\Phi^{-1}$$-based expression.

The certified radius grows with $$\sigma$$ (more noise → larger certificate) but the base accuracy decreases (more noise → harder classification). The accuracy-robustness tradeoff is:

$$\text{Certified accuracy at radius } R = P\left(p_A \geq \Phi\left(\Phi^{-1}(\bar{p}_A) + \frac{R}{\sigma}\right)\right)$$

where $$\bar{p}_A$$ is the Monte Carlo estimate of $$p_A$$. Optimal $$\sigma$$ balances these competing effects and scales as $$\sigma^* \propto R$$.

A crucial limitation: the certificate is in $$\ell_2$$. For $$\ell_\infty$$ threat models (which are often more practically relevant), the Gaussian kernel is suboptimal — it certificates $$\ell_\infty$$ robustness only within radius $$R_\infty = R_2 / \sqrt{d}$$, which shrinks with dimension. **No scalable certification method for $$\ell_\infty$$ robustness with certificates comparable to empirical robustness currently exists.** This is an open problem.

---

## Lipschitz Networks and Spectral Normalization

A classifier $$f$$ with Lipschitz constant $$\text{Lip}(f) = L$$ satisfies:

$$\|f(x) - f(x')\|_2 \leq L \|x - x'\|_2 \quad \forall x, x' \in \mathcal{X}$$

For a $$K$$-class classifier $$f: \mathbb{R}^d \to \mathbb{R}^K$$, if $$f(x)_y - \max_{y' \neq y} f(x)_{y'} \geq \gamma > 0$$ (margin $$\gamma$$) and $$\text{Lip}(f) \leq L$$, then for any $$x' \in \mathcal{B}_2(x, \epsilon)$$:

$$f(x')_y - \max_{y' \neq y} f(x')_{y'} \geq \gamma - 2L\epsilon$$

The classifier is certifiably correct on $$\mathcal{B}_2(x, \epsilon)$$ when $$\epsilon < \gamma / (2L)$$. The certified radius is $$R_{\text{cert}}(x) = \frac{\gamma(x)}{2L}$$ where $$\gamma(x)$$ is the margin at $$x$$.

For a neural network with layers $$f = f_K \circ \cdots \circ f_1$$, the Lipschitz constant of the composition satisfies:

$$\text{Lip}(f) \leq \prod_{k=1}^K \text{Lip}(f_k)$$

For a linear layer $$f_k(h) = W_k h$$, $$\text{Lip}(f_k) = \|W_k\|_2 = \sigma_{\max}(W_k)$$ (the spectral norm, i.e., largest singular value). For ReLU activations, $$\text{Lip}(\text{ReLU}) = 1$$. So:

$$\text{Lip}(f) \leq \prod_{k=1}^K \sigma_{\max}(W_k)$$

**Spectral normalization** (Miyato et al.) constrains each layer's spectral norm to $$\leq 1$$ during training:

$$\bar{W}_k = \frac{W_k}{\sigma_{\max}(W_k)}$$

This guarantees $$\text{Lip}(f) \leq 1$$ but is too aggressive — unit Lipschitz constant severely limits the network's expressiveness. **Orthogonal networks** (where all weight matrices are orthogonal: $$W_k^T W_k = I$$) achieve exactly unit spectral norm at each layer while preserving more expressiveness, since orthogonal matrices preserve norms.

The tightest known Lipschitz bound for deep ReLU networks uses the **singular value decomposition** of the composition:

$$\text{Lip}(f) = \sup_{x \neq x'} \frac{\|f(x) - f(x')\|_2}{\|x-x'\|_2} = \sup_x \|J_f(x)\|_2$$

where $$J_f(x) = \prod_{k=1}^K D_k W_k$$ is the Jacobian (with $$D_k = \text{diag}(\mathbf{1}[W_k h_{k-1} \geq 0])$$ the activation mask). Computing this exactly is NP-hard, but semidefinite programming (SDP) relaxations give tight upper bounds:

$$\text{Lip}(f)^2 \leq \min_\lambda \left\{\lambda : \begin{pmatrix} \lambda I & W_K^T D_K \cdots W_1^T D_1 \\ D_1 W_1 \cdots D_K W_K & I \end{pmatrix} \succeq 0\right\}$$

---

## PAC-Bayesian Robustness Bounds

**PAC-Bayesian theory** provides generalization bounds that hold for randomized classifiers. For robustness, the PAC-Bayes framework gives bounds on the adversarial risk of a distribution $$Q$$ over classifiers.

Let $$P$$ be a prior over classifiers (before seeing data), $$Q$$ be a posterior, and $$\hat{R}_{\text{adv}}(Q)$$ the empirical adversarial risk of $$Q$$. The **PAC-Bayes theorem** states: with probability $$\geq 1 - \delta$$ over the training set of size $$n$$:

$$R_{\text{adv}}(Q) \leq \hat{R}_{\text{adv}}(Q) + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(n/\delta)}{2n}}$$

The bound involves only $$D_{\text{KL}}(Q \| P)$$ — the KL divergence between posterior and prior — not the parameter count. For overparameterized networks, $$D_{\text{KL}}(Q \| P) \ll d$$ when the posterior concentrates on a low-dimensional submanifold of parameter space (i.e., when the model has low RLCT, connecting back to SLT). The bound is tight precisely when the model is implicitly regularized by the training procedure.

Combining PAC-Bayes with Lipschitz constraints gives the **robust PAC-Bayes bound** (Viallard et al.). For a stochastic classifier $$Q_\sigma(x) = \int f(x + \delta; \theta) dQ(\theta) d\mathcal{N}(\delta; 0, \sigma^2)$$:

$$R_{\text{adv}}(Q_\sigma, \epsilon) \leq \hat{R}(Q_\sigma) + \frac{1}{\sqrt{2\pi}\sigma}\epsilon + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(n/\delta)}{2n}}$$

The three terms are: (1) empirical clean loss, (2) robustness gap proportional to $$\epsilon/\sigma$$ (attack budget divided by smoothing noise — recovers the randomized smoothing bound), and (3) generalization gap proportional to $$\sqrt{D_{\text{KL}}/n}$$.

---

## The Accuracy-Robustness Tradeoff: A Theorem

An important theoretical result: standard accuracy and adversarial robustness are in tension, and this tension is not an artifact of current methods but a consequence of data geometry.

**Theorem (Zhang et al., 2019):** For any classifier $$f$$ and any distribution $$\mathcal{D}$$ with Bayes error $$\eta^*$$:

$$R_{\text{adv}}(f, \epsilon) \geq R(f) + \mathbb{E}_x\left[\max_{c \neq f(x)} P(\text{overlap}(c, f(x), \epsilon))\right] - \eta^*$$

where $$\text{overlap}(c, c', \epsilon)$$ is the probability mass in the intersection of the $$\epsilon$$-neighborhoods of inputs with labels $$c$$ and $$c'$$.

The overlap term is the **inherent robustness barrier**: when two classes have nearby examples (as in natural datasets where semantically similar images have different labels), any classifier must be wrong either on some clean inputs or on some adversarial inputs of those inputs. Reducing robustness error requires reducing overlap — separating the classes' $$\epsilon$$-neighborhoods — which requires assigning correct labels to regions between classes, which increases the complexity of the decision boundary and thus the clean error.

Formally, define the **robust Bayes error**:

$$\eta^*_{\text{adv}}(\epsilon) = \mathbb{E}_x\left[\min_c P_{x' \sim \mathcal{B}(x,\epsilon) \cap \mathcal{D}}(y(x') \neq c)\right]$$

This is the error of the optimal robust classifier — the minimum possible adversarial error. For the standard CIFAR-10 distribution with $$\epsilon = 8/255$$ under $$\ell_\infty$$, theoretical estimates give $$\eta^*_{\text{adv}} \approx 5\%$$ robust error, compared to $$\eta^* \approx 0.5\%$$ clean error. The fundamental barrier is roughly a 10× increase in irreducible error, entirely due to class overlap at the given perturbation radius.

---

## Concentration of Measure and the Inevitability of Adversarial Examples

The deepest explanation for adversarial examples is **concentration of measure** — a high-dimensional probability phenomenon that has nothing specifically to do with neural networks.

**Lévy's theorem:** On the $$d$$-dimensional sphere $$S^{d-1}$$ with the uniform probability measure $$\mu$$ and geodesic distance, for any function $$f: S^{d-1} \to \mathbb{R}$$ with median $$M_f$ and Lipschitz constant $$L$$:

$$\mu(\{x : |f(x) - M_f| \geq t\}) \leq 2\exp\left(-\frac{(d-2)t^2}{2L^2}\right)$$

The measure concentrates exponentially in $$d$$ around the median. Equivalently: most points on a high-dimensional sphere are close to the median of any Lipschitz function. For a binary function $$f$$ (a classifier), the median is either 0 or 1, and most points are near the boundary between the two regions — near the **equator** of the sphere.

For neural network classifiers, this manifests as: for any continuous function $$f: \mathbb{R}^d \to \{0,1\}^K$$, most correctly classified test points lie near the decision boundary. The distance to the boundary concentrates around a value that decays as $$O(1/\sqrt{d})$$. In $$d = 10^6$$ dimensions, the typical distance to the decision boundary is of order $$10^{-3}$$ — far smaller than any human-perceptible perturbation.

This is the measure-theoretic proof that adversarial examples are **inevitable** for any classifier in high dimensions that uses all available features. The argument does not assume anything about the classifier's architecture or training procedure — it holds for any continuous decision function on high-dimensional input space. The only escape is to use classifiers that are explicitly constrained to ignore most dimensions (sparse classifiers) or to use a threat model $$\mathcal{B}(x,\epsilon)$$ that is small enough to avoid the concentration regime.

---

## Interval Bound Propagation and Complete Verification

**Interval Bound Propagation (IBP)** is the simplest sound (but incomplete) verification method. For a neural network layer $$h^{(k+1)} = \sigma(W_k h^{(k)} + b_k)$$, if $$h^{(k)} \in [\underline{h}^{(k)}, \overline{h}^{(k)}]$$ (componentwise), then:

$$\underline{h}^{(k+1)} = \sigma(W_k^+ \underline{h}^{(k)} + W_k^- \overline{h}^{(k)} + b_k)$$
$$\overline{h}^{(k+1)} = \sigma(W_k^+ \overline{h}^{(k)} + W_k^- \underline{h}^{(k)} + b_k)$$

where $$W^+ = \max(W, 0)$$ and $$W^- = \min(W, 0)$$. This propagates interval bounds through each layer in closed form, requiring only two forward passes per layer.

IBP is sound: if the output bounds show that the logit for the correct class exceeds all other logits, the classifier is certifiably robust. But it is loose: the intervals grow exponentially with depth due to the Lipschitz multiplication at each layer. For deep networks, IBP produces vacuous bounds unless the network is trained with IBP-aware regularization that constrains the spectral norms.

**Complete verification** (via Mixed Integer Linear Programming) is NP-hard but exact. The MILP formulation encodes the ReLU as a binary variable $$z_i \in \{0,1\}$$ (active or inactive) and solves:

$$\min_{x', z} f(x')_y - \max_{y' \neq y} f(x')_{y'}$$
subject to: $$x' \in \mathcal{B}(x, \epsilon)$$, ReLU constraints, $$z_i \in \{0,1\}$$

If the optimal value is $$\geq 0$$, the network is certifiably robust. The hardness comes from the combinatorial explosion in activation patterns: $$2^N$$ possible patterns for $$N$$ hidden neurons. Branch-and-bound with LP relaxations ($$\ell_1$$-approximation of the ReLU) provides the best practical algorithms.

---

## Philosophical Implications: Robustness as a Topological Problem

The mathematical view of adversarial robustness suggests that it is fundamentally a **topological problem**: the question of whether a decision boundary can be arranged to have margin $$\epsilon$$ everywhere, given the topology of the data distribution.

The **Jordan curve theorem** (generalized to high dimensions as the **Jordan-Brouwer separation theorem**) states that a continuous injective map $$S^{d-1} \hookrightarrow S^d$$ separates $$S^d$$ into exactly two connected components. Any decision boundary that separates two classes in $$\mathbb{R}^d$$ is such a hypersurface. The robustness of the classifier is determined by the **reach** of this hypersurface: the largest $$\epsilon$$ such that every point within distance $$\epsilon$$ of the boundary has a unique closest boundary point.

Boundaries with low reach — boundaries that come close to themselves, have sharp corners, or approach the support of the data distribution — are inherently fragile. Maximizing robustness requires maximizing reach, which requires smoothing the decision boundary away from the data support. But the data distribution constrains where the boundary must go: between classes, and close to all the class-overlapping points identified by the robustness barrier theorem.

The accurate picture of adversarial robustness, then, is this: we are trying to place a smooth, high-reach hypersurface between two interleaved, high-dimensional point clouds, where the clouds are close together in the directions of many input features. The topology allows this surface to exist, but its reach is bounded by the minimum distance between the clouds — which, in high dimensions and natural data distributions, is small. Certified defense methods are methods for provably placing the boundary at a given reach from all training and test points. The fundamental limits come from the geometry of the data, not from any deficiency in our defenses.

This is not a counsel of despair. It suggests where progress can be made: in learning data representations where the class-conditional distributions are more separated (representation learning for robustness), in defining threat models that better match human perception (semantic robustness), and in certifying robustness in the representation space rather than the pixel space. The mathematics is clear about the limits. It is equally clear about the directions in which those limits can be pushed.
