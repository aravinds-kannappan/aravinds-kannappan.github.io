---
title: "The Information Bottleneck: Deriving Optimal Representations From First Principles"
date: 2025-10-01
permalink: /posts/2025/10/information-bottleneck/
tags:
  - information theory
  - representation learning
  - mathematics
---

In 1999, Tishby, Pereira, and Bialek posed a deceptively simple question: *what is the optimal representation of X for predicting Y?* Their answer — the Information Bottleneck — defines optimality in terms of mutual information, derives the representation as a tradeoff curve, and predicts a phase transition in learning that has since been observed empirically in deep networks.

I found this paper while trying to understand why certain learned representations generalized better than others, and it gave me a framework that changed how I think about feature selection, regularization, and compression. This post derives the IB from scratch, implements MINE (Mutual Information Neural Estimator) to measure it empirically, and shows how the information plane reveals what's happening inside a neural network during training.

---

## What Is Mutual Information?

The mutual information between X and Y measures how much knowing one reduces uncertainty about the other:

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

Equivalently, it's the KL divergence between the joint and the product of marginals:

```
I(X; Y) = KL(P(X,Y) || P(X)P(Y)) = 𝔼_{x,y}[log(P(x,y) / P(x)P(y))]
```

MI is zero iff X and Y are independent; it's symmetric (I(X;Y) = I(Y;X)), and it captures nonlinear dependencies that correlation misses.

For a concrete intuition: if X is an image of a digit and Y is the digit label, I(X;Y) is the information the image contains about the label. A representation Z = f(X) can only lose information: by the data processing inequality, I(Z;Y) ≤ I(X;Y). The IB asks: what's the minimal Z that preserves I(Z;Y) ≈ I(X;Y)?

---

## The Information Bottleneck Principle

Define a compressed representation Z of X that predicts Y. Z is constrained to be a Markov chain: Y → X → Z (Z is computed from X, not directly from Y).

The IB tradeoff: minimize the information Z contains about X (compression) while maximizing the information Z contains about Y (relevance):

```
min_{p(z|x)} I(X; Z) - β · I(Z; Y)
```

The Lagrange multiplier β controls the tradeoff. At β=0, Z discards everything; at β→∞, Z = X (no compression).

**The IB self-consistent equations** (Tishby et al. 1999):

```
p(z|x) = p(z)/Z(x,β) · exp(-β · KL(p(y|x) || p(y|z)))
p(z) = Σ_x p(x) p(z|x)
p(y|z) = Σ_x p(y|x) p(x|z)
```

These are solved iteratively (like EM) to produce the optimal encoder p(z|x) for each β. The result is a family of representations — the IB curve — parameterized by β.

---

## Estimating Mutual Information With Neural Networks

For continuous X and Z, computing MI directly is intractable — it requires estimating a joint density in high dimensions. MINE (Belghazi et al. 2018) provides a scalable lower bound using the Donsker-Varadhan representation of KL divergence:

```
KL(P||Q) = sup_T 𝔼_P[T] - log(𝔼_Q[e^T])
```

Substituting P = P(X,Z) and Q = P(X)P(Z):

```
I(X;Z) = sup_T 𝔼_{P(X,Z)}[T(x,z)] - log(𝔼_{P(X)P(Z)}[e^{T(x,z)}])
```

T is a neural network parameterized to maximize the lower bound — the higher the MI, the easier it is for T to distinguish joint samples from marginal samples.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MINE(nn.Module):
    """
    Mutual Information Neural Estimator.
    T: (x, z) -> scalar score.
    Higher score = joint sample is more distinguishable from marginal.
    """
    def __init__(self, x_dim: int, z_dim: int, hidden: int = 128):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.T(torch.cat([x, z], dim=-1))
    
    def estimate_mi(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        MINE estimator of I(X;Z).
        Joint samples: (x[i], z[i])
        Marginal samples: (x[i], z[j]) where j is a random permutation of i
        """
        joint_scores    = self.forward(x, z)
        
        # Shuffle z to get marginal samples from P(X)P(Z)
        z_perm = z[torch.randperm(z.size(0))]
        marginal_scores = self.forward(x, z_perm)
        
        # DV lower bound: E[T_joint] - log(E[e^T_marginal])
        # Use log-sum-exp for numerical stability
        mi_lower_bound = (
            joint_scores.mean()
            - torch.logsumexp(marginal_scores, dim=0) + np.log(x.size(0))
        )
        return mi_lower_bound


def train_mine_estimator(
    x: torch.Tensor, z: torch.Tensor,
    n_epochs: int = 500, lr: float = 1e-3, batch_size: int = 256
) -> list:
    mine = MINE(x.size(1), z.size(1))
    opt  = optim.Adam(mine.parameters(), lr=lr)
    mi_history = []
    
    dataset = torch.utils.data.TensorDataset(x, z)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        epoch_mi = []
        for x_batch, z_batch in loader:
            mi = mine.estimate_mi(x_batch, z_batch)
            loss = -mi  # maximize MI → minimize negative MI
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_mi.append(mi.item())
        
        if epoch % 50 == 0:
            avg_mi = np.mean(epoch_mi)
            mi_history.append(avg_mi)
            print(f"Epoch {epoch:4d} | MI estimate: {avg_mi:.4f} bits")
    
    return mi_history
```

---

## The Information Plane: Watching a Network Learn

Tishby and Schwartz-Ziv (2017) proposed visualizing the training dynamics of a deep network in the "information plane": plot I(X;T_l) vs I(T_l;Y) for each layer T_l at each training epoch.

Their finding: networks go through two phases:
1. **Fitting phase:** I(T_l;Y) increases — the network learns to predict Y
2. **Compression phase:** I(X;T_l) decreases — the network forgets irrelevant aspects of X

```python
class InformationPlaneMonitor:
    """
    Track I(X;T) and I(T;Y) for each layer during training.
    Uses MINE for continuous activations.
    """
    def __init__(self, model: nn.Module, x_dim: int, y_dim: int, hidden: int = 64):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.x_dim, self.y_dim = x_dim, y_dim
        self.history = []
        
        # Register forward hooks to capture layer activations
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self.activations.update({n: out.detach()})
                )
                self.hooks.append(hook)
    
    def measure_epoch(self, X: torch.Tensor, Y: torch.Tensor, n_epochs_mine: int = 200):
        """Run a forward pass and estimate MI for each layer."""
        with torch.no_grad():
            _ = self.model(X)
        
        epoch_info = {}
        for layer_name, Z in self.activations.items():
            # Flatten Z if needed
            z_flat = Z.view(Z.size(0), -1)
            
            # Estimate I(X;Z) and I(Z;Y)
            mine_xz = MINE(self.x_dim, z_flat.size(1))
            mine_zy = MINE(z_flat.size(1), self.y_dim)
            
            # Quick estimation (fewer epochs for speed)
            ixz = self._quick_mi(mine_xz, X, z_flat, n_epochs_mine)
            izy = self._quick_mi(mine_zy, z_flat, Y.float().unsqueeze(1), n_epochs_mine)
            
            epoch_info[layer_name] = {'I_X_Z': ixz, 'I_Z_Y': izy}
        
        self.history.append(epoch_info)
        return epoch_info
    
    def _quick_mi(self, mine, x, z, epochs):
        opt = optim.Adam(mine.parameters(), lr=2e-3)
        for _ in range(epochs):
            mi = mine.estimate_mi(x[:256], z[:256])
            (-mi).backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            return mine.estimate_mi(x, z).item()


def build_mlp_monitored(input_dim: int, hidden_dims: list, output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# Experiment: MNIST subset with information plane tracking
def run_information_plane_experiment():
    from torchvision import datasets, transforms
    
    mnist = datasets.MNIST('.', download=True, transform=transforms.ToTensor())
    X = mnist.data[:5000].float().view(5000, -1) / 255.0
    Y = mnist.targets[:5000]
    
    model = build_mlp_monitored(784, [512, 256, 128, 64], 10)
    monitor = InformationPlaneMonitor(model, x_dim=784, y_dim=1)
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    X_tensor = torch.tensor(X)
    Y_tensor = Y
    
    for epoch in range(100):
        # Training step
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Measure information plane every 10 epochs
        if epoch % 10 == 0:
            info = monitor.measure_epoch(X_tensor, Y_tensor)
            print(f"\nEpoch {epoch} | Loss: {loss.item():.4f}")
            for layer, mi in info.items():
                print(f"  {layer:20s} | I(X;T)={mi['I_X_Z']:.3f} | I(T;Y)={mi['I_Z_Y']:.3f}")
```

---

## The IB as a Regularizer

The most practical application: use the IB tradeoff as an explicit regularizer. The β-VAE objective is exactly this:

```
L = 𝔼[log p(x|z)] - β · KL(q(z|x) || p(z))
```

where `KL(q(z|x) || p(z))` is an upper bound on I(X;Z) when p(z) is the marginal. At β=1, this is the standard VAE. At β>1, the compression constraint is tightened, forcing the learned representation to discard more of X and retain only what's necessary for reconstruction.

In my own experiments, β-VAE representations at β=4 generalized substantially better to held-out distributions than β=1 representations — the additional compression acted as a form of invariance learning, discarding distribution-specific details.

```python
class BetaVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, beta: float = 4.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, input_dim),  nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def loss(self, x_hat, x, mu, log_var):
        # Reconstruction term: E[log p(x|z)]
        recon = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        # KL term: upper bound on I(X;Z)
        kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
        return recon + self.beta * kl, recon, kl
```

The IB isn't just theory — it's a lens for understanding what every regularized model is implicitly doing: trading off compression (how much of X does this representation remember?) against relevance (how much does it know about Y?).
