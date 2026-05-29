---
title: "Reinforcement Learning from First Principles: MDPs, Bellman Equations, and Policy Gradients"
date: 2025-11-01
permalink: /posts/2025/11/reinforcement-learning-first-principles/
tags:
  - reinforcement learning
  - mathematics
  - machine learning
---

Reinforcement learning is often taught through algorithms — Q-learning, PPO, DDPG — without making clear the mathematical structure that unifies them. This post derives RL from first principles: starting with the Markov Decision Process, building up to the Bellman equations, and deriving policy gradient methods from the score function estimator.

## The Markov Decision Process

An MDP is a tuple (S, A, P, R, γ) where:
- **S**: state space
- **A**: action space  
- **P(s'|s,a)**: transition probability — the probability of reaching s' from s after taking action a
- **R(s,a,s')**: reward function
- **γ ∈ [0,1)**: discount factor

The Markov property — P(sₜ₊₁|sₜ,aₜ,...,s₀,a₀) = P(sₜ₊₁|sₜ,aₜ) — says the future is conditionally independent of the past given the present state. This is an assumption, not a physical law, but it structures the problem in a tractable way.

A **policy** π(a|s) maps states to distributions over actions. The goal is to find the policy that maximizes expected discounted return:

```
J(π) = 𝔼_π[Σₜ γᵗ R(sₜ, aₜ, sₜ₊₁)]
```

The discount factor γ < 1 ensures the sum is finite for infinite-horizon problems and encodes a preference for immediate rewards over distant ones.

## Value Functions

The **state-value function** V^π(s) is the expected return starting from s and following π:

```
V^π(s) = 𝔼_π[Σₜ γᵗ Rₜ | s₀ = s]
```

The **action-value function** Q^π(s,a) is the expected return starting from s, taking action a, then following π:

```
Q^π(s,a) = 𝔼_π[Σₜ γᵗ Rₜ | s₀ = s, a₀ = a]
```

The relationship between them: V^π(s) = Σₐ π(a|s) Q^π(s,a) — the value of a state is the expectation of Q over the policy.

## The Bellman Equations

The key structural property of value functions is the **Bellman consistency equation**:

```
V^π(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

This says: the value of a state equals the expected immediate reward plus the discounted value of the next state. It's a self-consistency condition — V^π is the fixed point of the Bellman operator T^π:

```
(T^π V)(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

T^π is a contraction mapping (||T^π V - T^π U||∞ ≤ γ||V-U||∞), so iterating it converges to V^π — this is policy evaluation.

The **Bellman optimality equation** characterizes the optimal value function V*:

```
V*(s) = max_a Σₛ' P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

The optimal policy is greedy with respect to V*: π*(s) = argmax_a Q*(s,a). Q-learning directly estimates Q* using the Bellman optimality operator.

## Q-Learning and the Deep Q-Network

Q-learning updates toward the Bellman target:

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

The term in brackets is the **TD error** δ — the difference between the current Q-estimate and what Bellman says it should be. DQN (Deep Q-Network) approximates Q with a neural network and stabilizes training using:
1. **Experience replay:** sample transitions uniformly from a replay buffer to break temporal correlations
2. **Target network:** use a periodically-updated frozen copy of Q for the Bellman target, preventing the moving-target problem

```python
def dqn_update(Q, target_Q, batch, gamma, optimizer):
    states, actions, rewards, next_states, dones = batch
    
    # Current Q values
    q_values = Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target: r + γ max_a' Q_target(s', a')  (0 for terminal states)
    with torch.no_grad():
        next_q = target_Q(next_states).max(1)[0]
        targets = rewards + gamma * next_q * (1 - dones)
    
    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## Policy Gradient: Directly Optimizing the Policy

Value-based methods learn Q and derive the policy implicitly. Policy gradient methods directly parameterize the policy π_θ and optimize J(θ) by gradient ascent.

The challenge: J(θ) involves an expectation over trajectories whose distribution depends on θ — we can't straightforwardly differentiate through the sampling process.

The **policy gradient theorem** gives us the gradient:

```
∇_θ J(θ) = 𝔼_π[∇_θ log π_θ(a|s) · Q^π(s,a)]
```

**Derivation via the score function estimator (REINFORCE trick):**

```
∇_θ J = ∇_θ 𝔼_τ~π_θ [R(τ)]
       = ∇_θ ∫ p_θ(τ) R(τ) dτ
       = ∫ R(τ) ∇_θ p_θ(τ) dτ
       = ∫ R(τ) p_θ(τ) ∇_θ log p_θ(τ) dτ    [log-derivative trick]
       = 𝔼_τ~π_θ [R(τ) ∇_θ log p_θ(τ)]
```

Since log p_θ(τ) = Σₜ log π_θ(aₜ|sₜ) + log P(sₜ₊₁|sₜ,aₜ), and the transition log-probabilities don't depend on θ:

```
∇_θ log p_θ(τ) = Σₜ ∇_θ log π_θ(aₜ|sₜ)
```

This gives the REINFORCE estimator: for each trajectory, compute the return, then push up the log-probability of actions proportional to how good the return was.

## Variance Reduction: Baselines and Advantage

REINFORCE has high variance — a single trajectory's return is a noisy estimate of Q^π. The **advantage function** A^π(s,a) = Q^π(s,a) - V^π(s) subtracts a baseline:

```
∇_θ J(θ) = 𝔼_π[∇_θ log π_θ(a|s) · A^π(s,a)]
```

Subtracting any function of s (not a) from Q^π yields an unbiased estimator of the gradient — the baseline doesn't change the expectation. But it reduces variance by centering the returns: actions better than average get positive weight, actions worse than average get negative weight.

In practice, V^π (the critic) is learned alongside π (the actor) — this is the Actor-Critic family of algorithms.

## PPO: Stable Policy Updates

Vanilla policy gradient takes steps proportional to the gradient, but large steps can collapse the policy. **Proximal Policy Optimization (PPO)** constrains each update to stay close to the old policy:

```
L^CLIP(θ) = 𝔼[min(rₜ(θ) Aₜ, clip(rₜ(θ), 1-ε, 1+ε) Aₜ)]
```

where rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ) is the importance ratio. Clipping prevents the ratio from straying too far from 1 — large policy changes are penalized. PPO is the workhorse of modern RLHF precisely because this clipping makes it stable enough to fine-tune large language models without catastrophic forgetting.

## Why RL Is Hard (and When It's Worth It)

RL requires solving three problems simultaneously:
1. **Exploration:** how to collect informative data when you don't know which actions are good
2. **Credit assignment:** which actions caused the eventual reward? (Temporal credit assignment over long horizons)
3. **Function approximation:** how to generalize Q or π across the (often continuous) state space

Each of these is a research area unto itself. In practice, RL succeeds when: reward is dense (feedback every step, not just at the end), the environment is simulatable (so you can collect huge amounts of experience), and the state representation is well-chosen. When any of these fail, RL training becomes extremely sample-inefficient. This is why RLHF works for LLMs — human preference labels provide dense feedback, the "environment" is cheap (just run inference), and the text representation is already rich.
