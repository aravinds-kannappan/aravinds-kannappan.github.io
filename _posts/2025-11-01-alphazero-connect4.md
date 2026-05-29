---
title: "Implementing AlphaZero for Connect Four: MCTS + Neural Policy in C++ and Python"
date: 2025-11-01
permalink: /posts/2025/11/alphazero-connect4/
tags:
  - reinforcement learning
  - MCTS
  - C++
  - deep learning
---

Around iteration 40 of training, something changed. The agent, which had been playing essentially random Connect Four with a mild center preference, started blocking threats it had no reason to know about. A human playing against it dropped a piece that created a diagonal three-in-a-row. The agent, on its next move, dropped a piece that blocked the winning extension. Not because it had been told about diagonals. Because 40 iterations of self-play had accumulated enough evidence that unblocked diagonals eventually lead to losses.

That moment clarified something about AlphaZero that the paper doesn't quite communicate: the strategic knowledge is not programmed in. It is not emergent from clever reward shaping. It is the natural residue of a search algorithm and a neural network cooperating for long enough that patterns solidify. This post is about how that cooperation is engineered.

---

## The Algorithm: Why MCTS and a Neural Network Need Each Other

Pure Monte Carlo Tree Search, without a neural network, estimates position value by playing random games from that position and averaging outcomes. It works, but in games where winning requires strategic sequences of 5–8 moves, random play almost never reaches those positions. The signal is too sparse. A uniformly random Connect Four player wins by accident more often than by strategy.

A pure neural network, trained to evaluate positions statically, misses tactical combinations entirely. It can learn that a center column is generally good, but a 3-move forced win starting from column 4 requires lookahead that static evaluation cannot provide.

AlphaZero couples them: the network provides a **prior policy** $p(a \mid s)$ over moves and a **value estimate** $v(s)$ that replaces random rollouts. MCTS uses these to direct search, nodes with high prior get explored sooner; nodes with high value get reinforced. After hundreds of simulations from the root position, the visit counts encode a policy that is strictly better than either component alone.

The selection criterion at each node is PUCT (Predictor + UCT):

$$\text{score}(s, a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

$Q(s,a)$ is the running average value from previous simulations through action $a$. $P(s,a)$ is the neural network's prior. The second term is an exploration bonus that decays as $N(s,a)$ grows, heavily explored actions stop receiving the bonus. The constant $c_{\text{puct}} \approx 1.5$ controls the exploration-exploitation tradeoff.

```cpp
float MCTSNode::puct_score(float c_puct) const {
    float parent_n = parent ? (float)parent->visit_count : 1.f;
    float u = c_puct * prior * std::sqrt(parent_n) / (1.f + visit_count);
    return Q() + u;
}
```

This is the entire selection logic. Everything else in MCTS, expansion, backpropagation, the tree itself, is bookkeeping around this formula.

---

## The Architecture: What the Network Needs to Output

The network sees the board as a 3×6×7 tensor: one plane for the current player's pieces, one for the opponent's, one constant plane indicating who is to move (a convention from AlphaGo). It outputs two heads:

- **Policy head**: 7 logits, one per column. Passed through softmax to get $P(a \mid s)$.
- **Value head**: a single scalar in $[-1, 1]$ representing estimated win probability for the current player.

The network body is a residual tower: 5 residual blocks of 64 channels, each with two 3×3 convolutions and batch norm. This is small by AlphaZero standards (the original used 20 blocks and 256 channels for Go), but sufficient for Connect Four's search complexity.

The loss during training combines both heads:

$$\mathcal{L} = \underbrace{(z - v)^2}_{\text{value}} - \underbrace{\pi^T \log p}_{\text{policy}} + c\,\lVert\theta\rVert^2$$

where $z$ is the actual game outcome, $v$ is the value head's prediction, $\pi$ is the visit-count policy from MCTS, and $p$ is the policy head's output. The policy loss is cross-entropy against MCTS visit counts, the network is trained to predict not what moves are good in isolation, but what moves MCTS has found most valuable after extensive search.

---

## Self-Play: The Training Loop

Each training iteration: play 25 games via MCTS self-play, add the resulting (state, MCTS policy, game outcome) triples to a replay buffer, then train on 50 minibatches sampled from the buffer. The network that generates the training data is the same network being trained, there is no separate target network.

One subtlety: in early moves (before move 10), actions are sampled proportionally to visit counts, injecting exploration. In later moves, the best action is chosen greedily. This mirrors AlphaZero's temperature schedule and prevents the agent from converging to a single opening strategy.

Dirichlet noise is added to the root node's priors before each search ($\alpha = 0.3$, $\epsilon = 0.25$). This ensures the agent considers moves the neural network thinks are poor, without it, the search quickly becomes myopic, over-relying on the policy head's priors and failing to discover refutations.

---

## What Emerged

Training overnight on a MacBook Pro (M2), 100 iterations, 200 MCTS simulations per move:

| Iteration | Win rate vs random | Win rate vs prev. self |
|-----------|-------------------|----------------------|
| 0         | 62%               |,                    |
| 10        | 71%               |,                    |
| 25        | 84%               | 58%                  |
| 50        | 93%               | 64%                  |
| 75        | 97%               | 61%                  |
| 100       | 98%               | 59%                  |

The win rate against random play plateaus around 97–98%, a random Connect Four opponent wins occasionally by accident, which is a ceiling. The more meaningful metric is win rate against the previous version of self, which stabilizes around 59–64% by iteration 50: each version is modestly better than its predecessor, as expected from incremental self-play improvement.

The strategic patterns that emerged, in roughly chronological order:

1. **Center column preference** (iterations 5–15): the agent develops a strong prior for columns 3 and 4. This is optimal, center pieces connect in more directions, and emerged purely from self-play statistics.

2. **Threat blocking** (iterations 20–40): the agent consistently blocks opponent three-in-a-rows, even diagonal ones it had not been specifically trained to recognize.

3. **Fork construction** (iterations 50–80): the agent begins creating positions with two simultaneous winning threats, a basic tactic that requires 3–4 move lookahead. Against a human opponent, forks are decisive; they cannot both be blocked.

4. **Zugzwang awareness** (iterations 80+): the agent starts avoiding moves that are locally neutral but strategically poor, moves that give the opponent a forced win in 6–8 moves. This is the hardest pattern to acquire because it requires very deep MCTS search to see the eventual consequence.

None of these were explicitly programmed. The game rules are the only domain knowledge. Everything else, the geometry of threats, the concept of a fork, the strategic value of center control, crystallized from millions of simulated games.

---

## The Surprising Part

The most surprising result was not that the agent learned to play well. It was how *fast* the knowledge accumulated. The blocking behavior appeared at iteration 20, after roughly 500 games of self-play, about 15,000 board positions. A human child learning Connect Four would see far fewer positions before developing similar instincts. But the human is also doing something very different: bringing language, causal reasoning, and analogical transfer from other games. The agent has only the statistics of its own experience.

What both have in common: neither was told the rules of strategy. Both inferred them from the structure of the game.
