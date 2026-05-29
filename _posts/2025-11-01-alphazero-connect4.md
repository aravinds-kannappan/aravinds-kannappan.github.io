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

AlphaZero's paper describes a system that learns to play any two-player game from scratch — no domain knowledge, just self-play and a neural network. The algorithm sounds almost magical until you implement it. Then it becomes clear: the genius is in how MCTS and a neural network are coupled, each addressing the other's weakness.

I implemented AlphaZero for Connect Four with the tree search in C++ (fast enough for thousands of simulations per move) and the neural network in PyTorch. After training overnight on a laptop, it develops recognizable strategic patterns — defending threats, building forced wins, recognizing game-theoretic traps. This post walks through every component.

---

## The Algorithm: Why MCTS + Neural Network?

**Pure MCTS** (without a neural network) navigates a game tree by sampling random rollouts to estimate node value. It works, but converges slowly in games with long winning sequences — random play rarely stumbles onto the strategic insight that row 4, column 4 is the critical square.

**Pure neural network** (trained on position evaluation alone) generalizes well but has no lookahead — it evaluates positions statically and misses forced wins that require 5-move lookahead.

**MCTS + neural network** (AlphaZero): the network provides two things:
1. **Prior policy p(a|s):** probability distribution over legal moves, used to guide MCTS tree expansion
2. **Value estimate v(s):** estimated win probability, used in place of random rollouts

MCTS uses these to build a tree: nodes with high prior probability are explored sooner; nodes with high value are preferred. After many simulations, the visit counts encode a refined policy that is better than either component alone.

---

## C++: The Game Engine and MCTS

```cpp
// connect4.hpp
#pragma once
#include <array>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <optional>
#include <random>

constexpr int ROWS = 6, COLS = 7;
using Board = std::array<std::array<int8_t, COLS>, ROWS>;
// 0 = empty, 1 = player 1, -1 = player 2

struct GameState {
    Board board{};
    int  current_player = 1;   // +1 or -1
    int  last_col = -1;
    bool terminal = false;
    float outcome = 0.f;       // +1 = player 1 wins, -1 = player 2 wins, 0 = draw
    
    GameState() { for (auto& row : board) row.fill(0); }
    
    std::vector<int> legal_moves() const {
        std::vector<int> moves;
        for (int c = 0; c < COLS; c++)
            if (board[0][c] == 0) moves.push_back(c);
        return moves;
    }
    
    // Drop a piece in column c, return new state
    GameState apply(int col) const {
        GameState next = *this;
        for (int r = ROWS-1; r >= 0; r--) {
            if (next.board[r][col] == 0) {
                next.board[r][col] = current_player;
                next.last_col = col;
                next.current_player = -current_player;
                next.check_terminal(r, col);
                return next;
            }
        }
        assert(false && "Illegal move");
    }
    
    void check_terminal(int row, int col) {
        int player = -current_player;  // just placed
        auto in_bounds = [](int r, int c) { return r>=0 && r<ROWS && c>=0 && c<COLS; };
        
        const int dirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
        for (auto [dr, dc] : dirs) {
            int count = 1;
            for (int d : {-1, 1}) {
                int r = row+d*dr, c = col+d*dc;
                while (in_bounds(r,c) && board[r][c] == player) {
                    count++; r+=d*dr; c+=d*dc;
                }
            }
            if (count >= 4) { terminal = true; outcome = (float)player; return; }
        }
        if (legal_moves().empty()) { terminal = true; outcome = 0.f; }
    }
    
    // Encode board as float tensor for neural network: 3 × ROWS × COLS
    // Channel 0: current player's pieces, Channel 1: opponent's pieces, Channel 2: all ones (side to play)
    std::vector<float> to_tensor() const {
        std::vector<float> t(3 * ROWS * COLS, 0.f);
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (board[r][c] == current_player)
                    t[0 * ROWS*COLS + r*COLS + c] = 1.f;
                else if (board[r][c] == -current_player)
                    t[1 * ROWS*COLS + r*COLS + c] = 1.f;
                t[2 * ROWS*COLS + r*COLS + c] = 1.f;  // constant plane
            }
        }
        return t;
    }
};

// ─────────────────────────────────────────────
// MCTS Node
// ─────────────────────────────────────────────

struct MCTSNode {
    GameState state;
    MCTSNode* parent = nullptr;
    int action = -1;           // action that led to this node
    
    float prior = 0.f;         // P(action | parent state) from neural network
    float value_sum = 0.f;
    int   visit_count = 0;
    
    std::vector<MCTSNode*> children;
    bool expanded = false;
    
    float Q() const {
        return visit_count > 0 ? value_sum / visit_count : 0.f;
    }
    
    // PUCT selection criterion (AlphaZero variant of UCT)
    // Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    float puct_score(float c_puct = 1.5f) const {
        float parent_visits = parent ? (float)parent->visit_count : 1.f;
        float u = c_puct * prior * std::sqrt(parent_visits) / (1.f + visit_count);
        return Q() + u;
    }
    
    MCTSNode* best_child() const {
        MCTSNode* best = nullptr;
        float best_score = -1e9f;
        for (MCTSNode* child : children) {
            float score = child->puct_score();
            if (score > best_score) { best_score = score; best = child; }
        }
        return best;
    }
    
    ~MCTSNode() { for (auto* c : children) delete c; }
};

// ─────────────────────────────────────────────
// MCTS Search
// ─────────────────────────────────────────────

using PolicyValue = std::pair<std::vector<float>, float>;
using NetworkFn = std::function<PolicyValue(const GameState&)>;

struct MCTS {
    int n_simulations;
    float c_puct;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::mt19937 rng{42};
    
    MCTS(int n_sims = 800, float c = 1.5f, float alpha = 0.3f, float eps = 0.25f)
        : n_simulations(n_sims), c_puct(c),
          dirichlet_alpha(alpha), dirichlet_epsilon(eps) {}
    
    // Run n_simulations from root, return visit-count policy
    std::vector<float> search(const GameState& root_state, NetworkFn net_fn) {
        MCTSNode* root = new MCTSNode{root_state};
        
        // Get prior from network for root
        auto [policy, value] = net_fn(root_state);
        expand(root, policy);
        add_dirichlet_noise(root);
        
        for (int sim = 0; sim < n_simulations; sim++) {
            MCTSNode* node = root;
            
            // Selection: follow PUCT until unexpanded or terminal
            while (node->expanded && !node->state.terminal)
                node = node->best_child();
            
            float leaf_value;
            if (node->state.terminal) {
                // Back-propagate the actual outcome
                leaf_value = node->state.outcome;
                // Flip to current node's perspective
                if (node->state.current_player != root_state.current_player)
                    leaf_value = -leaf_value;
            } else {
                // Expansion + evaluation
                auto [p, v] = net_fn(node->state);
                expand(node, p);
                leaf_value = v;
            }
            
            // Backpropagation
            MCTSNode* n = node;
            while (n != nullptr) {
                n->visit_count++;
                // Value is from current player's perspective; flip at each level
                n->value_sum += (n->state.current_player == root_state.current_player)
                                 ? leaf_value : -leaf_value;
                n = n->parent;
            }
        }
        
        // Return visit-count policy (softmax of visit counts)
        auto moves = root_state.legal_moves();
        std::vector<float> pi(COLS, 0.f);
        int total = 0;
        for (MCTSNode* child : root->children)
            total += child->visit_count;
        for (MCTSNode* child : root->children)
            pi[child->action] = (float)child->visit_count / total;
        
        delete root;
        return pi;
    }
    
private:
    void expand(MCTSNode* node, const std::vector<float>& policy) {
        for (int col : node->state.legal_moves()) {
            MCTSNode* child = new MCTSNode{node->state.apply(col), node, col};
            child->prior = policy[col];
            node->children.push_back(child);
        }
        node->expanded = true;
    }
    
    // Add Dirichlet noise to root priors for exploration diversity
    void add_dirichlet_noise(MCTSNode* root) {
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.f);
        std::vector<float> noise(root->children.size());
        float noise_sum = 0.f;
        for (auto& n : noise) { n = gamma(rng); noise_sum += n; }
        for (int i = 0; i < (int)root->children.size(); i++) {
            root->children[i]->prior = (1-dirichlet_epsilon) * root->children[i]->prior
                                      + dirichlet_epsilon * (noise[i] / noise_sum);
        }
    }
};
```

---

## Python: Neural Network and Training Loop

```python
# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    """Standard residual block used in AlphaZero."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """
    Input: (batch, 3, 6, 7) — 3 planes (player, opponent, side-to-play) x board
    Output: policy (7 logits for each column), value (scalar in [-1, 1])
    """
    def __init__(self, n_res_blocks: int = 5, channels: int = 64):
        super().__init__()
        # Convolutional body
        self.conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_res_blocks)])
        
        # Policy head: (batch, channels, 6, 7) → (batch, 7)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7),
        )
        
        # Value head: (batch, channels, 6, 7) → (batch, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.res_blocks(self.conv(x))
        return self.policy_head(x), self.value_head(x)
    
    def predict(self, state_tensor: np.ndarray):
        """Single-state inference (no batch dim)."""
        x = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.forward(x)
        policy = F.softmax(policy_logits, dim=-1).squeeze().numpy()
        return policy, value.item()


# alphazero_trainer.py
import random
from collections import deque

class AlphaZeroTrainer:
    def __init__(self, net: AlphaZeroNet, n_simulations: int = 200):
        self.net = net
        self.n_simulations = n_simulations
        self.optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=50_000)  # (state_tensor, pi, z)
    
    def self_play_game(self) -> list:
        """Play one game via MCTS self-play. Returns list of (state, pi, z) tuples."""
        from connect4 import GameState, MCTS  # C++ extension via pybind11
        
        state = GameState()
        mcts  = MCTS(n_simulations=self.n_simulations)
        
        game_history = []
        
        while not state.terminal:
            # Get MCTS policy
            net_fn = lambda s: self.net.predict(s.to_tensor())
            pi = mcts.search(state, net_fn)
            
            # Temperature: explore early, exploit late
            move_num = sum(1 for r in range(6) for c in range(7) if state.board[r][c] != 0)
            if move_num < 10:
                # High temperature: sample proportionally
                action = random.choices(range(7), weights=pi)[0]
            else:
                # Low temperature: greedy
                action = int(np.argmax(pi))
            
            # Store (state, pi) — z filled in after game ends
            game_history.append([state.to_tensor(), pi, None])
            state = state.apply(action)
        
        # Fill in outcomes
        outcome = state.outcome
        for i, (s_tensor, pi, _) in enumerate(game_history):
            # Value target from the perspective of the player who moved at step i
            player_at_step = 1 if i % 2 == 0 else -1
            z = outcome * player_at_step
            game_history[i][2] = z
        
        return game_history
    
    def train_step(self, batch_size: int = 256):
        if len(self.replay_buffer) < batch_size:
            return None
        
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        pi_targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        z_targets  = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)
        
        policy_logits, values = self.net(states)
        
        # Policy loss: cross-entropy with MCTS visit counts
        policy_loss = F.cross_entropy(policy_logits, pi_targets)
        # Value loss: MSE with game outcome
        value_loss  = F.mse_loss(values, z_targets)
        # Total loss (AlphaZero paper uses equal weighting)
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
    
    def train(self, n_iterations: int = 100, games_per_iter: int = 25):
        for iteration in range(n_iterations):
            # Self-play
            new_examples = 0
            for _ in range(games_per_iter):
                game = self.self_play_game()
                self.replay_buffer.extend(game)
                new_examples += len(game)
            
            # Train
            losses = []
            for _ in range(50):  # gradient steps per iteration
                result = self.train_step()
                if result: losses.append(result)
            
            avg_loss = np.mean([l['policy_loss'] + l['value_loss'] for l in losses])
            print(f"Iter {iteration:3d} | "
                  f"Buffer: {len(self.replay_buffer):6d} | "
                  f"New: {new_examples:4d} | "
                  f"Loss: {avg_loss:.4f}")
```

---

## TypeScript: Game Server for Interactive Play

```typescript
// game-server.ts — serve the trained model for browser-based play
import express from 'express';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const app = express();
app.use(express.json());
app.use(express.static('public'));

interface BoardState {
    board: number[][];
    currentPlayer: number;
}

// Call Python subprocess for model inference
async function getModelMove(state: BoardState): Promise<number> {
    const stateJson = JSON.stringify(state);
    const { stdout } = await execFileAsync('python3', ['inference.py', stateJson]);
    return parseInt(stdout.trim());
}

app.post('/move', async (req, res) => {
    try {
        const state: BoardState = req.body;
        const move = await getModelMove(state);
        res.json({ move });
    } catch (err) {
        res.status(500).json({ error: String(err) });
    }
});

app.listen(3000, () => console.log('Connect Four server running on :3000'));
```

---

## What Emerges From Self-Play

After 50 iterations of self-play on a laptop (~3 hours), the agent exhibits:

1. **Threat recognition:** it identifies and blocks three-in-a-rows consistently by move 30
2. **Fork construction:** it creates two simultaneous winning threats — a basic tactic that requires 3-4 move lookahead
3. **Center preference:** it develops a strong prior for the center column, which is optimal — the center maximizes future winning possibilities

The network hasn't been told any of this. The center preference emerged from self-play discovering that center play leads to more winning positions. The IB framing applies here too: the policy head learned to extract exactly the board features relevant to winning moves — pieces, threats, and structure — while the value head learned to compress the entire board state into a single win probability.
