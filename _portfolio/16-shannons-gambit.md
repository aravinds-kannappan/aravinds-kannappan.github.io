---
title: "Shannon's Gambit: A Chess Engine That Learns at an Honest Rating"
excerpt: "An end-to-end chess RL system that routes each position to the method that owns it, retrains itself nightly by gated self-play, and reports only the Elo Stockfish says it can actually play."
collection: portfolio
date: 2026-06-18
---

Shannon's Gambit is a multi-agent chess engine that owns the whole pipeline. Each position is routed to the method that owns it: a learned opening book for the first moves, an AlphaZero-lite neural net for the middlegame, exact MDP/Bellman dynamic programming in solved endgames, and PPO plus DQN reinforcement learning in the low-material regime. It pre-trains on real Lichess games, improves by nightly gated self-play, and fine-tunes a personal checkpoint to how you play.

**Key work:** A router that dispatches each position to the opening book, neural net, endgame solver, or RL agent (`agents/router.py`) · Behavioural cloning on real strong-player Lichess games, then MCTS lookahead at serve time and self-play refinement across generations · Champion-gated nightly training, so a new generation ships only if it actually beats the incumbent and strength never regresses · Honest Elo: Stockfish is the referee and never a player, throttled to known Elo bands to score centipawn loss, top-1 agreement, and a calibrated rating (Bradley-Terry MLE) that becomes the engine's ceiling · A serving layer that scales strength to any target up to that ceiling and, above it, plays full strength and reports the ceiling rather than echoing a number it cannot play

**Stack:** Python, PPO/DQN, MCTS, AlphaZero-lite policy/value network, Stockfish-anchored evaluation

[GitHub →](https://github.com/aravinds-kannappan/Chess-Gambit-RL) · [Live demo →](https://chess-gambit-rl.vercel.app)
