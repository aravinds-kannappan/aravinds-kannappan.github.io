---
title: "GameNGen: Diffusion World Model for Game Simulation"
excerpt: "Trained a latent diffusion world model on 737K+ Super Mario Bros frames with VAE encoder, CNN reward model, and PPO agents."
collection: portfolio
date: 2026-01-01
---

Trained a world model to simulate Super Mario Bros gameplay end-to-end using latent diffusion. The system learns a compressed representation of game state via a VAE, generates plausible next frames through a diffusion model, and trains PPO agents entirely within the simulated environment — no real game engine needed at inference time.

**Key components:** VAE encoder/decoder for latent state compression · Latent diffusion model for next-frame generation · CNN-based reward model estimating reward signal from visual observations · PPO agents trained within the world model loop · Adversarial initialization for diverse rollout coverage

**Stack:** Python, PyTorch, diffusion models, VAE, PPO, computer vision

[GitHub →](https://github.com/aravinds-kannappan/gamengen-wm)
