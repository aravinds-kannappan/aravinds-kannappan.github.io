---
title: "Market Microstructure Forecasting with Deep RL"
excerpt: "DRL agent predicting short-horizon price movements from order book snapshots and executing trades via event-driven backtesting."
collection: portfolio
date: 2026-03-01
---

An end-to-end system for predicting short-horizon price movements from order book snapshots and using those predictions to inform a deep reinforcement learning agent that executes trades. Performance is evaluated via event-driven backtesting against historical market data.

**Key components:** Order book feature engineering from L2 snapshot data · Supervised forecasting model for short-horizon price direction · DRL agent (PPO/DQN) trained on predicted signals · Event-driven backtesting engine with realistic transaction costs and slippage

**Stack:** Python, PyTorch, reinforcement learning, quantitative finance, backtesting

[GitHub →](https://github.com/aravinds-kannappan/Market-Microstructure-Forecasting-with-Deep-Reinforcement-Learning)
