---
title: "VeriGrad RL: Mechanistic Interpretability for Safety RL"
excerpt: "An open-source mechanistic interpretability and AI-safety lab for RL post-training — train policies to choose activation-level interventions, then verify they're behaviorally safe, useful, and mechanistically faithful."
collection: portfolio
date: 2026-06-10
---

Safety-oriented RL systems can look good on scalar reward while failing for deeper reasons: reward hacking, train/test leakage, blanket refusal, brittle jailbreak behavior, or interventions that are behaviorally safe but mechanistically unfaithful. VeriGrad RL keeps every one of those pieces explicit. It trains a policy to choose activation-level interventions on a transparent synthetic residual-stream circuit, then scores safety, utility retention, mechanistic targeting, sparsity, and off-target activation damage.

**Key work:** A toy residual stream with named features (harmful intent, helpful intent, jailbreak pressure, refusal prior, uncertainty) where activation patching is the action space · A structured mechanistic verifier whose every reward component is inspectable · REINFORCE training with a moving baseline, plus train and out-of-distribution eval splits reporting safety, utility, over-refusal, and jailbreak-success rates · A defensive, non-operational biosafety / DNA-order-screening triage playground built on synthetic risk features · Reproducible JSONL logs, JSON checkpoints, and reward-hacking probes

**Stack:** Python, REINFORCE, mechanistic interpretability, activation patching

[GitHub →](https://github.com/aravinds-kannappan/VeriGrad-RL) · [Live demo →](https://verigrad-biosafety.vercel.app/)
