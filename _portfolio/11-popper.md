---
title: "Popper: Falsifying Specifications Before You Prove Them"
excerpt: "An open-source tool that hunts for counterexamples to mathematical and logical statements — catching wrong specifications, not just mismatched proofs."
collection: portfolio
date: 2026-06-13
---

Provers like Lean and Axiom's AXLE can certify, for certain, that a proof matches a statement. What they cannot tell you is whether the statement is the one you meant. Popper goes after the statement directly: it tries to break a spec by finding an input that makes it fail, and when it succeeds it hands you that concrete counterexample — which doubles as a repair signal. It runs *before* the expensive prover, because breaking is cheap and proving is not.

**Key work:** A common oracle interface over two engines that report FAITHFUL / FALSIFIED / UNSOUND / INCOMPLETE / VACUOUS / INCONCLUSIVE · A local Monte-Carlo math engine that samples thousands of cases to catch dropped assumptions and flipped inequalities (no API key, no prover) · A live code-spec engine that drives AXLE/Lean on real Verina tasks to detect too-tight, too-loose, and vacuous specs · A counterexample-driven repair loop that pushes broken specs back to faithful · A 346-statement benchmark where Popper catches 176/178 unfaithful specs (99%) with zero false alarms, while a proof checker catches none

**Stack:** Python (standard-library core), Lean / Axiom AXLE, Next.js, Claude agent

[GitHub →](https://github.com/aravinds-kannappan/Popper) · [Live demo →](https://popper-lovat.vercel.app/)
