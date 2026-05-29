---
title: "What Two Years of Production ML Taught Me About Software Engineering (and Vice Versa)"
date: 2025-12-01
permalink: /posts/2025/12/swe-lessons-for-ml/
tags:
  - software engineering
  - machine learning
  - reflection
---

I came into ML from a statistics background, not a CS background. For the first year of building production systems, I treated software engineering as the necessary boilerplate around the interesting ML work — something to get through so I could get back to the model. That was wrong, and expensive. This post is a collection of the most important conceptual shifts that came from building real systems.

## ML Code Is Software, and Software Has Engineering Principles

The first thing that surprised me was how applicable standard SWE principles are to ML code — not the algorithms, but the way you structure the *codebase* around them.

**Separation of concerns** matters enormously. A training script that mixes data loading, model definition, loss computation, evaluation, and logging in a single 500-line function is a maintainability disaster. When you change the loss function, you have to read all 500 lines to understand what else might break. When you want to swap the data loader, there's no clean interface to target. The standard ML research pattern (monolithic notebooks) works fine for exploration; it's a liability in production.

The pattern that works: separate data contracts (what schema does each stage expect?), model definitions (stateless, testable), training loops (compose the above), and evaluation pipelines (reproducible, versioned). These are just interfaces and dependency injection — standard SWE. The insight is that ML systems have the same modularity problems as any other software system, with the additional complication that many bugs are silent (the code runs, but the model learns the wrong thing).

**Testing is harder in ML but still essential.** Unit tests for data pipelines — does the preprocessing function handle missing values correctly? Does the feature engineering produce the expected output on a known input? — are fully tractable and often neglected. What's harder is testing the model itself: you can't unit test "does this model generalize?" But you can test invariants: a model trained on augmented data should produce similar outputs for semantically equivalent inputs. A fraud classifier shouldn't flip its prediction when you change the customer's name. These are property-based tests, and they catch real bugs.

## Latency Is a Product Requirement, Not an Optimization Detail

The hardest mental shift was treating latency as a first-class constraint from the beginning, not something to optimize after the system is built.

At Synthure, we initially built our clinical LLM pipeline with correctness as the only criterion. We got to 94% reduction in policy misinterpretation errors — a genuinely good result — but the p95 latency was 8 seconds. That's fine for a batch process; it's a product failure for a real-time clinical tool where a physician is waiting for a coding recommendation mid-patient-encounter. We had to rebuild significant portions of the architecture around a latency budget, which would have been much cheaper to design for from the start.

The principle: **define your latency SLO before you design your system.** Then derive your architecture from that constraint. If you need sub-2s p95 for a user-facing feature, and your LLM inference alone takes 1.5s at median, you have no budget for retrieval, preprocessing, or postprocessing unless they run in parallel. This forces you to think about the entire request path as a pipeline with a budget, not a collection of independent components.

The tools that made this concrete: p95 latency (not mean — tails determine user experience), waterfall profiling of each stage in the request path, and load testing under realistic traffic patterns before launch. A system that works at 1 RPS often breaks at 10 RPS in ways that aren't visible in development.

## Reproducibility Is Infrastructure, Not Discipline

In research, reproducibility is a matter of personal discipline — carefully documenting your random seeds, hyperparameters, and data splits. In production, this doesn't scale. The fifth time a colleague asks "which version of the model is running in prod?" and the answer requires digging through Slack history, you realize reproducibility needs to be an engineering primitive, not a cultural norm.

The solution is to make the mapping from artifact to provenance automatic:
- Every model artifact is tagged with its training run ID, which points to the exact config, data version, and code commit used to produce it
- Data pipelines are versioned and deterministic — same input, same output, always
- Evaluation results are stored alongside model artifacts, not in a separate spreadsheet

None of this is technically hard. It requires discipline to set up early, before you have 30 model versions and no way to know which one produced which evaluation number. The frameworks (MLflow, W&B, DVC) exist; the skill is treating them as mandatory infrastructure, not optional tooling.

## The Abstraction Gradient

One of the most useful mental models I've developed is the **abstraction gradient** — the idea that different parts of an ML system need to operate at different levels of abstraction, and confusing those levels creates subtle bugs.

Raw data lives at the lowest level of abstraction — actual bytes, actual schema. Features live at a higher level — derived quantities computed from raw data. Model inputs live at a higher level still — tensors of a specific shape and dtype. Predictions live above that — numbers with units and semantic meaning. Decisions (the downstream action taken on a prediction) live at the highest level.

Bugs often happen when abstractions collapse: a feature is accidentally computed from the model's own output (target leakage), or a prediction is used directly as a decision without calibration, or a raw feature is passed to a model that expects normalized inputs. The fix is to be explicit about which layer each function operates on, and to have tests at each layer boundary that verify the contract.

## What ML Taught Me About SWE

The reverse transfer is equally real. ML systems have a property that most software systems don't: **uncertainty is load-bearing.** A traditional software function either returns the right answer or it doesn't. An ML model always returns an answer; the question is how confident you should be in it.

This changes how you think about interfaces. A function that returns a prediction should, in most production contexts, also return a confidence score. An API that surfaces model outputs should expose calibration metadata. A monitoring system for an ML pipeline needs to track distribution metrics (PSI, KL-divergence between training and serving distributions), not just uptime and error rates.

The other insight from ML that applies broadly in SWE: **measure before you optimize.** ML practitioners are trained to profile models — measure inference time, memory footprint, and FLOPs before deciding what to optimize. This discipline doesn't always transfer to the surrounding software, where it's common to optimize a function that accounts for 2% of wall-clock time because it feels slow. Profile first.

## The Honest Assessment

The most important thing I've learned is that production ML is a collaborative discipline that requires both statistical depth and engineering rigor, and that neither substitutes for the other. A model with state-of-the-art accuracy that takes 30 seconds to serve, can't be monitored in production, and fails silently on distribution shift is worse than a simpler model that's fast, observable, and gracefully degraded. The engineering work isn't boilerplate around the ML — it *is* the ML, once you're building for real users.
