---
title: "PodBench: Agent Evaluation at Fleet Scale"
excerpt: "Deterministic, resettable task environments for LLM agents with a programmatic verifier, per-run token/cost metering, and Kubernetes autoscaling — pod health and model behavior on one pane."
collection: portfolio
date: 2026-06-15
---

PodBench is the smallest system that answers three production questions at once: is the agent getting better or worse run-over-run on the same task, what is one passing run costing in tokens and dollars, and does the harness fall over when hundreds of agents run concurrently. It does this with one agent loop that runs in two places — a dashboard and a Kubernetes worker — against the same deterministic environment, scored by the same verifier, metered the same way.

**Key work:** A three-function environment contract (seed → tools → verify) over a seeded, pinned-clock SQL operations database that resets in microseconds · A shared runner imported by both the serverless dashboard and the queue worker, eliminating eval-versus-production drift · Token metering split into uncached / cache-write / cache-read / output buckets with prompt-prefix caching engineered to clear the 4096-token floor · Rate-limit backoff surfaced as a first-class metric · KEDA queue-depth autoscaling (2–24 replicas) with an HPA fallback · A benchmarking tab that draws the cost-versus-reward efficiency frontier and recommends the cheapest model clearing a reward bar

**Stack:** TypeScript, Next.js, Kubernetes, KEDA, Redis, alasql, Anthropic API

[GitHub →](https://github.com/aravinds-kannappan/PodBench) · [Live demo →](https://pod-bench.vercel.app)
