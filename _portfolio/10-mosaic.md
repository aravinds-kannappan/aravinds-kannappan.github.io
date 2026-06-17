---
title: "MOSAIC: Calibrated Multi-Stream Outbreak Early Warning"
excerpt: "Fuses wastewater, genomic, and outbreak-text surveillance into one calibrated probability that transmission is growing right now, and proves the number is trustworthy."
collection: portfolio
date: 2026-06-05
---

MOSAIC fuses three asynchronous, differently-noised public surveillance streams (CDC wastewater, Nextstrain genomic lineages, and WHO/ProMED outbreak text) into a single unitless quantity: `P(Rt > 1)`, the posterior probability that the effective reproduction number exceeds one. Then it does the thing most dashboards skip: it validates, on the real multi-year CDC NWSS record, that the probability it reports is calibrated.

**Key work:** Per-stream detectors, namely Bayesian Online Change-Point Detection on wastewater and text counts, Jensen-Shannon divergence on lineage frequencies, and an EpiEstim renewal posterior for Rt · Logistic / noisy-or fusion in a serverless lite tier that faithfully reproduces a Python NumPyro + NUTS hierarchical backend · Calibration validated on 1,334 day-ahead forecasts: ECE 0.086, AUROC 0.917, median early-warning lead ~68 days before a wave peak · 39 sewershed sites across 19 countries with real US data clearly separated from modeled demo sites · Claude assistant with tool use that both explains and drives the console

**Stack:** TypeScript, Next.js, Python, NumPyro/NUTS, Bayesian inference, Recharts

[GitHub →](https://github.com/aravinds-kannappan/MOSAIC) · [Live demo →](https://mosaic-surveillance.vercel.app/)
