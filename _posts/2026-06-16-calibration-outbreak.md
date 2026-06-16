---
title: "Can You Trust a Probability That Was Never Checked Against Reality?"
date: 2026-06-16
permalink: /posts/2026/06/calibration-outbreak/
tags:
  - bayesian
  - calibration
  - open source
---

Most risk dashboards report a number and ask you to trust it. A model says "73% chance of an outbreak" and you have no way to know whether, across all the times it said 73%, an outbreak followed 73% of the time or 30% of the time. A probability that has never been checked against what actually happened is not a probability — it is a vibe with a decimal point. I built [MOSAIC](https://github.com/aravinds-kannappan/MOSAIC) partly to take that problem seriously.

---

## One number with a falsifiable meaning

MOSAIC expresses everything as `P(Rt > 1)` — the posterior probability that the effective reproduction number exceeds one, i.e. that transmission is growing right now.

- **0%** means strong evidence transmission is shrinking.
- **50%** means genuinely uncertain whether it's growing or shrinking.
- **100%** means strong evidence it's growing.

The point of pinning the quantity down this precisely is that it becomes **falsifiable**. You can wait, observe whether transmission actually grew, and score the forecast. That is what makes calibration possible at all; a vague "risk score" can never be wrong, and therefore can never be trusted.

---

## What calibration actually means

A forecaster is calibrated if, among all the days it assigned probability *p*, the event happened a *p* fraction of the time. You measure the gap with **Expected Calibration Error** — bin the predictions, compare predicted probability to observed frequency in each bin, average the differences. Below 0.10 is well-calibrated. Two other numbers round out the picture: the **Brier score** (mean squared error of the probabilistic forecast, a strictly proper scoring rule) and **AUROC** (how well the number ranks growth days above non-growth days, 0.5 being chance).

MOSAIC validates on the real multi-year CDC wastewater record (Dec 2021 – Sep 2025). At each day it computes `P(Rt > 1)` from data available *up to that day*, then labels the outcome by whether activity actually rose over the next 14 days:

| Metric | Value | Interpretation |
|---|---|---|
| ECE | 0.086 | below 0.10 is well-calibrated |
| Brier | 0.124 | proper scoring rule |
| AUROC | 0.917 | strong discrimination |
| N | 1,334 | day-ahead forecasts |

The median early-warning lead is roughly 68 days before a wave peak. When MOSAIC says 75%, transmission subsequently grows about that often — and the reliability curve, not a marketing claim, is what proves it.

---

## Fusing three streams that disagree

The hard part isn't any single signal; it's reading three asynchronous, differently-scaled, differently-noised feeds at once. MOSAIC runs a per-stream detector on each:

- **Wastewater** — Bayesian Online Change-Point Detection (Poisson-Gamma predictive) plus a sustained-elevation term, combined as a noisy-or.
- **Genomic** — the Jensen-Shannon divergence of the recent lineage distribution against a baseline, with the alarm as an empirical tail probability.
- **Outbreak text** — change-point detection on daily WHO/ProMED report counts, weighted by recency.

Underneath sits a renewal-equation model: expected incidence is `E[I_t | R_t] = R_t * sum_s w_s * I(t-s)` with a discretized serial interval, and `log R_t` follows a random walk. Rt comes from EpiEstim's conjugate Gamma posterior, so `P(Rt > 1) = 1 - F_Gamma(1)` is a closed-form read.

---

## The honest part: a faithful approximation

MOSAIC runs in two tiers that share the same observation models and the same target. The **lite tier** runs the entire pipeline — change-point detection, divergence scoring, Rt estimation, fusion, calibration — in TypeScript on serverless infrastructure, so the public result is always live with no backend to operate. The **full tier** fits the joint hierarchical posterior with NumPyro and NUTS. The lite tier uses a learned-logistic / noisy-or fusion rather than the full sampler, and the claim I'm willing to defend is narrow: it reproduces the backend's numbers on the wastewater record, but the backend remains the reference for the multi-stream joint posterior.

The rest of the honesty is in the labels. The US wastewater data and the WHO/ProMED news are real and live; the international sites and non-COVID pathogen panels are modeled for the demo and flagged as such in the UI. And calibration is retrospective — it comes from back-testing, and live calibration drifts as pathogens and reporting change, which is exactly why the dashboard exposes the reliability diagram instead of asserting calibration once and forgetting it.

---

## Why this generalizes

`P(Rt > 1)` answers whether transmission is growing, not how bad an outbreak will be — it's an early-warning trigger for human judgment, not a forecast of cases or deaths. But the discipline transfers to any high-stakes probability: define a falsifiable quantity, back-test it against what actually happened, and ship the reliability diagram alongside the number. A probability you can't check is one you shouldn't trust, including your own.

---

*This post discusses epidemic surveillance modeling. MOSAIC is a defensive system built on aggregate, de-identified public data and is meant to augment, not replace, public-health judgment.*
