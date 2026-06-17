---
title: "RallyScope: Tennis ML & Computer Vision in the Browser"
excerpt: "36,342 ATP/WTA matches turned into an unsupervised playstyle map, an exactly-explainable win-probability model, and an in-browser ball tracker, trained at build time and inferred client-side."
collection: portfolio
date: 2026-06-14
---

RallyScope turns 36,342 real ATP & WTA matches (2018 to 2024) into three interactive modules: an unsupervised playstyle embedding that arranges every player on a 2D style map, an explainable win-probability model that shows *why* it favors a player, and a computer-vision pipeline that tracks a tennis ball frame by frame, all running entirely in the browser. The core idea is to train where it's cheap (a build-time Python pass) and infer where it's free (the client).

**Key work:** PCA playstyle embedding implemented from scratch with power iteration over a hand-rolled covariance matrix · Logistic win-probability model (SGD + L2) with an *exact* per-feature logit decomposition instead of approximate SHAP · Forward-in-time train/test split (2018 to 2023 train, 2024 holdout) reaching ~0.71 ROC-AUC with near-diagonal calibration · In-browser ball tracker using optic-yellow chroma and inter-frame motion segmentation, court homography for speed, validated against a synthetic ground-truth trajectory · Pure-stdlib build pipeline (no NumPy/pandas/sklearn) baking a single static JSON the browser re-implements

**Stack:** Next.js, TypeScript, Python (standard library), PCA, logistic regression, canvas computer vision

[GitHub →](https://github.com/aravinds-kannappan/Rally-Scope) · [Live demo →](https://rally-scope-vvay.vercel.app/)
