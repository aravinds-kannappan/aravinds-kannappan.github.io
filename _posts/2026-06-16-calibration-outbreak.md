---
title: "Can You Trust a Probability That Was Never Checked Against Reality?"
date: 2026-06-16
permalink: /posts/2026/06/calibration-outbreak/
tags:
  - bayesian
  - calibration
  - open source
---

Most risk dashboards report a number and ask you to trust it. A model says there is a 73 percent chance of an outbreak, and you have no way to know whether, across all the times it has said 73 percent, an outbreak actually followed 73 percent of the time, or 30 percent, or 95 percent. The number looks authoritative because it has a decimal point, but a probability that has never been checked against what actually happened is not really a probability. It is a feeling with a unit attached. I built [MOSAIC](https://github.com/aravinds-kannappan/MOSAIC) in large part to take that problem seriously, and the discipline it forces, define a falsifiable quantity and then prove it is calibrated, turns out to matter far beyond epidemics.

---

## One number with a falsifiable meaning

MOSAIC expresses everything as a single quantity: `P(Rt > 1)`, the posterior probability that the effective reproduction number exceeds one. The effective reproduction number, Rt, is the average number of new infections caused by each current infection at a given moment. When it is above one, each case more than replaces itself and the epidemic curve turns upward. When it is below one, the outbreak shrinks. So `P(Rt > 1)` is, in plain terms, the probability that transmission is growing right now, for a specific pathogen in a specific place.

Three anchor points make the scale concrete. A reading near 0 percent means strong evidence that transmission is shrinking. A reading near 50 percent means genuine uncertainty, the data cannot yet say whether things are growing or shrinking. A reading near 100 percent means strong evidence that transmission is growing. The quantity is deliberately unitless and comparable across pathogens and places, so a 70 percent for influenza in one city means the same kind of thing as a 70 percent for a novel pathogen in another.

The reason for pinning the quantity down this precisely is not pedantry. It is that a precise quantity is falsifiable, and a falsifiable quantity can be checked. You can wait, observe whether transmission actually grew, and score the forecast against what happened. A vague "elevated risk" label can never be wrong, because it never committed to anything, and a claim that can never be wrong can never earn trust. The whole project rests on choosing a number that can be held to account.

---

## What calibration actually means

Calibration is the property that makes a probability trustworthy, and it has a precise definition. A forecaster is calibrated if, among all the occasions on which it assigned probability `p` to an event, the event actually happened a `p` fraction of the time. If you gather every day the system said 70 percent and transmission subsequently grew on about 70 percent of those days, the system is calibrated at 70 percent. If it grew on only 40 percent of them, the system is overconfident, and its 70 percent does not mean what it says.

The standard way to summarize this is the **Expected Calibration Error**, or ECE. You bin the predictions, say all the days the system predicted between 60 and 70 percent, compare the average predicted probability in each bin to the observed frequency of the event in that bin, and average the gaps across bins, weighted by how many predictions fall in each. An ECE below 0.10 is generally considered well-calibrated, meaning the predicted probabilities and the observed frequencies track each other closely. The visual version of the same idea is the reliability diagram, which plots predicted probability against observed frequency; a perfectly calibrated forecaster hugs the diagonal, and deviations above or below the line show where it is under or overconfident.

Two more numbers round out the picture. The **Brier score** is the mean squared error of the probabilistic forecast, a strictly proper scoring rule, which is a technical way of saying it is minimized only by reporting your true beliefs, so you cannot game it by shading your numbers toward the extremes. Lower is better. The **AUROC**, the area under the receiver operating characteristic curve, measures discrimination: how well the number ranks growth days above non-growth days. An AUROC of 0.5 is no better than a coin flip, and 1.0 is perfect separation. Calibration and discrimination are different virtues, a forecaster can be well-calibrated but useless at ranking, or a sharp ranker that is systematically overconfident, and a good system needs both.

---

## The validation setup, and the numbers

MOSAIC validates on the real multi-year CDC wastewater record, from December 2021 to September 2025. The procedure is the part that makes the numbers honest. For each day in the record, the system computes `P(Rt > 1)` using only data available up to that day, with no peeking at the future. Then it labels the outcome by what actually happened next: did wastewater activity rise over the following 14 days. This is a causal, out-of-sample, day-ahead evaluation, which is the only kind worth trusting, because it mimics the real situation of standing in the present and forecasting the unseen future rather than fitting a curve to data you already have.

| Metric | Value | Interpretation |
|---|---|---|
| ECE | 0.086 | below 0.10, so well-calibrated |
| Brier | 0.124 | proper scoring rule, lower is better |
| AUROC | 0.917 | strong discrimination |
| N | 1,334 | day-ahead forecasts |

Across 1,334 day-ahead forecasts, the ECE of 0.086 sits below the well-calibrated threshold, the AUROC of 0.917 shows the number separates growth days from non-growth days strongly, and the Brier score confirms the forecasts are sharp as well as honest. The median early-warning lead is roughly 68 days before a wave peak, which is the practical payoff: the signal turns upward more than two months before the peak that a clinical case count would eventually reveal. When MOSAIC says 75 percent, transmission subsequently grows about 75 percent of the time, and it is the reliability curve, not a marketing claim, that demonstrates it.

---

## Fusing three streams that disagree

The hard part of all this is not estimating Rt from one clean signal. It is that the early signals of an epidemic arrive in three very different forms, on different schedules, at different scales, with different kinds of noise, and reading them together is where the judgment lives. MOSAIC fuses three asynchronous public streams, and it runs a dedicated detector on each before combining them.

The **wastewater** stream is the backbone. Viral concentration in sewage rises before clinical cases, because people shed virus before they feel sick enough to seek care. The detector here is Bayesian Online Change-Point Detection, or BOCPD, which maintains a running posterior over how long the current statistical regime has lasted and raises an alarm when a change point becomes likely. That change-point probability is combined with a sustained-elevation term through a noisy-or, so a signal that is both changing and staying high counts for more than either alone.

The **genomic** stream watches the mix of circulating lineages. The detector computes the Jensen-Shannon divergence, a symmetric and bounded measure of how different two probability distributions are, between the recent lineage distribution and a baseline. A new variant sweeping through the population shows up as a sudden divergence, and the alarm is the empirical tail probability of that divergence against its own history, so the system reacts to shifts that are large relative to normal week-to-week churn.

The **outbreak-text** stream mines official reports and news, WHO Disease Outbreak News and ProMED, for early qualitative signals that clinicians and public-health officers post before the structured data catches up. The detector runs change-point detection on the daily count of relevant reports, weighted by recency, treating a sudden rise in chatter as a quantitative third signal rather than as anecdote.

Under all three sits a generative model of how infections actually propagate. The renewal equation says expected incidence at time `t`, given the reproduction number, is `R_t` times a weighted sum of past incidence, where the weights come from a discretized serial interval, the distribution of times between one case and the cases it causes. The log of `R_t` is modeled as a random walk, so the estimate is smooth but free to move. Rt itself comes from EpiEstim's conjugate Gamma posterior, which gives a clean closed form: `P(Rt > 1) = 1 - F_Gamma(1)`, one minus the Gamma cumulative distribution evaluated at one. That is the headline number, read straight off a posterior rather than guessed.

---

## The honest part: a faithful approximation

MOSAIC runs in two tiers that share the same observation models and the same target quantity, and being clear about the difference between them is part of being trustworthy. The lite tier runs the entire pipeline, change-point detection, divergence scoring, Rt estimation, fusion, and calibration, in TypeScript on serverless infrastructure. This is what the public site uses, and it is what makes the result always live with no backend to operate and no server to fall over. The full tier is optional and fits the joint hierarchical posterior with NumPyro and the No-U-Turn Sampler, a gold-standard Markov chain Monte Carlo method.

The claim I am willing to defend about the lite tier is narrow and specific. Its fusion is a learned-logistic and noisy-or approximation rather than the full sampler, and it reproduces the backend's numbers on the wastewater record, but the full Bayesian backend remains the reference for the multi-stream joint posterior. I say this plainly because the alternative, presenting an approximation as if it were the full model, is exactly the kind of quiet overclaim that calibration is supposed to guard against. A noisy-or fallback is used for cells where only one stream has signal, so a text-only outbreak is not diluted by absent wastewater and genomic data, which is a deliberate choice with a known cost.

The same honesty extends to the data. The United States wastewater data and the WHO and ProMED news are real and live. The international sites and the non-COVID pathogen panels are modeled for the demo, because per-site multi-pathogen wastewater simply is not uniformly available as open data, and every modeled element is flagged as such in the interface rather than dressed up as real. And the calibration itself is retrospective: the ECE and AUROC come from back-testing on the historical record, and live calibration drifts as pathogens, sampling, and reporting practices change. That is exactly why the dashboard exposes the reliability diagram as a live, recomputed object instead of asserting calibration once and asking you to take it on faith forever.

---

## Growth is not severity

One limitation is important enough to state on its own, because it is the kind of thing a confident number invites you to forget. `P(Rt > 1)` answers whether transmission is growing. It does not answer how bad the outbreak will be. A pathogen can have an Rt comfortably above one and cause mild disease, or sit near one and be devastating. The probability is an early-warning trigger for human judgment, a prompt to look closer and prepare, not a forecast of cases or deaths and not a substitute for the public-health expertise that decides what to do about a rising signal. The system is built to augment that judgment, not to replace it, and the emphasis on calibration and uncertainty is meant to discourage overreaction to weak signals as much as to flag strong ones.

---

## Why this generalizes

Strip away the epidemiology and the lesson is a general one about any high-stakes probability, including the ones you produce yourself. Choose a quantity precise enough to be falsifiable, so it can be checked against reality rather than admired. Back-test it honestly, computing each forecast from only the information available at the time and scoring it against what actually happened next. Then ship the reliability diagram alongside the number, so anyone can see where the forecaster is calibrated and where it is not, rather than asking them to trust a point estimate. A probability you cannot check is one you should not trust, and that applies with full force to your own confident numbers, not just to someone else's dashboard.

---

*This post discusses epidemic surveillance modeling. MOSAIC is a defensive system built on aggregate, de-identified public data, and it is meant to augment, not replace, public-health judgment.*
