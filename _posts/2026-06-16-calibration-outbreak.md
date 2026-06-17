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

MOSAIC expresses everything as a single quantity: $P(R_t > 1)$, the posterior probability that the effective reproduction number exceeds one. The effective reproduction number, $R_t$, is the average number of new infections caused by each current infection at a given moment. When it is above one, each case more than replaces itself and the epidemic curve turns upward. When it is below one, the outbreak shrinks. So $P(R_t > 1)$ is, in plain terms, the probability that transmission is growing right now, for a specific pathogen in a specific place.

Three anchor points make the scale concrete. A reading near 0 percent means strong evidence that transmission is shrinking. A reading near 50 percent means genuine uncertainty, the data cannot yet say whether things are growing or shrinking. A reading near 100 percent means strong evidence that transmission is growing. The quantity is deliberately unitless and comparable across pathogens and places, so a 70 percent for influenza in one city means the same kind of thing as a 70 percent for a novel pathogen in another.

The reason for pinning the quantity down this precisely is not pedantry. It is that a precise quantity is falsifiable, and a falsifiable quantity can be checked. You can wait, observe whether transmission actually grew, and score the forecast against what happened. A vague "elevated risk" label can never be wrong, because it never committed to anything, and a claim that can never be wrong can never earn trust. The whole project rests on choosing a number that can be held to account.

---

## Calibration, written down

The intuition has a precise form. Let $f(X) \in [0,1]$ be the forecast and $Y \in \{0,1\}$ the outcome, here whether transmission grew over the next 14 days. The forecaster is *perfectly calibrated* if

$$\Pr\big(Y = 1 \mid f(X) = p\big) = p \qquad \text{for all } p \in [0,1].$$

Reading that aloud: among all the occasions the system said $p$, the event happened a $p$ fraction of the time. The Expected Calibration Error is the average distance from that ideal, weighted by where the forecasts actually land,

$$\mathrm{ECE} = \mathbb{E}_{p \sim f(X)}\Big[\, \big|\, \Pr(Y = 1 \mid f(X) = p) - p \,\big| \,\Big],$$

estimated in practice by binning the forecasts and comparing each bin's mean prediction to its observed frequency. The reliability diagram is just the plot of $\Pr(Y = 1 \mid f = p)$ against $p$; perfect calibration is the diagonal, and a curve sagging below it means the forecaster is overconfident.

Calibration is not the same as usefulness, and the Brier score makes the separation exact. Writing $\mathrm{BS} = \mathbb{E}[(f(X) - Y)^2]$, Murphy's decomposition splits it into three interpretable terms,

$$\mathrm{BS} = \underbrace{\mathbb{E}\big[(f - \mathbb{E}[Y \mid f])^2\big]}_{\text{reliability, lower better}} \; - \; \underbrace{\mathbb{E}\big[(\mathbb{E}[Y \mid f] - \bar{y})^2\big]}_{\text{resolution, higher better}} \; + \; \underbrace{\bar{y}(1 - \bar{y})}_{\text{uncertainty}},$$

where $\bar{y} = \mathbb{E}[Y]$ is the base rate. Reliability is the calibration term. Resolution rewards forecasts that actually separate high-risk days from low-risk ones. Uncertainty is the irreducible variance of the outcome, which no forecaster can reduce. The Brier score is also a *strictly proper* scoring rule, meaning that for a true outcome probability $q$ the expected score $\mathbb{E}_{Y \sim q}[(p - Y)^2]$ is uniquely minimized at $p = q$, so you cannot improve your score by shading your reported probability away from your honest belief. AUROC captures the other virtue, discrimination, and has a clean probabilistic reading,

$$\mathrm{AUROC} = \Pr\big(f(X^+) > f(X^-)\big),$$

the chance that a randomly chosen growth day is scored above a randomly chosen non-growth day. A forecaster can be well-calibrated but poor at ranking, or a sharp ranker that is systematically overconfident, and a good system needs both.

---

## The validation setup, and the numbers

MOSAIC validates on the real multi-year CDC wastewater record, from December 2021 to September 2025. The procedure is the part that makes the numbers honest. For each day in the record, the system computes $P(R_t > 1)$ using only data available up to that day, with no peeking at the future. Then it labels the outcome by what actually happened next: did wastewater activity rise over the following 14 days. This is a causal, out-of-sample, day-ahead evaluation, which is the only kind worth trusting, because it mimics the real situation of standing in the present and forecasting the unseen future rather than fitting a curve to data you already have.

| Metric | Value | Interpretation |
|---|---|---|
| ECE | 0.086 | below 0.10, so well-calibrated |
| Brier | 0.124 | proper scoring rule, lower is better |
| AUROC | 0.917 | strong discrimination |
| N | 1,334 | day-ahead forecasts |

Across 1,334 day-ahead forecasts, the ECE of 0.086 sits below the well-calibrated threshold, the AUROC of 0.917 shows the number separates growth days from non-growth days strongly, and the Brier score confirms the forecasts are sharp as well as honest. The median early-warning lead is roughly 68 days before a wave peak, which is the practical payoff: the signal turns upward more than two months before the peak that a clinical case count would eventually reveal. When MOSAIC says 75 percent, transmission subsequently grows about 75 percent of the time, and it is the reliability curve, not a marketing claim, that demonstrates it.

---

## Fusing three streams that disagree

The hard part of all this is not estimating $R_t$ from one clean signal. It is that the early signals of an epidemic arrive in three very different forms, on different schedules, at different scales, with different kinds of noise, and reading them together is where the judgment lives. MOSAIC fuses three asynchronous public streams, and it runs a dedicated detector on each before combining them.

The **wastewater** stream is the backbone. Viral concentration in sewage rises before clinical cases, because people shed virus before they feel sick enough to seek care. The detector here is Bayesian Online Change-Point Detection, or BOCPD, which maintains a posterior over the current run length $r_t$, the time since the last change point, through the recursion

$$\Pr(r_t,\, x_{1:t}) = \sum_{r_{t-1}} \Pr(r_t \mid r_{t-1})\; \Pr\big(x_t \mid r_{t-1}, x^{(r)}_{t-1}\big)\; \Pr(r_{t-1},\, x_{1:t-1}),$$

and raises an alarm from the posterior mass on $r_t = 0$, the probability that a change is happening right now. That change-point probability is combined with a sustained-elevation term so a signal that is both changing and staying high counts for more than either alone.

The **genomic** stream watches the mix of circulating lineages. The detector computes the Jensen-Shannon divergence between the recent lineage distribution $P$ and a baseline $Q$,

$$\mathrm{JSD}(P \,\|\, Q) = \tfrac{1}{2}\,\mathrm{KL}(P \,\|\, M) + \tfrac{1}{2}\,\mathrm{KL}(Q \,\|\, M), \qquad M = \tfrac{1}{2}(P + Q),$$

a symmetric and bounded measure, $0 \le \mathrm{JSD} \le \ln 2$, of how far the lineage mix has moved. A new variant sweeping through shows up as a sudden divergence, and the alarm is the empirical tail probability of that divergence against its own history, so the system reacts to shifts that are large relative to normal week-to-week churn.

The **outbreak-text** stream mines official reports and news, WHO Disease Outbreak News and ProMED, for early qualitative signals that clinicians post before the structured data catches up. The detector runs change-point detection on the daily count of relevant reports, weighted by recency, treating a sudden rise in chatter as a quantitative third signal rather than as anecdote.

---

## The model underneath

Under all three detectors sits a generative model of how infections actually propagate. The renewal equation says expected incidence at time $t$, given the reproduction number, is

$$\mathbb{E}[I_t \mid R_t] = R_t \sum_{s \ge 1} w_s\, I_{t-s},$$

where $\{w_s\}$ is the discretized serial-interval distribution, the probability that a case generated $s$ days ago infects someone today. A smoothness prior lets $R_t$ drift rather than jump, modeled as a log-random walk,

$$\log R_t = \log R_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma^2),$$

so the estimate is smooth but free to move when the data demand it. Under a Poisson incidence likelihood and a conjugate Gamma prior $R_t \sim \mathrm{Gamma}(a, b)$, the posterior over a trailing window $\tau$ stays Gamma,

$$R_t \mid I \;\sim\; \mathrm{Gamma}\!\Big(a + \sum_{u \in \tau} I_u,\;\; b + \sum_{u \in \tau} \Lambda_u\Big), \qquad \Lambda_u = \sum_{s \ge 1} w_s\, I_{u-s},$$

which is the EpiEstim construction. The headline number is then a single cumulative-distribution evaluation against that posterior,

$$P(R_t > 1) = 1 - F_{\mathrm{Gamma}}(1),$$

read straight off the posterior rather than guessed. When several streams carry signal, their individual alarms $p_1, \dots, p_k$ are combined by a noisy-or,

$$P(\text{growth}) = 1 - \prod_{i=1}^{k} (1 - p_i),$$

which saturates toward one as independent streams agree and degrades gracefully to a single stream's alarm when the others are silent, so a text-only outbreak is not diluted by absent wastewater or genomic data.

---

## The honest part: a faithful approximation

MOSAIC runs in two tiers that share the same observation models and the same target quantity, and being clear about the difference between them is part of being trustworthy. The lite tier runs the entire pipeline, change-point detection, divergence scoring, $R_t$ estimation, fusion, and calibration, in TypeScript on serverless infrastructure. This is what the public site uses, and it is what makes the result always live with no backend to operate and no server to fall over. The full tier is optional and fits the joint hierarchical posterior with NumPyro and the No-U-Turn Sampler, a gold-standard Markov chain Monte Carlo method.

The claim I am willing to defend about the lite tier is narrow and specific. Its fusion is a learned-logistic and noisy-or approximation rather than the full sampler, and it reproduces the backend's numbers on the wastewater record, but the full Bayesian backend remains the reference for the multi-stream joint posterior. I say this plainly because the alternative, presenting an approximation as if it were the full model, is exactly the kind of quiet overclaim that calibration is supposed to guard against. The noisy-or fallback is used for cells where only one stream has signal, so a text-only outbreak is not diluted by absent wastewater and genomic data, which is a deliberate choice with a known cost.

The same honesty extends to the data. The United States wastewater data and the WHO and ProMED news are real and live. The international sites and the non-COVID pathogen panels are modeled for the demo, because per-site multi-pathogen wastewater simply is not uniformly available as open data, and every modeled element is flagged as such in the interface rather than dressed up as real. And the calibration itself is retrospective: the ECE and AUROC come from back-testing on the historical record, and live calibration drifts as pathogens, sampling, and reporting practices change. That is exactly why the dashboard exposes the reliability diagram as a live, recomputed object instead of asserting calibration once and asking you to take it on faith forever.

---

## Growth is not severity

One limitation is important enough to state on its own, because it is the kind of thing a confident number invites you to forget. $P(R_t > 1)$ answers whether transmission is growing. It does not answer how bad the outbreak will be. A pathogen can have an $R_t$ comfortably above one and cause mild disease, or sit near one and be devastating. The probability is an early-warning trigger for human judgment, a prompt to look closer and prepare, not a forecast of cases or deaths and not a substitute for the public-health expertise that decides what to do about a rising signal. The system is built to augment that judgment, not to replace it, and the emphasis on calibration and uncertainty is meant to discourage overreaction to weak signals as much as to flag strong ones.

---

## Why this generalizes

Strip away the epidemiology and the lesson is a general one about any high-stakes probability, including the ones you produce yourself. Choose a quantity precise enough to be falsifiable, so it can be checked against reality rather than admired. Back-test it honestly, computing each forecast from only the information available at the time and scoring it with a proper rule against what actually happened next. Then ship the reliability diagram alongside the number, so anyone can see where the forecaster is calibrated and where it is not, rather than asking them to trust a point estimate. A probability you cannot check is one you should not trust, and that applies with full force to your own confident numbers, not just to someone else's dashboard.

---

*This post discusses epidemic surveillance modeling. MOSAIC is a defensive system built on aggregate, de-identified public data, and it is meant to augment, not replace, public-health judgment.*
