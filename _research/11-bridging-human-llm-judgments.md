---
title: "Bridging the Gap: What LLM Judges See That Humans Don't"
excerpt: "A statistical framework that models an LLM judge as a shared human preference plus a linear bias on interpretable features, so you can both correct the judge and formally test where it diverges from people."
collection: research
order: 11
date: 2025-12-01
tags:
  - evaluation
  - LLM-as-judge
  - statistics
  - AI safety
---

So much of modern evaluation, including the safety training the models in this collection received, runs on LLM-as-a-judge. That makes the systematic ways LLM judges disagree with humans a first-class problem, not a footnote. "Bridging Human and LLM Judgments: Understanding and Narrowing the Gap" by Maia Polo, Wang, Yurochkin, Xu, Banerjee, and Sun (NeurIPS 2025) is a statistician's answer to it, and I appreciate that it treats the gap as something to *model and test* rather than just complain about. Most work on judge bias documents a specific failure and proposes a patch; this one builds the object that makes all those failures instances of a single, testable structure.

---

## The model

The framework, called `Bridge`, posits a single latent **human preference score** $Z^h$ for each prompt-response pair, and models the LLM's judgment as that same latent factor plus a **linear transformation of interpretable covariates** $X$ (response length, sentiment, creativity, formatting, structural attributes) that capture exactly where the LLM drifts from people. Both human and LLM judgments are fit with an **ordinal logistic regression**, which respects the fact that ratings have an order (a 4 is above a 3) and that pairwise comparisons ("A wins / tie / B wins") are themselves ordered. For a human rating with ordered cutoffs $\alpha_1 < \cdots < \alpha_K$,

$$\mathbb{P}(Y^h = k \mid I, O) = \sigma(\alpha_{k+1} - Z^h) - \sigma(\alpha_k - Z^h),$$

and the LLM's version replaces the latent score with $\beta Z^h + \gamma^\top X$. The vector $\gamma$ *is* the systematic bias: it says how much each interpretable feature shifts the LLM's judgment relative to a human's, holding the shared preference fixed.

The clever engineering piece is the **logit trick**, which fits the whole thing without ever observing the latent human scores. Because the LLM's rating probabilities are computable (either exactly from the log-probabilities of the outcome tokens, or estimated by sampling many chain-of-thought judgments), you can recover the latent LLM scores and cutoffs by matching model-implied probabilities to observed ones, then fit the human ordinal model on top. The name is apt: under binary judgments, the latent score is precisely the logit of the LLM's "A wins" probability. And because the estimators are shown to be **asymptotically normal**, you get honest tools: confidence intervals for each parameter, and a formal test of $H_0: \gamma_j = 0$ against $H_1: \gamma_j \neq 0$ with p-value

$$\text{p-value} = 2\Phi\!\left(-\sqrt{n/\hat{V}_{j+1,j+1}}\,|\hat\gamma_j|\right),$$

with Benjamini-Yekutieli correction to control the false discovery rate across the many features you test at once. Rejecting $H_0$ means feature $j$ systematically shifts the LLM's judgments relative to humans. The framework is LLM-agnostic, needs no model weights, and works on top of any API.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 210" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="210" rx="10" fill="#f8fafc"/>
  <rect x="250" y="20" width="120" height="42" rx="8" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="310" y="46" font-size="13" text-anchor="middle" fill="#1f2937">shared Z&#8341;</text>
  <rect x="60" y="140" width="150" height="42" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="135" y="160" font-size="12" text-anchor="middle" fill="#1f2937">human judgment</text>
  <text x="135" y="176" font-size="10" text-anchor="middle" fill="#555">Y&#8341; = f(Z&#8341;)</text>
  <rect x="410" y="140" width="150" height="42" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="485" y="160" font-size="12" text-anchor="middle" fill="#1f2937">LLM judgment</text>
  <text x="485" y="176" font-size="10" text-anchor="middle" fill="#555">Y&#8343; = f(&#946;Z&#8341; + &#947;&#7488;X)</text>
  <g stroke="#1f2937" stroke-width="1.6" fill="none" marker-end="url(#ah)">
    <path d="M290 62 L150 138"/><path d="M330 62 L470 138"/>
  </g>
  <text x="470" y="95" font-size="11" fill="#dc2626">&#947; = systematic bias on features X</text>
  <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker></defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">Human and LLM judgments share a latent preference; the LLM adds a linear bias term whose coefficients you can estimate and test.</figcaption>
</figure>

## What it buys you, and what it finds

Two applications. You can **recalibrate** an LLM judge with only a small number of human labels and get better accuracy, calibration, and KL alignment with human ratings, which matters when annotation is expensive and fine-tuning the judge is impractical or impossible behind an API. Even in the simplified no-covariate case ($\gamma = 0$), the recalibration helps, because the ordinal model imposes sensible inductive biases the raw LLM probabilities lack. And you can **detect and test** which features systematically shift a judge, with p-values, turning "LLM judges are biased" into specific, quantified claims.

Across six judges (GPT-4.1, GPT-4.1-nano, GPT-4o-mini, LLaMa-3.1-8B, Selene-1-Mini, Prometheus-v2) and two benchmarks (BigGen Bench and Chatbot Arena), the substantive findings are: LLM judges **favor brevity** (longer answers receive systematically *lower* scores), which contradicts the widely repeated claim of a pro-length bias, and the authors attribute the discrepancy to prior work aggregating at the system level rather than the instance level; **humans reward creativity, engagement, and positive sentiment more** than the judges do (intuitive on Chatbot Arena, where users are there to play with the model); and bias profiles **overlap considerably across judges**, pointing to shared biases inherited from similar training data and procedures. `Bridge` matches or beats both the raw LLM judgments and a naive logistic-regression baseline on every metric.

## The logit trick, and the theory that makes it honest

The name "logit trick" deserves unpacking, because it is the piece that makes the whole framework practical against a black-box API. The obstacle is that the latent human score $Z^h$ is never observed, and neither are the LLM's true latent scores $Z^l$; all you have are some human labels and the ability to query the judge. The trick is to notice that the LLM's rating probabilities $\mathbb{P}(Y^l = k \mid I, O)$ *can* be computed, and to back out the latent scores and cutoffs by matching model-implied probabilities to those:

$$\big(\{\eta_k\}, \{Z^l_i\}\big) = \arg\min \sum_i \sum_k \big| p_k(\bar\eta, z_i) - \mathbb{P}(Y^l = k \mid I_i, O_i)\big|,$$

after which you fit the human ordinal model on top with $Z^l$ in hand. There are two ways to get those LLM probabilities, and the difference is not cosmetic. For closed models you read them straight off the **log-probabilities** of the outcome tokens, which is exact and cheap but can be biased when the judge is prompted to reason first, because once the model has emitted a chain of thought its final-token probabilities are conditioned on that specific reasoning. For open models you instead **sample many chain-of-thought judgments** and estimate the probabilities by their empirical frequencies, which effectively marginalizes over reasoning paths, $\mathbb{P}(Y^l = k \mid I, O) = \sum_r \mathbb{P}(Y^l = k \mid I, O, R = r)\,\mathbb{P}(R = r \mid I, O)$, and tends to give higher-quality judgments at the cost of many more calls. In the binary case the trick has a clean interpretation: the latent score *is* the logit of the judge's "A wins" probability, which is where the name comes from.

What lifts this above a clever fitting procedure is that the estimators come with asymptotic theory. Under stated conditions the maximum-likelihood estimates of the cutoffs, the loading $\beta$, and the bias vector $\gamma$ are consistent and asymptotically normal, with a consistent variance estimator from the observed Fisher information. That is what turns "the judge seems to like short answers" into a hypothesis test with a confidence interval and a p-value, and it is why the calibration application is trustworthy rather than hopeful: recalibrated probabilities $p_k(\hat\alpha, \hat Z^h)$ are meant to satisfy a class-wise calibration property, $\mathbb{P}(Y^h = k \mid p_k = p) \approx p$, which is the same guarantee Platt scaling gives a classifier, now applied to an LLM judge. Statistical rigor is not usually the exciting part of an alignment paper, but here it is the whole reason the bias numbers are worth believing.

## My take

Two things make this paper worth its place next to the alignment work. First, the framing that corrections should be made *relative to human preference*, not in absolute terms. If a feature is valued by *both* humans and the LLM, then "debiasing" the LLM toward zero on that feature would actually *widen* the human-LLM gap rather than close it. That is a subtle, important point about what "aligning a judge" even means, and it is easy to get backwards: the goal is not a judge with no preferences, it is a judge whose preferences match ours.

Second, and this is the safety connection I care about most: every propensity measurement filtered through an LLM judge inherits that judge's systematic distortions. The OpenAgentSafety paper elsewhere in this collection showed LLM judges *miss* implicit unsafe reasoning; this paper shows they also carry consistent, *measurable* biases on surface features. Put those together and a "harmlessness score" produced by an LLM judge is a noisy, biased proxy for a human harm judgment, and `Bridge` at least makes the bias term explicit and testable instead of invisible. When an LLM judge sits inside a training loop, as it increasingly does, its biases become the model's incentives, so being able to name and correct them is not a niche concern.

My reservation is scope, and the paper is fairly honest about it. The interpretable covariates capture stylistic surface features (length, sentiment, formatting, structure), which is where the method's power and interpretability come from, but the divergences I worry about most in safety are not stylistic. They are about whether a judge and a human agree on whether a response is *dangerous*, and that content lives in dimensions the covariate approach does not obviously reach. The authors even note that on subjective, non-technical content the discrepancies shrink with more data, which suggests the framework is strongest exactly where the stakes are lowest and weakest where they are highest. Still, as a way to turn "LLM judges are biased" from a vibe into a hypothesis test with confidence intervals and FDR control, this is the cleanest tool I have seen, and I would want it running behind any evaluation pipeline that feeds a training loop.


What I ultimately take from this paper is a discipline rather than a result. It insists that if you are going to let a language model stand in for a human judge, you should be able to say, with a confidence interval, how and where it differs from the human it is replacing. That sounds obvious and is almost never done; most LLM-judge pipelines report agreement rates and stop. Turning the gap into an estimable, testable object is the move that lets you decide whether a judge is fit for a given purpose instead of assuming it, and in a world where these judges increasingly define what models are trained toward, the difference between assuming and measuring is the difference between an aligned incentive and a silently biased one.
