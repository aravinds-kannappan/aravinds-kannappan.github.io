---
title: "The Doubling Clock: Measuring How Long a Task an AI Can Finish"
excerpt: "METR replaces saturating benchmarks with a human-anchored metric: the length of task, in human time, that a model completes half the time. It has been doubling roughly every seven months."
collection: research
order: 4
date: 2025-03-18
tags:
  - evaluations
  - forecasting
  - AI safety
---

Benchmark scores are almost useless for the question people actually care about: how capable is this model, really, and how fast is that changing? Scores saturate within a year, they are gamed by adversarial filtering (HellaSwag and Humanity's Last Exam were both built by filtering against the best models of the day), and a number on any single benchmark does not tell you whether a model can do a day of your job. "Measuring AI Ability to Complete Long Software Tasks" by Kwa, West, Becker and colleagues at METR (2025) proposes a metric that sidesteps all of this by anchoring capability to *human time*. It is the rare eval paper whose central quantity I find genuinely intuitive, and whose main result I have not been able to stop thinking about.

---

## The metric

Define the **50% task-completion time horizon** as the length of task, measured by how long skilled humans take to do it, that a model completes with 50% success. The construction is deliberately close to item response theory from psychometrics. For each agent, you fit a logistic curve relating its success probability on a task to that task's human-time length:

$$p_{\text{success}}(\text{agent}, \text{task}) = \sigma\big((\log h_{\text{agent}} - \log t_{\text{task}}) \cdot \beta_{\text{agent}}\big),$$

where $t_{\text{task}}$ is the geometric-mean time successful humans take, $\sigma$ is the logistic function, $\beta_{\text{agent}}$ controls how sharply success falls off with length, and $h_{\text{agent}}$ is the fitted horizon: the task length at which the agent's success probability crosses 50%. The crucial difference from standard IRT is that difficulty is not learned from the agents' own performance; it is grounded in real human clock time, which keeps the ruler independent of the thing being measured.

They build that ruler from 170 tasks across three suites chosen to span an enormous range. **HCAST** contributes 97 diverse software tasks from one minute to thirty hours. **RE-Bench** adds seven difficult ML-research-engineering tasks, each roughly eight hours. And **SWAA** (software atomic actions) supplies 66 tiny one-to-thirty-second tasks, added specifically so that weak pre-2023 models register a non-zero horizon at all. Difficulty is calibrated with over **800 hours of baselining** from skilled professionals, most from top-100 universities, with software engineers generally having more experience than the ML or cybersecurity baseliners. Continuously scored tasks are binarized against the human target score.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 260" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="260" rx="10" fill="#f8fafc"/>
  <line x1="70" y1="220" x2="600" y2="220" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="70" y1="30" x2="70" y2="220" stroke="#1f2937" stroke-width="1.5"/>
  <text x="335" y="250" font-size="12" text-anchor="middle" fill="#1f2937">model release date, 2019 &#8594; 2025</text>
  <text x="24" y="125" font-size="12" text-anchor="middle" fill="#1f2937" transform="rotate(-90 24 125)">50% time horizon (log scale)</text>
  <g font-size="10" fill="#94a3b8" text-anchor="end">
    <line x1="70" y1="190" x2="600" y2="190" stroke="#e2e8f0"/><text x="64" y="194">2 s</text>
    <line x1="70" y1="140" x2="600" y2="140" stroke="#e2e8f0"/><text x="64" y="144">2 min</text>
    <line x1="70" y1="90" x2="600" y2="90" stroke="#e2e8f0"/><text x="64" y="94">30 min</text>
    <line x1="70" y1="55" x2="600" y2="55" stroke="#e2e8f0"/><text x="64" y="59">2 hr</text>
  </g>
  <line x1="90" y1="195" x2="560" y2="70" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6 4"/>
  <g fill="#dc2626">
    <circle cx="100" cy="192" r="4"/><circle cx="200" cy="168" r="4"/><circle cx="300" cy="150" r="4"/>
    <circle cx="400" cy="120" r="4"/><circle cx="470" cy="98" r="4"/><circle cx="540" cy="78" r="4"/>
  </g>
  <text x="120" y="188" font-size="10" fill="#555">GPT-2</text>
  <text x="500" y="72" font-size="10" fill="#555">o3 &#8776; 110 min</text>
  <text x="330" y="45" font-size="12" fill="#2563eb" text-anchor="middle">doubling &#8776; every 7 months</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">On a log axis the horizon is close to a straight line: exponential growth, with a roughly seven-month doubling time.</figcaption>
</figure>

## The result

Fitting an ordinary-least-squares line to $\log(\text{horizon})$ against release date, they find the 50% time horizon has grown **exponentially since 2019, doubling about every seven months** (207 days, with a 95% bootstrap interval of 166 to 240 days), and an $R^2$ near **0.97**. Frontier models like o3 sit around a **110-minute** horizon; GPT-2 was about **two seconds**. The 2024-to-2025 subset looks roughly 20% faster than the six-year trend. A separate replication on SWE-bench Verified reproduces the same exponential shape with an even steeper (roughly 70-day-doubling) slope, though the authors note SWE-bench's own time annotations likely underestimate contractor time and thereby shorten the apparent doubling.

Two details keep me honest about what this measures. The gains are not one skill: qualitative analysis of failed transcripts attributes progress to greater **reliability, the ability to adapt to mistakes, logical reasoning, and tool use** rather than any single capability. And the **80% horizon is four to six times shorter than the 50% horizon**, a quiet reminder that being *reliable* is much harder than being occasionally right. A model that can sometimes do a two-hour task is not a model you can hand a two-hour task and walk away from.

## The failure modes, and what they imply

I appreciate that the paper reads transcripts rather than only fitting curves. Categorizing failed runs, they find older models fail mostly by *repeating failed actions* and poor tool choice, while newer models fail on harder tasks in more human-like ways: incorrect reasoning, premature task abandonment, and poor planning. Crucially, models struggle most on "messy" tasks, those without clear feedback loops or where the agent must proactively seek out information nobody handed it. Controlling for length, higher "messiness" (they score tasks against 16 such factors) predicts worse performance, and on a small uncontaminated set of their own internal pull requests, agents did worse than the maintainer time-to-complete would predict.

## The extrapolation

If the trend continues and generalizes, a **one-month (167 work-hour) horizon** on software tasks lands somewhere between **mid-2028 and mid-2030** (80% confidence, central estimate around mid-2029), and as early as 2027 if the faster recent slope holds. The choice of one month is deliberate: it is roughly the point at which a new employee begins to generate real economic value, and a system that can autonomously carry a month-long software project would be transformative, including for catastrophic misuse. The authors are careful that this is a horizon on *software tasks*, not a claim about general one-month AGI, and that the extrapolation's uncertainty is dominated by external validity and possible trend changes, not by the curve fit.

## How much to trust the curve

Before leaning on a six-year exponential, it is worth understanding why the authors trust the slope but not the individual points, because the answer is a small lesson in reading capability trends. Each model's horizon is estimated by a logistic fit over a finite, lumpy set of tasks, and tasks at the same nominal human-time rating vary enormously in how hard they are for a model. Sample an easy subset and *every* model's horizon estimate rises together; sample a hard subset and they all fall together. That means the errors on individual horizons are large *and correlated across models*, which is exactly the structure that leaves the *slope* well-determined even when each point wobbles. They quantify this with a three-level hierarchical bootstrap, resampling over task families, then tasks, then runs, and report that the fit is insensitive to the nuisance choices you would worry about: regularization, task weighting, and ordinary-least-squares versus weighted least squares all leave the doubling time near seven months.

The external-validity checks are where I give them the most credit, because a lesser paper would have stopped at the headline. The SWE-bench Verified replication is a genuine out-of-distribution test on an industry-standard benchmark built from real GitHub issues, and it reproduces the exponential shape, though with an important asterisk: SWE-bench's time annotations assume an engineer who has spent a few hours familiarizing themselves with the codebase, which underestimates how long the task takes a fresh contractor, and that differential compression is what shortens the apparent doubling time to roughly 70 days rather than confirming a faster true trend. The "messiness" study is the other honest move. They score tasks against sixteen factors that separate clean benchmark tasks from hard real-world ones (resource limits, dynamic environments, the need to proactively seek information, absence of a clean feedback loop) and find that, controlling for length, higher messiness predicts worse performance, and on their own uncontaminated internal pull requests the models underperform what maintainer time-to-complete would predict. None of this breaks the trend, but all of it tells you the trend lives in a cleaner world than the one where the economic value is, which is precisely the gap between "can finish a well-specified two-hour task" and "can do two hours of your actual job."

## My take

I trust the *slope* here far more than any single point, and the authors clearly do too, which is why they lean on it and downplay individual horizon estimates. The per-model error bars are wide and correlated (sample an easy set of tasks and every model looks better at once), but a clean exponential over six years, across three task suites and an independent SWE-bench replication, is very hard to dismiss as an artifact.

Where I stay skeptical is external validity, and to their credit that is the paper's own loudest caveat. These tasks are automatically scored, self-contained, and free of the things that make real work hard: dynamic environments, other agents pushing back, resource constraints, and the need to go find information that was not provided. Their own messiness analysis shows the gap. So I read the doubling clock as a lower bound on *structured* capability, not a forecast of when an AI can do your actual job, which is messier by construction. Even with that discount the safety framing is correct, and it is the reason the paper belongs next to the extreme-risk work rather than in a leaderboard: the argument is not "the model is smart," it is "time horizon is a proxy for autonomy, and autonomy is what converts a dangerous capability into a dangerous action." A ruler this legible is exactly the kind of instrument governance can attach a policy to, which may be the most durable thing the paper produces, more durable than the specific 2029 estimate that will inevitably be revised as new points land on the line.


The last thing I appreciate is that the paper resists the temptation to editorialize its own number. It would have been easy to lead with '1-month AI by 2029' and let the caveats fade into an appendix. Instead the caveats are load-bearing, stated in the main text, and honestly capable of moving the estimate by years in either direction. That restraint is what makes the metric useful as infrastructure rather than as a headline: a doubling clock you can keep re-reading as new models land, updating the forecast without discarding the framework. I expect the specific date to be wrong and the ruler to outlive it, which is exactly the right ambition for a measurement paper in a field this fast.
