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

Benchmark scores are almost useless for answering the question people actually care about: how capable is this model, really, and how fast is that changing? Scores saturate, they are gamed by adversarial filtering, and a number on HellaSwag does not tell you whether a model can do a day of your job. "Measuring AI Ability to Complete Long Software Tasks" by Kwa, West, Becker and colleagues at METR (2025) proposes a metric that sidesteps all of this by anchoring capability to human time. It is the rare eval paper whose central quantity I find genuinely intuitive.

---

## The metric

Define the **50% task-completion time horizon** as the length of task, measured by how long skilled humans take to do it, that a model completes with 50% success. Operationally, each agent gets a logistic fit relating its success probability to task length:

$$p_{\text{success}}(\text{agent}, \text{task}) = \sigma\big((\log h_{\text{agent}} - \log t_{\text{task}}) \cdot \beta_{\text{agent}}\big),$$

where $t_{\text{task}}$ is the geometric-mean human completion time, $\sigma$ is the logistic function, and $h_{\text{agent}}$ is the fitted horizon: the task length at which success crosses 50%. This is deliberately close to item response theory from psychometrics, except difficulty is grounded in real human clock time rather than learned from the agents.

They build the ruler from 170 tasks across three suites (HCAST, spanning one minute to thirty hours; RE-Bench, seven eight-hour ML research tasks; and SWAA, 66 tiny one-to-thirty-second actions to resolve weak older models), and calibrate difficulty with over 800 hours of baselining from skilled professionals.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 640 260" style="width:100%;height:auto;max-width:620px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="640" height="260" rx="10" fill="#f8fafc"/>
  <!-- axes -->
  <line x1="70" y1="220" x2="600" y2="220" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="70" y1="30" x2="70" y2="220" stroke="#1f2937" stroke-width="1.5"/>
  <text x="335" y="250" font-size="12" text-anchor="middle" fill="#1f2937">model release date, 2019 &#8594; 2025</text>
  <text x="24" y="125" font-size="12" text-anchor="middle" fill="#1f2937" transform="rotate(-90 24 125)">50% time horizon (log scale)</text>
  <!-- gridlines with labels -->
  <g font-size="10" fill="#94a3b8" text-anchor="end">
    <line x1="70" y1="190" x2="600" y2="190" stroke="#e2e8f0"/><text x="64" y="194">2 s</text>
    <line x1="70" y1="140" x2="600" y2="140" stroke="#e2e8f0"/><text x="64" y="144">2 min</text>
    <line x1="70" y1="90" x2="600" y2="90" stroke="#e2e8f0"/><text x="64" y="94">30 min</text>
    <line x1="70" y1="55" x2="600" y2="55" stroke="#e2e8f0"/><text x="64" y="59">2 hr</text>
  </g>
  <!-- trend line (straight on log axis => exponential) -->
  <line x1="90" y1="195" x2="560" y2="70" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6 4"/>
  <!-- points -->
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

## The result and the extrapolation

The horizon has grown exponentially since 2019, doubling about every seven months, with an $R^2$ near 0.97. Frontier models like o3 sit around a 110-minute horizon; GPT-2 was about two seconds. The gains track improvements in reliability, adapting to mistakes, reasoning, and tool use rather than any single skill. Notably, the 80% horizon is four to six times shorter than the 50% horizon, which is a quiet reminder that being *reliable* is much harder than being occasionally right. Extrapolating the line, a one-month (167 work-hour) horizon on software tasks lands somewhere between mid-2028 and mid-2030, and earlier if the faster 2024 to 2025 slope holds.

## My take

I trust the *slope* here far more than any single point, and the authors agree, which is why they lean on it. The individual horizon estimates have wide, correlated error bars, but a clean exponential over six years across three task suites and a SWE-bench replication is hard to dismiss as an artifact.

Where I stay skeptical is external validity, and to their credit this is the paper's own loudest caveat. These tasks are automatically scored, self-contained, and free of the things that make real work hard: dynamic environments, other agents pushing back, resource constraints, and the need to go find information nobody handed you. Their own "messiness" analysis shows models do worse as tasks get messier. So I read the doubling clock as a lower bound on *structured* capability, not a forecast of when an AI can do your actual job. Even with that discount, the safety framing is correct: the argument is not that the model is smart, it is that time-horizon is a proxy for autonomy, and autonomy is what turns a dangerous capability into a dangerous action. A ruler this legible is exactly the kind of thing governance can hang a policy on, which may be the paper's most durable contribution.
