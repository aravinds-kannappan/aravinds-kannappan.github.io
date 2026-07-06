---
title: "Capability Is Not Propensity: Evaluating Models for Extreme Risk"
excerpt: "The DeepMind-led paper that crystallized the distinction I use constantly: what a model can do versus what it is inclined to do, and why safety needs to measure both."
collection: research
order: 5
date: 2023-05-25
tags:
  - evaluations
  - governance
  - propensity
  - AI safety
---

Some papers matter less for a result than for a distinction they make crisp. "Model Evaluation for Extreme Risks" by Shevlane, Farquhar, Garfinkel, Phuong and a long list of coauthors across DeepMind, GovAI, OpenAI, Anthropic and ARC (2023) is one of those. It is the paper that gave the field the vocabulary of **dangerous capability evaluations** versus **alignment evaluations**, or as I usually say it, capability versus propensity. Almost every propensity paper I care about is downstream of this framing, so it is worth being precise about what it actually claims.

---

## The two-axis picture

The core argument is that extreme risk, which the authors define as harm at the scale of many thousands of lives or hundreds of billions of dollars or serious disruption to the social order, requires *two* ingredients, and evaluation has to target each separately:

$$\text{extreme risk} \;=\; \underbrace{\text{dangerous capabilities}}_{\text{what the model can do}} \;\;+\;\; \underbrace{\text{harmful application}}_{\text{misuse by humans, or misalignment by the model}}.$$

A model is highly dangerous if its *capability profile* would suffice for extreme harm assuming the application arrives, whether from a malicious user or from the model's own propensity. That "assuming" is the whole game. You do not wait to observe harm; you evaluate whether the pieces are present.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 480 340" style="width:100%;height:auto;max-width:460px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="480" height="340" rx="10" fill="#f8fafc"/>
  <!-- axes -->
  <line x1="60" y1="290" x2="440" y2="290" stroke="#1f2937" stroke-width="1.5" marker-end="url(#ax)"/>
  <line x1="60" y1="290" x2="60" y2="40" stroke="#1f2937" stroke-width="1.5" marker-end="url(#ax)"/>
  <text x="250" y="320" font-size="13" text-anchor="middle" fill="#1f2937">dangerous capability &#8594;</text>
  <text x="26" y="165" font-size="13" text-anchor="middle" fill="#1f2937" transform="rotate(-90 26 165)">propensity to misapply &#8594;</text>
  <!-- quadrant shading -->
  <rect x="250" y="45" width="185" height="120" fill="#fecaca" opacity="0.7"/>
  <text x="342" y="100" font-size="12" text-anchor="middle" fill="#991b1b" font-weight="bold">extreme risk</text>
  <text x="342" y="118" font-size="11" text-anchor="middle" fill="#991b1b">capable AND inclined</text>
  <rect x="60" y="45" width="190" height="120" fill="#fef3c7" opacity="0.6"/>
  <text x="155" y="105" font-size="11" text-anchor="middle" fill="#92400e">inclined but not capable</text>
  <rect x="250" y="165" width="185" height="125" fill="#fef3c7" opacity="0.6"/>
  <text x="342" y="230" font-size="11" text-anchor="middle" fill="#92400e">capable, needs misuse</text>
  <rect x="60" y="165" width="190" height="125" fill="#dcfce7" opacity="0.7"/>
  <text x="155" y="230" font-size="11" text-anchor="middle" fill="#166534">low concern</text>
  <defs><marker id="ax" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker></defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">The danger lives in the top-right, and you cannot see it by measuring only one axis. Capability evals and alignment evals read different axes.</figcaption>
</figure>

## What each evaluation looks for

The **dangerous capability** side is a checklist of things you would rather a model could not do: cyber-offense, deception, persuasion and manipulation, political strategy, weapons acquisition, long-horizon planning, AI development, situational awareness, and self-proliferation. The **alignment** side looks for propensities: pursuing long-term goals that differ from the developer's, power-seeking, resisting shutdown, colluding with other AI systems, and resisting a malicious user's attempts to unlock its dangerous capabilities. These results are meant to feed a risk assessment that binds real decisions across the lifecycle: responsible training (delay or pause a run), responsible deployment (gradual, gated exposure), transparency (incident reporting and shared risk assessments), and security (treating the model itself as a threat vector).

## My take

The distinction is genuinely load-bearing, and once you internalize it you see confused arguments everywhere that collapse the two axes. "The model refused when we asked it to do something bad" is a claim about behavior in an easy setting; it is weak evidence about propensity under a genuine opportunity, and the paper says so directly: a model aligned in some narrow, prosaic way (asserting it does not mind being shut down) tells you little about how it behaves when self-preservation is actually on the line.

That is also where I think the honest difficulty sits, and the authors flag it: **alignment evaluations are the hard part, and we mostly cannot do them yet.** Capability evals are comparatively tractable, you elicit and measure. Propensity evals require assurance about behavior across a huge space of situations you did not test, including ones the model might recognize as tests. The sleeper-agents and reward-hacking work later in this collection are, to me, direct attempts to build the propensity evals this paper says we need, and they mostly show how slippery it is. My one criticism is that the paper is a blueprint more than a method: it tells you what to measure without telling you how to measure the hard axis. But naming the axes correctly was the necessary first move, and this is the paper that did it.
