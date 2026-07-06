---
title: "Covera: An Insurance Marketplace That Texts You the Right Plan"
excerpt: "A multi-agent concierge you text like a person: it simulates thousands of years of your real care against every real plan, ranks on risk-adjusted all-in cost, and stays on as a year-round advocate."
collection: portfolio
date: 2026-06-20
---

Covera turns the near-impossible task of choosing a health plan into a text conversation. You describe your situation, a team of agents builds your profile, and a cost engine estimates what you would actually pay: not the premium (one number) but the full distribution of a year of unpredictable care. Every figure traces to public data, with no synthetic plans, prices, or claims.

**Key work:** Risk-adjusted plan ranking that scores expected cost plus a downside-risk penalty (p90 cost, probability of hitting the out-of-pocket max), because the top 5% of people drive roughly half of all spending and a single average hides the year that bankrupts people · A closed-form cost engine (milliseconds per plan) plus a variance-reduced Monte-Carlo pass that samples care from real AHRQ MEPS utilization data and runs it through each plan's actual deductible, coinsurance, and OOP-max rules · Real CMS formulary matching, so a plan that drops one of your drugs takes a ranking penalty and gets flagged · Three interactive lenses on one engine: a patient optimizer, an employer ICHRA modeler, and a hospital cost desk with a browser-side bill auditor · A portable Coverage Card that shows a provider a live cost estimate with zero access to your records · Year-round agents that estimate a procedure before you book it, audit a suspicious bill, and draft an appeal for a denial

**Stack:** TypeScript, Next.js, multi-agent LLM orchestration, Monte-Carlo simulation, real CMS and AHRQ data

[GitHub →](https://github.com/aravinds-kannappan/Covera) · [Live demo →](https://covera-six.vercel.app)
