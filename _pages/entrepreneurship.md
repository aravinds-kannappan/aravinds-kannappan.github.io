---
layout: archive
title: "Founder"
permalink: /entrepreneurship/
author_profile: true
---

{% include base_path %}

Building startups taught me things that no course or research role could, about tradeoffs under uncertainty, the cost of technical debt at the worst possible time, and what it actually means to own a system end-to-end. The two ventures below shaped how I think about ML engineering as much as any paper or theorem.

---

## Synthure | Care Clarity AI
*Founder & ML Engineer · New York, NY · Jul 2023 to Jul 2025*

[Live platform → synthure.vercel.app](https://synthure.vercel.app/) · [Code → github.com/aravinds-kannappan/Synthure](https://github.com/aravinds-kannappan/Synthure)

Healthcare billing is a system designed, unintentionally or not, to fail patients. Claim denials cost U.S. providers over $260 billion annually, and the root cause is almost always the same: a mismatch between a clinical note written in natural language and a billing system built on rigid ICD-10 codes, payer-specific policy documents, and opaque adjudication rules. When I started Synthure the original insight was narrow, that retrieval over policy documents and coding guidelines could close the gap between what a clinician documents and what a payer expects. What it became was broader: a multi-agent clinical platform where a single clinical note enters once and a network of specialized agents handles everything downstream, from prior authorizations, claims routing, denial prediction, patient education, and appeal generation to employer benefits optimization, across four distinct stakeholder portals for patients, physicians, hospital billing teams, and employers.

The technical spine is a pipeline of typed agents that never pass raw dictionaries across boundaries. A quality-gate agent validates ICD-10/CPT codes and deduplicates before anything reaches inference; a NER cascade runs biomedical named-entity recognition, falls back to Claude Haiku, then to regex; a retrieval agent searches 1.43M ICD-10 codes via pgvector cosine similarity at MRR@5 of 0.91; a gradient-boosting denial predictor (AUC 0.87) routes high-risk claims to a frontier model and the rest to a cheaper one; and a generation agent forces a tool-use schema and validates every citation after the fact. The whole thing sits behind a three-tier autonomy model: actions that execute autonomously, actions that need one-tap physician approval, and a hard-coded prohibition on anything clinical like prescribing or diagnosis. By production it held NER accuracy at 94.2%, insurance-plan match at 91.3%, end-to-end p95 latency at 1.8s, citation drift at 2.3%, and fabricated clinical facts at zero. Along the way I moved it off heavier infrastructure onto a leaner Vercel + Supabase + pgvector stack with Claude Haiku and Sonnet doing the generation.

Looking back, the most important thing I learned at Synthure wasn't technical, it was about the hidden cost of building for a domain you don't fully control. Healthcare moves slowly for reasons that have nothing to do with technology: procurement cycles, compliance reviews, trust-building with clinical staff. We built a genuinely good system faster than the market could adopt it, and that gap between product readiness and organizational readiness is something I didn't fully appreciate until I lived it. I also learned that production ML in healthcare demands a different kind of rigor than research ML. Calibration isn't a nice-to-have when a miscalibrated confidence score could influence a clinical decision. Drift monitoring isn't optional when payer policies update quarterly. These experiences permanently changed how I think about what it means to build responsibly.

---

## Replays AI
*Software Engineer, AI/ML · New York, NY · Sep 2025 to Apr 2026*

[Live platform → replays-ai.vercel.app](https://replays-ai.vercel.app/) · [Code → github.com/aravinds-kannappan/ReplaysAI](https://github.com/aravinds-kannappan/ReplaysAI)

Sports media has a content abundance problem: every game generates hours of footage, thousands of structured events, and an audience that wants personalized, contextual highlights, not a broadcast edited for the median fan. Replays AI was built on the thesis that multimodal AI could close this gap, and over time it grew from a recap generator into a full personalized fan command center. After login the app opens into one tabbed dashboard with a personalized feed, live games, an assistant chat, predictions, a roster outlook, agent-generated reels, and an agent-status view, all in one place.

The engineering forced me to think seriously about pipeline architecture in a way that academic projects never quite do. The platform pulls real NBA and NFL data straight from ESPN's public endpoints (no API key, derived at request time), authenticates through Clerk, and runs a FastAPI backend behind a React/TypeScript/Vite client. The inference layer is four agents coordinated with `asyncio.gather`: a pure-Python event-extraction agent for scoring runs and clutch moments, a Claude Vision agent that classifies play types from extracted video frames, a four-way parallel LLM summarization agent that writes the recap, and an on-demand fan-perspective agent that rewrites any recap from your team's point of view, celebratory for a win and an honest post-mortem for a loss. On top of that sits a gamification layer of predictions, points, streaks, badges, leaderboards, and weekly roster building. The most interesting constraint was making it run on Vercel with no database at all, deriving everything live from ESPN with optional Redis caching and permanent recap caching to keep response times low.

What Replays taught me above all else was how to reason about latency as a first-class product requirement, not an afterthought. In research settings, a model that takes 10 seconds to generate an answer is fine. In a mobile app serving sports fans who want their recap immediately after the final buzzer, 10 seconds is a product failure. Every architectural decision we made, from running the expensive CV and LLM steps in parallel to caching recaps to deriving data at request time, was driven by a latency budget, and working backward from that budget shaped the entire system design. It also deepened my appreciation for the difference between a system that works and a system that scales: the former is a proof of concept, the latter is an engineering discipline.
