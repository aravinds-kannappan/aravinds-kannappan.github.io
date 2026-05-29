---
layout: archive
title: "Entrepreneurship"
permalink: /entrepreneurship/
author_profile: true
---

{% include base_path %}

Building companies taught me things that no course or research role could — about tradeoffs under uncertainty, the cost of technical debt at the worst possible time, and what it actually means to own a system end-to-end. The two ventures below shaped how I think about ML engineering as much as any paper or theorem.

---

## Synthure | Care Clarity AI
*Founder & ML Engineer · New York, NY · Jul 2023 – Jul 2025*

Healthcare billing is a system designed, unintentionally or not, to fail patients. Claim denials cost U.S. providers over $260 billion annually, and the root cause is almost always the same: a mismatch between a clinical note written in natural language and a billing system built on rigid ICD-10 codes, payer-specific policy documents, and opaque adjudication rules. When I started Synthure, I wasn't trying to build an LLM product — I was trying to fix a specific, painful, and well-documented failure mode that I saw firsthand while working with clinical teams. The original insight was that RAG over policy documents and medical coding guidelines could close the gap between what a clinician documents and what a payer expects to see, without requiring clinicians to become billing experts.

The technical journey was harder than the idea. Building a retrieval pipeline over Snowflake for medical code classification sounds straightforward until you're dealing with the reality of medical data: inconsistent schemas across EHR vendors, entity disambiguation across thousands of provider IDs, and a regulatory environment where a hallucinated code isn't just a wrong answer — it's a HIPAA violation and a potential fraud liability. We built a typed intermediate representation layer that validated schema, ran deduplication, and scored entity extraction confidence before anything reached downstream inference. We used multi-agent orchestration with task-split routing — lighter models for high-volume entity tagging, frontier LLMs via Claude APIs for plain-language generation — and implemented gated controlled generation that constrained outputs to source-grounded rewrites. By the time we reached production scale, we were processing 60K+ records with vLLM serving at under 1.8s p95 on AWS, with HIPAA-aligned JWT access control throughout.

Looking back, the most important thing I learned at Synthure wasn't technical — it was about the hidden cost of building for a domain you don't fully control. Healthcare moves slowly for reasons that have nothing to do with technology: procurement cycles, compliance reviews, trust-building with clinical staff. We built a genuinely good system faster than the market could adopt it, and that gap between product readiness and organizational readiness is something I didn't fully appreciate until I lived it. I also learned that production ML in healthcare demands a different kind of rigor than research ML. Calibration isn't a nice-to-have when a miscalibrated confidence score could influence a clinical decision. Drift monitoring isn't optional when payer policies update quarterly. These experiences permanently changed how I think about what it means to build responsibly.

---

## Replays AI
*Software Engineer, AI/ML · New York, NY · Sep 2025 – Apr 2026*

Sports media has a content abundance problem: every game generates hours of footage, thousands of structured events, and an audience that wants personalized, contextual highlights — not a broadcast edited for the median fan. Replays AI was built on the thesis that multimodal AI could close this gap: ingest the structured play-by-play data, align it to raw video via computer vision, and use LLMs to generate recaps that actually reflect how a specific fan watches the game. The underlying technical problem was more interesting than it first appeared — it required combining three fundamentally different inference modalities (structured event data, visual content, and language generation) in a pipeline fast enough to serve users who expected sub-second response times on a mobile app.

The engineering at Replays forced me to think seriously about pipeline architecture in a way that academic projects never quite do. We were ingesting 5M+ structured play-by-play events into PostgreSQL, with raw video stored in GCP referenced by pointer to decouple storage from compute. The real challenge was the inference layer: three parallel agentic pipelines — event feature extraction, CV play classification, and task-split LLM summarization — had to be orchestrated so that the most expensive steps (CV inference and LLM generation) ran in parallel without becoming the bottleneck for the entire recap flow. We reduced inference latency by 35% through careful parallelism design, and the REST API layer with recap caching kept live ranking endpoints at sub-second response times even under load. Building the React/TypeScript mobile client on top of this — and reducing mean load time by 28% through component memoization — made me a better full-stack engineer than I expected to become.

What Replays taught me above all else was how to reason about latency as a first-class product requirement, not an afterthought. In research settings, a model that takes 10 seconds to generate an answer is fine. In a mobile app serving sports fans who want their recap immediately after the final buzzer, 10 seconds is a product failure. Every architectural decision we made — from the pointer-based video storage to the parallel inference pipelines to the caching strategy — was driven by a latency budget, and working backward from that budget shaped the entire system design. It also deepened my appreciation for the difference between a system that works and a system that scales: the former is a proof of concept, the latter is an engineering discipline.
