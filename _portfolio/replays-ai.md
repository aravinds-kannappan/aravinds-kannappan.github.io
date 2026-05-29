---
title: "Replays AI — AI-Powered Post-Game Recaps"
excerpt: "Multi-agent sports analytics platform processing 5M+ play-by-play events with parallel CV and LLM inference pipelines, serving sub-second response times."
collection: portfolio
date: 2025-09-01
---

**Role:** Software Engineer, AI/ML · Sep 2025 – Apr 2026

An immersive post-game recap system orchestrating computer vision and LLM inference over structured sports event data at scale.

**Key work:**
- Designed ingestion pipeline writing 5M+ structured play-by-play events into PostgreSQL; stored raw video in GCP referenced by pointer, cutting compute costs by 40%
- Orchestrated 3 parallel agentic inference pipelines — event feature extraction, CV play classification, and task-split LLM summarization — reducing inference latency by 35%
- Decoupled storage and inference using REST APIs with recap caching, sustaining sub-second response times
- Built React + TypeScript mobile app rendering live rankings and CV-scored highlight reels; reduced mean load time by 28% via component memoization

**Stack:** Python, PostgreSQL, GCP, REST APIs, React, TypeScript, LangGraph, Computer Vision, Docker
