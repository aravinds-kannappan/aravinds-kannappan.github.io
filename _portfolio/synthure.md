---
title: "Synthure | Care Clarity AI"
excerpt: "Healthcare LLM platform with RAG over Snowflake, multi-agent orchestration, and HIPAA-compliant vLLM serving at sub-2s p95 latency."
collection: portfolio
date: 2023-07-01
---

**Role:** Founder & ML Engineer · Jul 2023 – Jul 2025

A production clinical AI platform that reduced policy misinterpretation errors by 94% across clinical workflows through RAG-powered medical code classification and claim denial routing.

**Key work:**
- Built RAG retrieval pipeline over Snowflake for medical code classification and claim denial routing
- Designed multi-agent orchestration with task-split LLM routing via Claude APIs — lighter models for entity tagging, frontier LLMs for plain-language generation
- Implemented gated controlled generation, reducing hallucinated content across 60K+ records
- Deployed vLLM inference at under 1.8s p95 on AWS with MongoDB and Redis microservices
- Enforced HIPAA-aligned JWT access control and input validation across all services

**Stack:** Python, LangChain, vLLM, Snowflake, AWS (EC2, S3, Bedrock), MongoDB, Redis, Docker, FastAPI
