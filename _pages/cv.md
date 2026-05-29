---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

## Education

* **MS in Applied Statistics**, New York University, Aug 2024 – Expected Aug 2026
  * Specialization in Machine Learning
  * Coursework: NLP, Deep Learning, Data Engineering, Distributed Systems, Big Data, Bayesian Methods, Biostatistics

* **BS in Statistics**, Baylor University, Aug 2020 – Aug 2024
  * Minor in Biology · Honors College · BU Chess Club
  * Coursework: Data Structures & Algorithms, Linear Algebra, Calculus, Database Systems, System Design, Operating Systems

---

## Work Experience

**Machine Learning Engineer** — NYU AI Whistleblower Initiative \| CORDA AI Fellow
*Apr 2026 – Present · Remote*
- Architected an RLHF pipeline over a large language model, collecting annotations from 10+ experts and converting pairwise ratings into a reward model, improving precision by 30% over baseline zero-shot prompting
- Built a drift detection layer computing KL-divergence across consecutive model output distributions, triggering remediation workflows when shift exceeded a tuned threshold across 500+ weekly sessions
- Constructed a semantic clustering pipeline vectorizing annotated session transcripts with sentence-transformers and FAISS, reducing retrieval latency by 60%
- Evaluated prompt engineering methods on expert-labeled datasets; selected chain-of-thought, reducing hallucinated legal citations by 22% on held-out jurisdiction-specific tests
- Feedback fine-tuned via LoRA adapter training, cutting full model retraining cost by 80% while maintaining within 2% on domain evals

**Machine Learning Engineer** — Supervised Program for Alignment Research (SPAR)
*Feb 2026 – May 2026 · Remote*
- Investigated AI safety properties of frontier LLMs using Bayesian methods to model AI consciousness; designed experiments across 3 models using Python and PyTorch
- Fine-tuned LLMs to generate 6 semantically equivalent question variants per safety indicator; measured cross-variant output distributions to stress-test alignment stability, reducing variance by 30%
- Developed mathematical models characterizing ground-truth drift in LLM judgment across model generations; analyzed how scaling and RLHF-driven training affect model behavior
- Built statistical analysis over response distributions including Likert histograms, cross-model EDA, and semantic analysis of chain-of-thought justifications across 50+ safety-relevant indicators
- Open-sourced full evaluation framework on GitHub for reproducible AI safety research

**Software Engineer, AI/ML** — Replays AI
*Sep 2025 – Apr 2026 · New York, NY*
- Designed ingestion pipeline writing 5M+ structured play-by-play events into PostgreSQL as system of record; stored raw video in GCP referenced by pointer, cutting compute costs by 40%
- Preprocessed using Python to handle schema validation, player disambiguation, and deduplication; anchored video segmentation to event timestamps before Computer Vision inference
- Orchestrated 3 parallel agentic inference pipelines spanning event feature extraction, CV play classification, and task-split LLM summarization; reduced inference latency by 35%
- Decoupled storage and inference from the frontend using REST APIs; implemented recap caching sustaining sub-second response times
- Built React and TypeScript mobile app rendering live rankings, game recaps, and CV-scored highlight reels; reduced mean load time by 28% via component memoization

**Founder & ML Engineer** — Synthure \| Care Clarity AI
*Jul 2023 – Jul 2025 · New York, NY*
- Built RAG retrieval pipeline over Snowflake for medical code classification and claim denial routing; reduced policy misinterpretation errors by 94% across production clinical workflows
- Engineered typed intermediate representation and data quality gate validating schema, deduplication, and entity extraction confidence before downstream inference
- Designed multi-agent orchestration with task-split LLM routing using Claude APIs; directed high-volume entity tagging to lighter models and plain-language generation to frontier LLMs
- Implemented gated controlled generation constraining outputs to source-grounded entity rewrites; reduced hallucinated content across 60K+ records
- Deployed vLLM inference at under 1.8s p95 latency on AWS within MongoDB and Redis microservices; enforced HIPAA-aligned JWT access control and input validation across all services

---

## Projects

**CSCI 2271 Computer Vision** — [GitHub](https://github.com/aravinds-kannappan)
- Trained diffusion world model on 737K+ frames with VAE, CNN reward model, and PPO agents

**RobustSight: Advancing AI Safety and Alignment** — [GitHub](https://github.com/aravinds-kannappan)
- AI safety framework investigating the intersection of adversarial robustness and interpretability

---

## Publications

<ul>{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}</ul>

---

## Skills

**Languages:** Python, TypeScript, SQL, R, C++

**AI/ML Frameworks:** LangChain, LangGraph, PyTorch, HuggingFace, vLLM, Scikit-learn, TensorFlow, FAISS, RAG, RLHF, Prompt Engineering, Computer Vision

**Frontend:** React, REST APIs

**Cloud & DevOps:** AWS (S3, EC2, Bedrock), GCP, Docker, Kubernetes, CI/CD, Git, PostgreSQL, MongoDB, Redis, Snowflake
