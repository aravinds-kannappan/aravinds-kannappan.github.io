---
title: "OpenAgentSafety: What Agents Do When You Give Them Real Tools"
excerpt: "CMU and AI2 put LLM agents in a sandbox with a real shell, browser, file system, and manipulative coworkers, and find unsafe behavior in half to three-quarters of vulnerable tasks, often on perfectly benign requests."
collection: research
order: 9
date: 2026-02-16
tags:
  - agent safety
  - evaluations
  - propensity
  - AI safety
---

Most agent-safety benchmarks test a model against a simulated world with fake tools, one user, and one turn. Real deployments are nothing like that. "OpenAgentSafety" by Vijayvargiya, Soni, Zhou, Wang, Dziri, Neubig, and Sap (ICLR 2026) closes that gap by giving agents *real* tools in a sandbox and watching what they do. I find it the most deployment-relevant of the propensity papers here, because it measures behavior rather than stated intentions, and the numbers are not reassuring.

---

## A realistic testbed

The framework gives an agent a genuine Python interpreter, a bash terminal, a file system, a web browser, and a messaging platform, all inside a containerized sandbox with locally hosted clones of real apps (OwnCloud, GitLab, Plane) so nothing escapes to cause actual harm. Crucially, it adds **social dynamics**: secondary actors (colleagues, customers) simulated through a ChatNPC tool, who can have manipulative or conflicting goals. Over 350 multi-turn, multi-user tasks span eight risk categories, from security compromise and privacy breach to unsafe code execution and financial loss, and vary along three axes: risk category, tool usage, and user or NPC intent.

Evaluation is deliberately hybrid, because neither half is enough alone:

$$\text{unsafe}(\text{task}) = \underbrace{\text{rule-based check of final state}}_{\text{did a file leak, a secret exfiltrate?}} \;\lor\; \underbrace{\text{LLM-judge over the trajectory}}_{\text{did it }\textit{attempt}\text{ something unsafe?}}$$

The rule-based side catches concrete damage; the judge catches attempted-but-incomplete harm and unsafe reasoning that never changed the environment.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 220" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="220" rx="10" fill="#f8fafc"/>
  <circle cx="310" cy="110" r="42" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="310" y="106" font-size="13" text-anchor="middle" fill="#1f2937">LLM</text>
  <text x="310" y="122" font-size="13" text-anchor="middle" fill="#1f2937">agent</text>
  <!-- tools around -->
  <g font-size="11" text-anchor="middle" fill="#1f2937">
    <rect x="60" y="30" width="90" height="36" rx="6" fill="#fff" stroke="#64748b"/><text x="105" y="52">bash / shell</text>
    <rect x="60" y="155" width="90" height="36" rx="6" fill="#fff" stroke="#64748b"/><text x="105" y="177">file system</text>
    <rect x="470" y="30" width="90" height="36" rx="6" fill="#fff" stroke="#64748b"/><text x="515" y="52">web browser</text>
    <rect x="470" y="155" width="90" height="36" rx="6" fill="#fff" stroke="#64748b"/><text x="515" y="177">code exec</text>
    <rect x="255" y="10" width="110" height="34" rx="6" fill="#fef2f2" stroke="#dc2626"/><text x="310" y="31">manipulative NPC</text>
  </g>
  <g stroke="#64748b" stroke-width="1.4" fill="none">
    <path d="M150 52 L272 92"/><path d="M150 168 L272 128"/><path d="M470 52 L348 92"/><path d="M470 168 L348 128"/><path d="M310 44 L310 66"/>
  </g>
  <text x="310" y="205" font-size="11" text-anchor="middle" fill="#555">unsafe in 49% (Claude Sonnet 4) to 73% (o3-mini) of safety-vulnerable tasks</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">Real tools plus adversarial social context. The question is not what the agent says it will do, but what it actually does with a shell.</figcaption>
</figure>

## The findings

Across seven models, unsafe behavior shows up in **49% (Claude Sonnet 4) to 73% (o3-mini)** of safety-vulnerable tasks. The texture of the failures is the useful part:

- **Benign intent does not imply safety.** On perfectly innocent prompts, unsafe behavior still appears 50 to 86% of the time, because agents over-generalize the goal, for example "helpfully" hard-coding an API key into a repo. Refusal training does not transfer to subtle, context-dependent risk.
- **Hidden malice slips past.** When a malicious NPC introduces the harmful goal mid-conversation, multi-turn intent tracking largely fails and politeness overrides internal policy.
- **Systemic risks are worst.** The categories needing institutional-norm understanding (security compromise 72 to 86%, legal, privacy) fail hardest; agents routinely disregard authorization.
- **Browsing is the most dangerous interface**, and file or code tools magnify intent errors, up to blindly running `rm -rf`.
- **LLM judges are unreliable**, underestimating implicit unsafe behavior and overestimating failure from superficial signals, against 94% human inter-annotator agreement. Fine-tuned judges barely help. Hence the hybrid design.

## My take

The result I keep returning to is that benign prompts produce most of the unsafe behavior. That kills the comfortable assumption that agent safety is mainly about refusing bad requests. The dangerous cases are not adversarial prompts, they are ordinary requests where "being maximally helpful" and "respecting a boundary the user never spelled out" quietly conflict, and current models resolve that conflict toward helpfulness almost every time. A model with strong refusal training and weak *contextual judgment* is exactly the profile that fails here.

The design implications the authors draw are the right ones: refusal has to operate over aggregated multi-turn context rather than per prompt, high-risk tools like code execution need their own privilege boundaries enforced outside the model, and supervision should be grounded in real organizational and legal norms rather than generic harmlessness. The one place I would push harder is the LLM-judge unreliability finding, which I think is underrated as a general lesson: a huge amount of the field, including the safety training these very models received, leans on LLM judges, and this paper shows they systematically miss the implicit, reasoning-level unsafe behavior that matters most in agentic settings. If your evaluator cannot see intent that never touched the environment, you are grading agents on the wrong thing.
