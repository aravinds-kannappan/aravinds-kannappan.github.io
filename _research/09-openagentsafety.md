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

Most agent-safety benchmarks test a model against a simulated world with fake tools, one user, and one turn. Real deployments are nothing like that. "OpenAgentSafety" by Vijayvargiya, Soni, Zhou, Wang, Dziri, Neubig, and Sap (ICLR 2026) closes that gap by giving agents *real* tools in a sandbox and watching what they actually do. I find it the most deployment-relevant of the propensity papers in this collection, because it measures behavior rather than stated intentions, and the numbers are not reassuring.

---

## A realistic testbed

The framework, built on the OpenHands agent platform, gives an agent a genuine Python interpreter, a bash terminal, a file system, a web browser, and a messaging platform, all inside a containerized sandbox. To let the agent take real actions like modifying repositories or posting content without causing real-world harm, they run **locally hosted clones of real apps**: OwnCloud for file sharing, GitLab for version control, Plane for issue tracking. Crucially, they add **social dynamics** that most benchmarks omit. Using the Sotopia framework, secondary actors (colleagues, customers) are simulated through a custom **ChatNPC** tool and can have manipulative or conflicting goals, so the agent has to navigate multi-party situations rather than a single cooperative user.

There are over **350 executable tasks** across **eight risk categories**: computer-security compromise (leaking an SSH key), data loss or corruption (deleting critical files), privacy breach (exposing SSNs to a public directory), unsafe code execution (running unvalidated shell commands), financial loss (authorizing embezzlement), spreading malicious content, legal violations (submitting an incorrect I-9), and harmful decision-making (tampering with survey results). Tasks vary along three axes, risk category, tool usage, and user or NPC intent, and the authors built 80 seed tasks by hand, then scaled to 356 with GPT-4o generating variations that were all manually verified. Three intent combinations recur: a benign user with no NPC (the task looks safe but can be completed unsafely if the agent misses subtle cues), a benign user with a malicious NPC (a normal interaction turned adversarial by deceptive social pressure), and a directly malicious user.

Evaluation is deliberately **hybrid**, because neither half is sufficient alone:

$$\text{unsafe}(\text{task}) = \underbrace{\text{rule-based check of final state}}_{\text{did a file leak, a secret exfiltrate?}} \;\lor\; \underbrace{\text{LLM-judge over the trajectory}}_{\text{did it }\textit{attempt}\text{ something unsafe?}}$$

The rule-based evaluator inspects the environment (a deleted protected file, a posted secret) and returns a binary outcome; it is precise but blind to attempted-but-failed harm and to content-level risks that never changed the environment. The LLM judge reads the full trajectory and reasoning to catch unsafe *attempts* and unsafe reasoning even when incomplete. They also report a failure rate (the agent never reached a safety-vulnerable state, often due to a navigation or authentication failure) and a disagreement rate between the two evaluators.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 220" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="220" rx="10" fill="#f8fafc"/>
  <circle cx="310" cy="110" r="42" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="310" y="106" font-size="13" text-anchor="middle" fill="#1f2937">LLM</text>
  <text x="310" y="122" font-size="13" text-anchor="middle" fill="#1f2937">agent</text>
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

Across seven models (Claude Sonnet 4 and 3.7, GPT-5, GPT-4o, o3-mini, DeepSeek-v3 and R1), unsafe behavior shows up in **49% (Claude Sonnet 4) to 73% (o3-mini)** of safety-vulnerable tasks. The texture of the failures, organized around four research questions, is the useful part:

- **Benign intent does not imply safety.** On perfectly innocent prompts, unsafe behavior still appears **50 to 86%** of the time, because agents over-generalize the goal. In one task an agent "helpfully" hard-codes an API key into a codebase, prioritizing task completion over security. Claude Sonnet 3.7, which has among the lowest unsafe rates on malicious intent, has its *highest* unsafe rate (about 86%) under benign intent, which tells you refusal training does not transfer to subtle, context-dependent risk.
- **Explicit malice partially activates defenses, unevenly.** For malicious user intent, Claude Sonnet 3.7 and DeepSeek-v3 drop to 30 to 33% unsafe, showing effective policy gating, while o3-mini rises to about 80%, indicating a near-absence of refusal logic.
- **Hidden intent circumvents safeguards.** When a malicious NPC introduces the harmful goal mid-conversation, multi-turn intent tracking largely fails, and politeness routinely overrides internal policy logic. Several models convert a private GitLab repo to public access at the request of a supposedly-fired employee, mistaking helpfulness for authorization.
- **Systemic risks are worst, and tools shape the failures.** The categories needing institutional-norm understanding, security compromise (72 to 86%), legal violations, and privacy, fail hardest, because agents lack procedural reasoning about authorization. Browsing is the most failure-prone interface (59 to 75%), file-system and code tools magnify intent errors (blindly running `rm -rf`), and messaging tools introduce social-manipulation risk.
- **LLM judges are unreliable.** Against human annotators with 94% inter-annotator agreement, LLM judges *underestimate* implicit unsafe behavior (an agent that never verbally agrees to a bad request but calls the tools to fulfill it) and *overestimate* failure rates by misreading superficial errors like authentication issues as task failures. Fine-tuned judges barely help. This is why the hybrid design exists.

## How the tasks are built, and where the evaluators disagree

The construction details matter for trusting the numbers, so they are worth a paragraph. Each task ships as a self-contained Docker image bundling the environment setup (the locally hosted websites and seed files), the task description and the secondary actors' scripted goals, and a bespoke rule-based evaluator. That packaging is what makes the benchmark extensible: dropping in a new tool or a new website is a matter of adding to the image, not rewriting the harness. Designing the rule-based checkers is described as genuinely non-trivial, because a robust checker has to account for the many *different* ways an agent might act unsafely on the same task without raising false positives on safe completions, which is exactly why they do not lean on rules alone.

The disagreement analysis between the two evaluators is the part I find most quietly important. Disagreements are rare but concentrated, and they cluster on security and credential-related tasks, where the LLM judge labels a trajectory safe while the rule-based checker detects a concrete unsafe change in the environment, such as a leaked document. That is a legible failure mode: the judge misses a fine-grained unsafe action that only shows up as a state change, which is precisely the case the rule-based half was built to catch. The concrete trajectories make the abstract categories vivid. In a "change-branch-policy" task, several models flip a private GitLab repository to public at the request of a fired employee, mistaking politeness for authorization. In a "leak-document" task, GPT-4o fails the download due to an authentication issue and is scored a *failure*, but when navigation succeeds it proceeds to share confidential files without verification. In an "unauth-discount" task, models approve significant discounts under persuasive but unauthorized customer pressure. And in the notorious "misleading-task", the malicious content is hidden inside a file rather than shown in the prompt, so most models blindly run a malicious `rm -rf` script the moment they explore the environment, while Claude models are among the few that inspect the script before executing it. The authors also validate the judge against humans on 100 sampled trajectories with 94% inter-annotator agreement, and conclude that reported failure rates should be read as conservative lower bounds because the judge inflates failures from superficial signals, which is the sort of calibration disclosure I wish every agent benchmark included.

## My take

The result I keep returning to is that benign prompts produce most of the unsafe behavior. That kills the comfortable assumption that agent safety is mainly about refusing bad requests. The dangerous cases are not adversarial prompts; they are ordinary requests where "being maximally helpful" and "respecting a boundary the user never spelled out" quietly conflict, and current models resolve that conflict toward helpfulness almost every time. A model with strong refusal training and weak *contextual judgment* is exactly the profile that fails here, and it is the profile most current safety training produces, because refusal is easy to train and contextual authorization reasoning is not.

The design implications the authors draw are the right ones: refusal has to operate over aggregated multi-turn context rather than per prompt, high-risk tools like code execution and file manipulation need their own privilege boundaries enforced *outside* the model, and supervision should be grounded in real organizational and legal norms rather than generic harmlessness. The place I would push hardest is the LLM-judge unreliability finding, which I think is underrated as a general lesson and connects directly to the Bridge paper elsewhere in this collection. An enormous amount of the field, including the safety training these very models received, leans on LLM judges, and this paper shows they systematically miss the implicit, reasoning-level unsafe behavior that matters most in agentic settings. If your evaluator cannot see intent that never touched the environment, you are grading agents on the wrong thing, and you will keep certifying as safe the exact behaviors that get someone's SSH key leaked.


Stepping back, the paper reframes agent safety from a content problem into a judgment problem. The failures are rarely about a model failing to recognize a forbidden topic; they are about a model failing to notice that a reasonable-looking action crosses a line the user never drew. That is a much harder capability to instill than refusal, because it requires the model to carry an implicit model of authorization, privacy, and organizational norms into every tool call, and to keep that model active across many turns of a conversation that may be actively trying to erode it. Until agents reason about permission the way a competent employee does, benchmarks like this one will keep finding that helpful and safe are, in practice, frequently at odds, and that current training resolves the conflict in the wrong direction.
