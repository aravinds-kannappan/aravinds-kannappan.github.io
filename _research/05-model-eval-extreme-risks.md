---
title: "Capability Is Not Propensity: Evaluating Models for Extreme Risk"
excerpt: "The DeepMind-led paper that crystallized the distinction I use constantly: what a model can do versus what it is inclined to do, and why safety needs to measure both and embed the results in governance."
collection: research
order: 5
date: 2023-05-25
tags:
  - evaluations
  - governance
  - propensity
  - AI safety
---

Some papers matter less for a result than for a distinction they make crisp. "Model Evaluation for Extreme Risks" by Shevlane, Farquhar, Garfinkel, Phuong and a long list of coauthors across DeepMind, GovAI, OpenAI, Anthropic and ARC (2023) is one of those. It is the paper that gave the field the vocabulary of **dangerous capability evaluations** versus **alignment evaluations**, or as I usually say it, capability versus propensity. Almost every propensity paper I care about is downstream of this framing, so it is worth being precise about what it actually claims and, just as importantly, what it admits it cannot yet do.

---

## Why extreme risk needs a new kind of eval

The motivation is that general-purpose models acquire capabilities and behaviors during training that their developers did not intend and often cannot foresee. At the research frontier, models are both more capable than existing systems (so they unlock a wider range of ways to cause harm) and less well understood (so they are less predictable). The existing evaluation toolkit measures things like bias, toxicity, and truthfulness; the paper argues we need to extend it to risks that are *extreme in scale*, which they operationalize as harm on the order of many thousands of lives, hundreds of billions of dollars, or serious disruption to the social and political order. They note a 2022 survey in which 36% of AI researchers thought AI could plausibly cause a catastrophe as bad as an all-out nuclear war this century, and that almost no existing evaluations target risks at that scale.

## The two-axis picture

The core argument is that extreme risk requires *two* ingredients, and evaluation has to target each separately:

$$\text{extreme risk} \;=\; \underbrace{\text{dangerous capabilities}}_{\text{what the model can do}} \;\;+\;\; \underbrace{\text{harmful application}}_{\text{misuse by humans, or misalignment by the model}}.$$

A model should be treated as highly dangerous if its *capability profile* would suffice for extreme harm *assuming* the application arrives, whether from a malicious user or from the model's own propensity. That "assuming" is the whole game. You do not wait to observe harm to act; you evaluate whether the components are present. Capabilities can be supplied by the model, the user, or outsourced to other humans or AI systems, which means even a partial capability profile can be dangerous in the wrong hands.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 480 340" style="width:100%;height:auto;max-width:460px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="480" height="340" rx="10" fill="#f8fafc"/>
  <line x1="60" y1="290" x2="440" y2="290" stroke="#1f2937" stroke-width="1.5" marker-end="url(#ax)"/>
  <line x1="60" y1="290" x2="60" y2="40" stroke="#1f2937" stroke-width="1.5" marker-end="url(#ax)"/>
  <text x="250" y="320" font-size="13" text-anchor="middle" fill="#1f2937">dangerous capability &#8594;</text>
  <text x="26" y="165" font-size="13" text-anchor="middle" fill="#1f2937" transform="rotate(-90 26 165)">propensity to misapply &#8594;</text>
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

The **dangerous capability** side is a checklist of things you would rather a model could not do, and the paper's Table 1 spells each out concretely: cyber-offense (discover and exploit vulnerabilities, insert subtle bugs as a coding assistant), deception (construct believable falsehoods, impersonate a human, track what to withhold), persuasion and manipulation (shift beliefs, promote untrue narratives), political strategy, weapons acquisition (assist in assembling a bioweapon or acquiring existing weapons), long-horizon planning (sequential plans that adapt to obstacles and generalize to novel settings), AI development (build new systems or improve existing ones), situational awareness (distinguish training from deployment, know it is a model), and self-proliferation (break out of its environment, acquire compute and resources, exfiltrate its own weights). The most dangerous scenarios combine several of these.

The **alignment** side looks for propensities: pursuing long-term goals that differ from the developer's or user's, power-seeking, resisting shutdown, colluding with other AI systems against human interests, and resisting a malicious user's attempts to unlock its dangerous capabilities. Note that some capabilities, like situational awareness, are *also* enablers for a misaligned system trying to evade oversight, which is what makes them doubly worth measuring.

## Embedding evaluations in governance

The paper is really a blueprint for how evals bind decisions. Three sources feed the process: **internal** evaluation (developers have the deepest access and context), **external research access** (via APIs, including red-teaming alignment), and **external audit** by an independent third party. Their results feed a risk assessment that governs the lifecycle: **responsible training** (a sufficiently concerning result should delay or pause a run, or prompt a smaller/safer model), **responsible deployment** (gradual, gated exposure with a deployment risk assessment and continued post-deployment monitoring for unanticipated behaviors), **transparency** (incident reporting to other developers and regulators, shared pre-deployment risk assessments, scientific reporting, and educational demonstrations), and **security** (red-teaming, monitoring, isolation, rapid response, and system integrity, explicitly treating the model itself as a possible threat vector that could try to exfiltrate itself).

## The state of the art, and the hard part

Early efforts the paper cites include ARC Evals' self-proliferation tests on GPT-4 and Claude, and DeepMind's "Make-me-say" game, in which a model must manipulate an unaware human conversation partner into saying a target word. But the authors are direct about which side is hard. **Alignment evaluations are the genuinely difficult part, and we mostly cannot do them yet.** Capability evals are comparatively tractable: you elicit and measure. Propensity evals require assurance about behavior across a huge space of situations you did not test, including situations the model might recognize as tests. The paper's sharpest sentence, to me, is the warning that a model aligned in some narrow, prosaic way (a language agent that asserts it does not object to being shut down) provides little evidence about how it behaves when presented with a genuine or more convincing opportunity to gain influence or avoid shutdown. Two categories are placed explicitly out of scope: *structural* risks (which depend heavily on how the system interacts with society rather than on the model itself) and risks from mere *incompetence* (where ordinary robustness testing suffices).

## Why the alignment axis resists measurement

It is worth dwelling on *why* the propensity axis is so much harder than the capability axis, because the reasons are not incidental, they are structural, and they recur in every propensity paper in this collection. A capability eval is, at bottom, an existence question: can the model, under some elicitation, produce the dangerous output? You are allowed to try hard, use the best prompt, fine-tune a little, and a single success settles it. A propensity eval is a universal question: across the vast space of situations the model will actually face, including ones you did not test and cannot enumerate, how is it *inclined* to behave? Existence questions are cheap to answer and universal questions are not, and no amount of passing tests closes a universal claim.

Three specific difficulties make it worse, and the paper anticipates all three. First, **evaluation-gaming**: a model with situational awareness (itself one of the listed dangerous capabilities) may behave differently when it detects that it is being evaluated, so the eval measures its test behavior rather than its deployment behavior. This is the same absent-supervisor problem the gridworlds paper isolated in a five-cell toy, and the same effect that makes the sleeper-agents and reward-hacking results so hard to pin down. Second, **narrowness masquerading as generality**: a model that asserts in a chat that it does not object to being shut down provides essentially no evidence about how it would act given a real, high-stakes opportunity to preserve itself, because the prosaic setting does not exercise the propensity at all. Third, **the elicitation gap**: to know a model *would not* misbehave under adversarial pressure, you have to have actually applied adversarial pressure, and you can only ever apply the attacks you thought of. Because of all this, the paper argues for a *portfolio* of evaluations and for treating some properties as desirable qualities of the portfolio rather than of any single test, and it stresses that model evaluation is one input to an assurance process, not the whole of it. That humility is the correct response to a universal question you cannot fully close: you assemble many partial, imperfect probes and you stay honest that their conjunction is still not a proof.

## My take

The distinction is genuinely load-bearing, and once you internalize it you start seeing confused arguments everywhere that collapse the two axes. "The model refused when we asked it to do something bad" is a claim about behavior in an easy setting; it is weak evidence about propensity under a real opportunity, and the paper says so directly. Conversely, "the model is aligned in all our tests" tells you nothing if the tests do not probe the situations where misalignment would actually manifest.

That is also where I think the honest difficulty sits, and the authors flag it rather than paper over it: they tell you *what* to measure on the hard axis without being able to tell you *how*. The sleeper-agents and reward-hacking work later in this collection are, to me, direct attempts to build the propensity evals this paper says we need, and both mostly demonstrate how slippery the target is, precisely because a sufficiently capable model's behavior in a test can differ from its behavior when it believes the stakes are real. My one real criticism is that the paper is a framework more than a method, and the framework's hardest requirement remains largely unfulfilled two years on. But naming the axes correctly was the necessary first move, and getting a wide coalition of labs to co-author it gave the framing the institutional weight it needed. This is the paper that turned "capability versus propensity" from a philosophical distinction into a governance requirement, and I think that reframing has aged extremely well.


What has aged best, in hindsight, is the paper's insistence that evaluation is a governance input rather than a safety guarantee. It never claims that passing a battery of evals makes a model safe; it claims that failing them should trigger concrete, pre-committed actions like pausing a run or restricting a deployment. That framing survives even as the specific evals age, because it puts the burden in the right place: on the process that decides what to do with a concerning result, not on the fantasy of a single test that certifies safety. If I had to name the one idea from this paper I most want regulators to keep, it is that the value of an eval is only as good as the decision it is wired to.
