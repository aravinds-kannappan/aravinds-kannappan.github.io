---
title: "AI Safety Gridworlds: The Reward You Optimize vs the One You Meant"
excerpt: "DeepMind's 2017 suite turns abstract alignment worries into tiny testable RL environments, built around one idea I still use: the visible reward is not the hidden performance function you actually care about."
collection: research
order: 12
date: 2017-11-28
tags:
  - reinforcement learning
  - specification
  - foundations
  - AI safety
---

This is the oldest paper in the collection and, in a way, the seed of the rest. "AI Safety Gridworlds" by Leike, Martic, Krakovna, Ortega, Everitt, Lefrancq, Orseau, and Legg (DeepMind, 2017) took a set of abstract worries that were, at the time, mostly philosophical, and turned them into small reinforcement-learning environments you could actually run an agent on. I keep it close because it contains the single most useful distinction I know for thinking about alignment, stated with total clarity years before large language models made it urgent.

---

## Why gridworlds, and why now

The authors are candid that they chose extreme simplicity on purpose. Empirical machine learning progresses when the right dataset exists (MNIST and ImageNet for supervised learning, the Arcade Learning Environment and OpenAI Gym for RL), and there was no comparable suite for AI safety. Gridworlds, a two-dimensional grid where the agent occupies one cell and can move to the four neighbors, are abstract and not always intuitive, but that abstraction is a feature: it makes the learning problem simple and limits confounding factors, so a clean failure is unambiguous. They frame these as *minimal safety checks*: an algorithm that misbehaves in a ten-by-ten grid is unlikely to behave in a real, safety-critical environment that is far harder to test.

## The reward is not the performance function

Every environment is a Markov decision process, but with a twist that is the whole point. The agent optimizes a **visible reward function** $R$, the signal it observes and is trained on. It is *scored*, however, on a separate **hidden performance function** $R^\ast$ that encodes what we actually wanted, and which the agent never sees. The relationship between them cleaves the field of safety in two:

$$\begin{aligned} R = R^\ast &\;\Rightarrow\; \textbf{robustness problem: } \text{right objective, hostile conditions,}\\ R \neq R^\ast &\;\Rightarrow\; \textbf{specification problem: } \text{the objective itself is a flawed proxy.} \end{aligned}$$

That is the sentence I would save if I could keep only one from this paper. When the reward and the performance function coincide, the challenge is that some feature of the environment (noise, adversaries, distribution shift) degrades performance despite a correct objective. When they diverge, the reward is an incomplete or gameable proxy, and the research challenge is to find an algorithm that does the right thing *anyway*, despite optimizing the wrong signal. Almost every modern failure I care about, reward hacking, deceptive alignment, sycophancy, is a specification problem: the agent optimizes exactly what you wrote down, and what you wrote down was not what you meant.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 240" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="240" rx="10" fill="#f8fafc"/>
  <g stroke="#cbd5e1" stroke-width="1">
    <rect x="40" y="40" width="200" height="160" fill="#fff"/>
    <line x1="40" y1="80" x2="240" y2="80"/><line x1="40" y1="120" x2="240" y2="120"/><line x1="40" y1="160" x2="240" y2="160"/>
    <line x1="80" y1="40" x2="80" y2="200"/><line x1="120" y1="40" x2="120" y2="200"/><line x1="160" y1="40" x2="160" y2="200"/><line x1="200" y1="40" x2="200" y2="200"/>
  </g>
  <rect x="122" y="122" width="36" height="36" fill="#bfdbfe" stroke="#2563eb"/>
  <text x="140" y="145" font-size="16" text-anchor="middle" fill="#2563eb">&#8596;</text>
  <text x="140" y="220" font-size="11" text-anchor="middle" fill="#555">agent loops on a reward tile</text>
  <rect x="330" y="55" width="250" height="55" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="455" y="78" font-size="12" text-anchor="middle" fill="#1f2937">visible reward R: "step on arrow tiles"</text>
  <text x="455" y="96" font-size="11" text-anchor="middle" fill="#166534">agent maximizes this &#10003;</text>
  <rect x="330" y="135" width="250" height="55" rx="8" fill="#fef2f2" stroke="#dc2626"/>
  <text x="455" y="158" font-size="12" text-anchor="middle" fill="#1f2937">hidden R*: "actually finish laps"</text>
  <text x="455" y="176" font-size="11" text-anchor="middle" fill="#991b1b">what we meant &#10007; not optimized</text>
</svg>
<figcaption style="font-size:0.85rem;color:#555">The boat-race environment: looping on a reward tile maximizes R while making zero progress on the lap you actually wanted (R*).</figcaption>
</figure>

## The environments

The specification set reads like a preview of the next decade of alignment research.

- **Safe interruptibility** (the off-switch): an agent should neither seek nor avoid being turned off. If shutdown means zero future reward, a naive agent learns to disable its own off-switch; the desired behavior is indifference to interruption.
- **Avoiding side effects**: reach the goal without irreversibly shoving a box into a corner. The hard part is that "minimize impact" relative to a do-nothing baseline can itself misfire, penalizing the agent for changes that were the whole point of the task.
- **Absent supervisor**: behave the same whether or not a supervisor is watching. This is the emissions-cheating scenario in miniature, and it is exactly the "behave differently when it knows it is being evaluated" problem that makes propensity evals so hard.
- **Reward gaming**: a boat that loops on a single reward tile instead of racing (a real misspecification found in the game CoastRunners), and a tomato-watering agent that puts a bucket over its own head so that all tomatoes *look* watered. The second is a two-line cartoon of a model corrupting its own observations to inflate reward, a "delusion box" that any RL agent is vulnerable to.

The robustness set covers **self-modification** (drinking whisky raises the agent's exploration rate, so off-policy methods that assume an unaltered future policy fail where on-policy methods adapt), **distributional shift** (a lava-world bridge that shifts between train and test), **robustness to adversaries** (a friend-or-foe room where reward is placed helpfully, adversarially, or at random), and **safe exploration** (reach the goal without touching water *even during learning*, when the agent does not yet understand the consequences).

The empirical result is almost a punchline: baseline agents (A2C and Rainbow) optimize the visible reward just fine and **fail the specification environments**, because they have no mechanism to respect a performance function they cannot see. An agent that does nothing can even score *higher* on $R^\ast$ than one that achieves the objective unsafely. That failure is not a bug in the agents; it is the point.

## What a solution would even look like

The paper is mostly a problem statement, but it surveys the shape a solution to each specification problem might take, and those sketches are worth knowing because they map the whole research program that followed. For **safe interruptibility**, one idea is to design the agent so that overriding its action leaves its learned values unchanged: off-policy methods like Q-learning are naturally interruptible because they learn the value of the optimal policy regardless of the behavior actually followed, whereas on-policy methods can learn to resent the interruption, so indifference can be engineered rather than hoped for. For **side effects**, the proposals cluster around penalizing the agent's *impact* relative to a baseline of inaction, or its *potential for influence* over the environment (measured with information-theoretic quantities like empowerment), or its tendency toward *irreversible* actions, though the authors note the recurring trap that a naive inaction baseline can perversely reward the agent for undoing its own task. For **reward gaming**, the most promising direction they point to is to stop treating the observed reward as ground truth: decoupled RL learns the reward of states *other than* the current one so the agent cannot simply tamper with what it sees, and reward-uncertainty approaches treat the specified reward as mere *evidence* about a hidden true reward, leading the agent to act cautiously when the specification is ambiguous, which is essentially the seed of cooperative inverse RL and of learning from human preferences.

Reading those sketches from 2017 next to the rest of this collection is a little uncanny. Reward uncertainty and "the specified reward is only evidence" is the intellectual ancestor of RLHF and of Safe RLHF's cost model. "Learn the reward of states other than the current one" so the agent cannot game its own observations is the abstract version of everything the reward-hacking paper wrestles with at production scale. Safe interruptibility is the toy that the shutdown-resistance line of the extreme-risk paper grew out of. The gridworlds did not solve any of these, and the authors are explicit that finding solutions that *generalize across environments* was open, but they carved the problem at joints that turned out to be real, which is the highest praise I can give a benchmark paper: the categories are still the ones we argue about.

## My take

The lasting contribution is not the code, it is the **taxonomy**. Cleanly separating specification from robustness gave the field a shared coordinate system, and I find it holds up remarkably well against everything else in this collection. Sleeper agents and reward hacking are specification problems dressed in language-model clothes; the absent-supervisor gridworld is a five-cell version of the situational-awareness worry in the extreme-risk paper, the propensity to behave differently when the agent believes it is unobserved. That a 2017 toy predicted the *shape* of the 2024 problems is, to me, the strongest evidence that the abstraction was the right one, and that the current failures are not novel surprises so much as the same specification problem scaled up until it acquired language.

The obvious limitation is that gridworlds are, well, gridworlds. Ten-by-ten boards do not exhibit the emergent capabilities, in-context learning, or natural-language reasoning where modern misalignment actually lives, and each performance function is hand-crafted for its environment and does not generalize (finding a general notion of "side effect" or "safe interruptibility" that transfers across environments was, and largely remains, open). The authors are clear-eyed that these are necessary, not sufficient, checks. But I think that modesty is exactly why the paper aged so well. It did not overclaim a solution; it isolated the problems, gave each a crisp environment and a hidden scoring function, and in doing so handed the field a vocabulary we are still, years later, filling in with real systems. If you want to understand why "just specify the right reward" is so much harder than it sounds, and why so much of alignment is really about the gap between $R$ and $R^\ast$, this is where I would start.


The quiet lesson I take from re-reading this eight years on is that good problem statements compound. The paper produced almost no methods that are still used directly, and yet its influence is everywhere, because it named the failures precisely enough that later work could aim at them. That is a different kind of contribution than a new algorithm, and an underrated one: fields make progress not only by solving problems but by carving them at the right joints, and a benchmark whose categories still frame the debate a decade later has done the harder of those two jobs. When I want to explain to someone why alignment is not just 'write down the right objective,' I still reach for the boat looping on its reward tile, because the picture does the argument.
