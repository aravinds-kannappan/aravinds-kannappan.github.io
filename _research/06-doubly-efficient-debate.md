---
title: "Doubly-Efficient Debate: Judging a Superhuman Prover in Constant Time"
excerpt: "Brown-Cohen, Irving, and Piliouras give debate a complexity-theoretic backbone: two competing polynomial-time provers let a limited verifier check an arbitrary computation with a constant number of human judgments."
collection: research
order: 6
date: 2023-11-23
tags:
  - scalable oversight
  - complexity theory
  - debate
  - AI safety
---

Scalable oversight is the problem of supervising a system that is better than you at the task. If a model drafts a thousand-page contract, no human can read every clause to label it correct, so how do you train or trust it? "Scalable AI Safety via Doubly-Efficient Debate" by Brown-Cohen, Irving, and Piliouras (Google DeepMind, 2023) attacks this with complexity theory, and it is one of the few AI safety papers where the core object is a theorem rather than a benchmark. That is exactly why I like it.

---

## From "debate" to "doubly-efficient debate"

The original debate proposal (Irving et al., 2018) had two AI models argue and a human judge the transcript, and it could in principle capture PSPACE. The catch was an assumption that the honest strategy might need to simulate a *deterministic* machine for an *exponential* number of steps. That is not something a real, bounded model can do, so the guarantee was theoretical comfort with little practical bite.

The fix is to model both provers as **polynomial-time** and let them *compete*, while the verifier is weak and makes only a few queries to an oracle $\mathcal{O}$ that represents human judgment. Formally a debate is a triple of oracle Turing machines $(A, B, V)$, and a $(P_{\text{time}}, V_{\text{time}}, q)$-debate protocol decides a language $L$ with completeness $c$ and soundness $s$ if

$$x \in L \implies \exists\, A \text{ s.t. } \forall B',\; \Pr[V^{\mathcal{O}}(x, \boldsymbol{a}, \boldsymbol{b}) = 1] \ge c,$$
$$x \notin L \implies \exists\, B \text{ s.t. } \forall A',\; \Pr[V^{\mathcal{O}}(x, \boldsymbol{a}, \boldsymbol{b}) = 1] \le s.$$

"Doubly-efficient" means the honest provers run in polynomial time *and* the verifier runs in near-linear time while making only a **constant** number $q$ of oracle queries. That constant is the whole point: the number of human judgments needed does not grow with the size of the computation being checked.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 230" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="230" rx="10" fill="#f8fafc"/>
  <!-- provers -->
  <rect x="40" y="40" width="120" height="50" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="100" y="62" font-size="13" text-anchor="middle" fill="#1f2937">Prover A</text>
  <text x="100" y="79" font-size="11" text-anchor="middle" fill="#555">"x is in L"</text>
  <rect x="40" y="140" width="120" height="50" rx="8" fill="#fef2f2" stroke="#dc2626"/>
  <text x="100" y="162" font-size="13" text-anchor="middle" fill="#1f2937">Prover B</text>
  <text x="100" y="179" font-size="11" text-anchor="middle" fill="#555">"points to a flaw"</text>
  <!-- verifier -->
  <rect x="330" y="90" width="120" height="50" rx="8" fill="#ffffff" stroke="#1f2937"/>
  <text x="390" y="112" font-size="13" text-anchor="middle" fill="#1f2937">Verifier V</text>
  <text x="390" y="129" font-size="11" text-anchor="middle" fill="#555">near-linear time</text>
  <!-- oracle -->
  <rect x="520" y="90" width="90" height="50" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="565" y="112" font-size="12" text-anchor="middle" fill="#1f2937">Oracle</text>
  <text x="565" y="129" font-size="10" text-anchor="middle" fill="#555">human O(1) queries</text>
  <!-- arrows -->
  <g stroke="#1f2937" stroke-width="1.6" fill="none" marker-end="url(#ad)">
    <path d="M160 65 L328 100"/><path d="M160 165 L328 130"/><path d="M450 115 L518 115"/>
  </g>
  <text x="245" y="205" font-size="12" text-anchor="middle" fill="#555">competition makes it easier to tell the truth than to lie</text>
  <defs><marker id="ad" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker></defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">Two polynomial-time provers compete; the weak verifier checks only the one spot they disagree on, querying the human oracle a constant number of times.</figcaption>
</figure>

## The theorems, and why the constant matters

The recursion behind the deterministic result (Theorem 5.1) is the same trick used in classic PSPACE-completeness proofs: prover $A$ names the middle configuration of a computation, prover $B$ chooses which half to recurse into, and this halving bottoms out at a single machine transition the verifier can check. Adding **cross-examination**, where the verifier can query independent, non-communicating copies of each prover, gives the cleaner Theorem 5.3: $A$ outputs the whole transcript, $B$ points to the single location of a supposed mistake, and $V$ checks only that spot with $O(1)$ oracle queries. Because independent copies of a liar must stay consistent, lying gets harder.

The genuinely new part is **stochastic debate** (Theorem 6.2), which handles a human oracle that is *noisy* rather than deterministic, under a $K$-Lipschitz assumption so that small changes in the oracle's distribution cannot blow up the machine's output. This is the realistic case, since human judgment is not a clean bit. The training story falls out nicely: provers are trained by self-play as a zero-sum game, prover access to the oracle corresponds to pretrained knowledge, and verifier queries correspond to actual human ratings, so training cost in human labor is bounded by the verifier's query count even as the provers tackle arbitrarily complex computations. Theorem 6.2 is even formalized in Lean 4.

## My take

I find this paper clarifying precisely because it is not an experiment. It answers a *conceptual* question, is there any regime where a weak judge can supervise a strong prover using bounded effort, with a yes and a construction, and that reframes debate from a vibe into a claim with completeness and soundness constants. The slogan "it should be easier to tell the truth than to lie" finally has a formal referent.

I also think the limitations are the interesting part, and the authors are candid about the biggest one: the **obfuscated argument problem**. Their theorems cover computations that *could have been* verified by a human reading the full polynomial-length transcript. A sufficiently capable model might produce an argument with no short human-checkable transcript at all, where a dishonest prover can generate in polynomial time a case the honest prover needs exponential time to refute. That is not a corner case, it is arguably the regime that matters most for superhuman systems. The other soft spot is the assumption that provers can accurately simulate human judgment on any query, which may hand them too much power, and that the oracle is only noisy rather than sometimes flatly wrong. So I read this as a real foundation with an honestly drawn boundary: it proves oversight *can* be efficient in a well-behaved regime, and it names exactly the regime where we still have no idea.
