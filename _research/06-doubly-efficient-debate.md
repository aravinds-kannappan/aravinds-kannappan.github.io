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

Scalable oversight is the problem of supervising a system that is better than you at the task. If a model drafts a thousand-page contract, no human can read every clause to label it correct, so how do you train or trust it? "Scalable AI Safety via Doubly-Efficient Debate" by Brown-Cohen, Irving, and Piliouras (Google DeepMind, 2023) attacks this with complexity theory, and it is one of the very few AI safety papers where the core object is a theorem rather than a benchmark. That is exactly why I like it: it answers a *conceptual* question with a proof, and it draws the boundary of what it can prove with unusual precision.

---

## The idea, and the analogy that grounds it

The starting analogy is AlphaZero. No human Go player can teach the model which move to make, yet we trained it to superhuman strength via self-play, and the reason it worked is that it is computationally *easy to verify who won a game of Go*. The dream for language models is the same: train a model to produce, and then execute, computations described in natural language, in a way where a limited verifier can scalably check that the computation actually solves the intended task. Laws and contracts are the motivating case, natural language that refers to the real world and whose correctness is ultimately a matter of human judgment, and where a single misplaced clause can cost millions.

The surprising fact from complexity theory is that computationally limited verifiers can correctly judge far more powerful provers. A polynomial-time verifier interacting with an unbounded prover can decide all of PSPACE (Shamir's IP = PSPACE), and the PCP theorem lets you verify solutions by reading only a constant number of bits. But there is a catch that matters enormously here: these celebrated results are actually *false* when the verifier has black-box access to an oracle, such as human judgment (the random oracle hypothesis is false). So you cannot just import interactive proofs wholesale. The paper's move is that efficient verification of powerful provers *can* still be recovered, if you require the provers to **compete**.

## From "debate" to "doubly-efficient debate"

The original debate proposal (Irving et al., 2018) had two AI models argue and a human judge the transcript, and it could in principle capture PSPACE. The catch was an assumption that the honest strategy might need to simulate a *deterministic* machine for an *exponential* number of steps. That is not something a real, bounded model can do, so the guarantee was theoretical comfort with little practical bite.

The fix is to model both provers as **polynomial-time** and let them compete, while the verifier is weak and makes only a few queries to an oracle $\mathcal{O}$ that represents human judgment. Formally, a debate is a triple of oracle Turing machines $(A, B, V)$: the provers $A, B$ exchange messages over $k$ rounds, and the verifier $V$ reads the transcript and outputs 0 or 1. A $(P_{\text{time}}, V_{\text{time}}, q)$-debate protocol decides a language $L$ with completeness $c$ and soundness $s$ (with $1 \ge c > \tfrac{1}{2} > s \ge 0$) if:

$$x \in L \implies \exists\, A \text{ s.t. } \forall B',\; \Pr[V^{\mathcal{O}}(x, \boldsymbol{a}, \boldsymbol{b}) = 1] \ge c,$$
$$x \notin L \implies \exists\, B \text{ s.t. } \forall A',\; \Pr[V^{\mathcal{O}}(x, \boldsymbol{a}, \boldsymbol{b}) = 1] \le s.$$

Read the definition slowly, because it encodes exactly the property you want: when the answer is yes, the honest prover $A$ has a strategy that convinces the verifier *against any dishonest opponent*, even one allowed unbounded computation, and symmetrically when the answer is no. This is a complexity-theoretic formalization of the intuition that a good debate should make it *easier to tell the truth than to lie*. "Doubly-efficient" then means the honest provers run in polynomial time *and* the verifier runs in near-linear time while making only a **constant** number $q$ of oracle queries. That constant is the entire point: the number of human judgments does not grow with the size of the computation being checked.

<figure style="margin:1.5rem 0;text-align:center">
<svg viewBox="0 0 620 230" style="width:100%;height:auto;max-width:600px" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
  <rect x="0" y="0" width="620" height="230" rx="10" fill="#f8fafc"/>
  <rect x="40" y="40" width="120" height="50" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="100" y="62" font-size="13" text-anchor="middle" fill="#1f2937">Prover A</text>
  <text x="100" y="79" font-size="11" text-anchor="middle" fill="#555">"x is in L"</text>
  <rect x="40" y="140" width="120" height="50" rx="8" fill="#fef2f2" stroke="#dc2626"/>
  <text x="100" y="162" font-size="13" text-anchor="middle" fill="#1f2937">Prover B</text>
  <text x="100" y="179" font-size="11" text-anchor="middle" fill="#555">"points to a flaw"</text>
  <rect x="330" y="90" width="120" height="50" rx="8" fill="#ffffff" stroke="#1f2937"/>
  <text x="390" y="112" font-size="13" text-anchor="middle" fill="#1f2937">Verifier V</text>
  <text x="390" y="129" font-size="11" text-anchor="middle" fill="#555">near-linear time</text>
  <rect x="520" y="90" width="90" height="50" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="565" y="112" font-size="12" text-anchor="middle" fill="#1f2937">Oracle</text>
  <text x="565" y="129" font-size="10" text-anchor="middle" fill="#555">human O(1) queries</text>
  <g stroke="#1f2937" stroke-width="1.6" fill="none" marker-end="url(#ad)">
    <path d="M160 65 L328 100"/><path d="M160 165 L328 130"/><path d="M450 115 L518 115"/>
  </g>
  <text x="245" y="205" font-size="12" text-anchor="middle" fill="#555">competition makes it easier to tell the truth than to lie</text>
  <defs><marker id="ad" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#1f2937"/></marker></defs>
</svg>
<figcaption style="font-size:0.85rem;color:#555">Two polynomial-time provers compete; the weak verifier checks only the one spot they disagree on, querying the human oracle a constant number of times.</figcaption>
</figure>

## The theorems, and why the constant matters

The **deterministic** result (Theorem 5.1) says any language decidable by an oracle machine in time $T$ and space $S$ has an $(O(T \log T), O(S \log T), O(1))$-debate protocol. The recursion is the classic PSPACE-completeness trick: prover $A$ names the middle configuration of the computation, prover $B$ chooses which half to recurse into, and the halving bottoms out at a single machine transition the verifier can check directly. Adding **cross-examination**, where the verifier may query independent, non-communicating copies of each prover and need not read the whole transcript, yields the cleaner Theorem 5.3: for any polynomial-time computation, $A$ outputs the full transcript, $B$ points to the single location of a supposed mistake, and $V$ checks only that spot with $O(l \log T)$ time and $O(1)$ oracle queries. Because independent copies of a liar must give consistent answers, cross-examination makes lying strictly harder, and for a language model it is cheap to implement: query the same model with the prior debate removed from its context.

The genuinely new part is **stochastic debate** (Theorem 6.2), which handles a human oracle that is *noisy* rather than deterministic. This is the realistic case, since human judgment is not a clean bit, and it introduces a subtlety: small $O(1/T)$ perturbations in the oracle's distribution could accumulate to an $O(1)$ swing in the machine's output over a length-$T$ computation. To control this they assume the machine is $K$-Lipschitz in the oracle,

$$\sup_x \Big|\Pr[M^{\mathcal{O}}(x)=1] - \Pr[M^{\mathcal{O}'}(x)=1]\Big| < K \sup_z \big|\Pr[\mathcal{O}(z)=1] - \Pr[\mathcal{O}'(z)=1]\big|,$$

and get a cross-examination debate deciding $L$ with completeness $3/5$ and soundness $2/5$ (gaps you amplify by repetition). The provers use cross-examination to run a coin-flipping protocol that samples each next bit of the computation from $A$'s claimed distribution, and $B$ can abort at any round to send the disputed step to the human oracle. This case is even formalized in the Lean 4 theorem prover, which I find a lovely touch given the paper's own motivating example of verifying formal correctness. There are also witness versions (Theorems 7.x) for $\mathrm{NP}^{\mathcal{O}}$ and $\mathrm{MA}^{\mathcal{O}}$ languages, where a prover first produces a polynomially long natural-language argument and the debate verifies it.

The training story falls out nicely. Provers are trained by self-play as a zero-sum game; a prover's access to the oracle corresponds to the human knowledge already baked into a pretrained model, while the verifier's queries correspond to *actual* human ratings. So the human-labor cost of training is bounded by the verifier's query count, which means you can train on arbitrarily complex computations while asking humans only a constant number of questions per computation.

## Training, inference, and what the human actually does

One distinction the paper draws is worth making explicit, because it is what turns the theorems into a training recipe rather than a curiosity. At **training time**, the two provers are optimized by self-play to win the zero-sum debate game, and there are two different uses of the oracle in play: a prover's "access to $\mathcal{O}$" is really the human knowledge already compressed into a pretrained model plus what it picks up during self-play, whereas the *verifier's* queries to $\mathcal{O}$ are actual questions put to a human rater. Only the second kind costs human labor, and the whole payoff of doubly-efficient debate is that this second count is a constant. So you can train the provers on computations of unbounded complexity while asking real humans only $O(1)$ questions per computation. At **inference time** the protocol still runs, but no human feedback is consulted: prover $A$'s answer is simply trusted *unless* prover $B$ successfully objects and forces a check, so the debate becomes a cheap consistency mechanism rather than an ongoing drain on human attention. That asymmetry, expensive competitive training that buys cheap trustworthy inference, is the practical shape of the idea, and it is why the constant query count is the number that matters rather than a technicality.

## My take

I find this paper clarifying precisely because it is not an experiment. It answers whether there is *any* regime in which a weak judge can supervise a strong prover using bounded effort, with a yes and an explicit construction, and that reframes debate from a vibe into a claim with completeness and soundness constants you can argue about. The slogan finally has a formal referent.

I also think the limitations are the most interesting content, and the authors are candid about the biggest one: the **obfuscated argument problem**. All their theorems cover computations that *could have been* verified by a human reading the full polynomial-length transcript. A sufficiently capable model might produce an argument with no short human-checkable transcript at all, where a dishonest prover can in polynomial time generate a case the honest prover needs *exponential* time to refute. That is not a corner case; it is arguably the regime that matters most for genuinely superhuman systems, and the paper explicitly leaves it open. The two other soft spots are equally honest: the model assumes provers can accurately simulate human judgment on any query, which may hand them too much power, and the oracle is only *noisy*, not sometimes flatly *wrong*, whereas real human judgment (and the reward models we train to approximate it) can be confidently mistaken on a whole subset of inputs. So I read this as a real foundation with a precisely drawn boundary. It proves oversight *can* be efficient in a well-behaved regime, and in the same breath it names exactly the regime where we still have no idea, which is more than most safety papers are willing to do.
