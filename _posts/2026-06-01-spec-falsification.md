---
title: "Why Does a Valid Proof Tell You Nothing About Whether You Proved the Right Thing?"
date: 2026-06-01
permalink: /posts/2026/06/spec-falsification/
tags:
  - formal methods
  - verification
  - open source
---

When you prove something with a computer, you write down a statement — a specification, a precise description of what the code or the math is supposed to do — and then you prove that your work matches that statement. A prover like Lean or Axiom's AXLE will then tell you, for certain, whether the proof is valid. The catch is the one almost nobody checks: the prover only ever compares the proof against the statement. It never compares the statement against what you actually meant.

So if the statement is wrong, the prover happily certifies a proof of the wrong thing and reports success. This is the blind spot I built [Popper](https://github.com/aravinds-kannappan/Popper) to attack.

---

## The failure mode is the statement, not the proof

A specification can fail in three quiet ways. It can be **too loose**, accepting wrong answers. It can be **too tight**, rejecting right ones. Or it can be **vacuous**, saying nothing at all — `for all x, true` proves instantly, and a definition of "sorted" as "has the same length as the input" is satisfied by code that does nothing. In every one of those cases the proof still goes through, and the whole thing looks correct.

This is not a rare corner case. On Verina, a benchmark of code-with-specifications, the best general model writes code that is correct about 73% of the time but writes specs that are both sound and complete only about 52% of the time. Here "sound" means the spec doesn't reject a correct answer and "complete" means it doesn't accept a wrong one. The spec, not the proof, is where things break — and the prover cannot see it.

---

## Three levels of trust

It helps to think about what each tool in the stack still misses.

| Stack | What you get | What it still misses |
|---|---|---|
| A model writes a proof | Fluent text | No guarantee at all |
| Model + prover (Lean / AXLE) | A proof that matches the statement | Whether the statement is faithful |
| Model + prover + Popper | A proof, plus an active attempt to break the statement | (it does not prove or certify) |

The prover is a verifier: it answers "can this statement be proved, and here is the proof." Popper is a screen that sits in front of it and answers a different question — "is this the right statement to prove" — by trying to break it and handing back a concrete input when it can.

---

## Breaking is cheaper than proving

The practical argument for running the screen first is economic. A proof search is heavy compute; a few thousand random samples are not. It is wasteful to point the expensive tool at a statement the cheap tool can already show is wrong. Two situations make this concrete:

- A **vacuous or too-weak** spec proves *easily* and certifies nothing. Without a screen you'd walk away from a clean proof of `for all x, true` thinking you verified something.
- A **wrong or too-strong** spec is *unprovable*, and the prover can grind a large search budget failing on it. A falsifier finds the breaking input quickly and tells you to fix the spec instead.

---

## How the falsifier actually works

Popper exposes one oracle interface over two engines, each returning a verdict: FAITHFUL, FALSIFIED, UNSOUND, INCOMPLETE, VACUOUS, or INCONCLUSIVE (reported honestly when there isn't enough signal, rather than guessed).

The **math engine** handles inequalities and identities by Monte-Carlo. A statement carries an assumption and a conclusion; the engine samples thousands of random cases and checks whether the conclusion holds whenever the assumption does. If the statement dropped an assumption or flipped a direction, some random case breaks it — and that case *is* the counterexample. It runs locally, no prover, no API key:

```
kl_nonneg_DROPPED_norm    sum q = 3.53 (not 1) => KL = -1.15 < 0
dpi_DROPPED_markov        I(X;Z) = 0.83 > I(X;Y) = 0.05  (Z leaks X directly)
entropy_convex_WRONG      H(mix) = 1.08 > the average 1.06  (entropy is concave)
```

The **code engine** asks AXLE to evaluate a spec against a known-correct answer and several known-wrong ones. A correct answer rejected means the spec is too tight (unsound); a wrong answer accepted means it's too loose (incomplete); nonsense accepted means it's vacuous.

A counterexample is strictly more useful than a pass/fail bit, because it tells you *which* input breaks the spec — so it doubles as a repair hint. Popper's repair loop takes the witness, adjusts the statement, and checks again:

```
sort_by_length        VACUOUS    -> FAITHFUL   (add sortedness and permutation)
max_lower_bound_only  INCOMPLETE -> FAITHFUL   (also require out in {a, b})
abs_strictly_positive UNSOUND    -> FAITHFUL   (relax > 0 to >= 0 for abs(0))
```

---

## The numbers

I wanted a measurement, not just an argument, so I built a benchmark of 346 statements where the ground truth — faithful, or unfaithful and what kind — is known ahead of time.

| checker | unfaithful caught | false alarms | gives a counterexample |
|---|---|---|---|
| Popper | 176/178 (99%) | 0/165 (0%) | yes, every time |
| Proof checker (AXLE/Lean) | 0/178 (0%) | 0/165 (0%) | no |

The proof checker catches none of them — and that is the point, not a knock on it. Every spec in the set is valid as far as the proof is concerned; catching a bad spec is simply not a thing a proof checker does. The two misses are the rarest subtle bugs, which fail on roughly 1 input in 10,000, and they're a function of sample budget:

| draws per statement | math recall | subtle bugs caught |
|---|---|---|
| 100 | 98% | 6/10 |
| 2,000 | 99% | 8/10 |
| 10,000 | 100% | 10/10 |

---

## A note on honesty

Popper breaks statements; it does not certify them. A FAITHFUL verdict means it could not find a counterexample within its budget, not that none exists — proving that no counterexample exists is undecidable in general, so I don't claim it. What it does catch is the common, real failure: a dropped assumption, a flipped direction, a spec that is too loose or too tight. Lean and AXLE remain the final word on the proof itself. Popper just makes sure that word is spent on a statement worth proving.
