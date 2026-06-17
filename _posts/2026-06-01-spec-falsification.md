---
title: "Why Does a Valid Proof Tell You Nothing About Whether You Proved the Right Thing?"
date: 2026-06-01
permalink: /posts/2026/06/spec-falsification/
tags:
  - formal methods
  - verification
  - open source
---

When you prove something with a computer, the workflow has two halves that are easy to confuse. First you write down a statement, which formal-methods people call a specification, or spec for short. A spec is a precise description, in a language a machine can read, of what your code or your theorem is supposed to do. Then you write a proof that your work satisfies that spec, and you hand both to a proof checker like Lean or Axiom's AXLE. The checker does something genuinely remarkable: it tells you, with certainty, whether the proof is valid. No hand-waving, no "looks right to me," just a mechanical verdict that the proof establishes the statement.

Here is the part almost nobody stops to examine. The checker only ever compares two things, the proof and the spec. It never compares the spec against what you actually wanted, because your intent lives in your head and was never written down in a form the machine can see. So if the spec is wrong, the checker will happily certify a flawless proof of the wrong thing, print "success," and leave you believing you verified something you did not. The proof is correct. The result is worthless. That gap is the entire reason I built [Popper](https://github.com/aravinds-kannappan/Popper).

---

## The failure mode is the statement, not the proof

A specification can fail in three quiet ways, and all three are invisible to a proof checker.

It can be **too loose**, meaning it accepts answers that are actually wrong. Suppose you are specifying a function that should return the maximum of two numbers, and your spec says only "the output is greater than or equal to `a` and greater than or equal to `b`." That sounds right until you notice it never says the output has to be one of `a` or `b`. A function that returns a billion satisfies the spec for any reasonable inputs. A proof that your code meets this spec proves nothing useful, because the spec itself was satisfiable by nonsense.

It can be **too tight**, meaning it rejects answers that are actually correct. Imagine specifying absolute value with the clause "the output is strictly greater than zero." That looks reasonable for a moment, and then `abs(0) = 0` walks in and fails a spec that should have accepted it. Now your correct implementation cannot be proven, and you may spend hours hunting for a bug in code that was right the whole time, because the bug was in the statement.

It can be **vacuous**, meaning it says nothing at all. The clearest case is a spec that amounts to "for all `x`, true." That proves instantly, because there is nothing to establish. A subtler version is a definition of "sorted" that says only "the output has the same length as the input." A function that ignores its argument and returns the input unchanged satisfies it. So does a function that shuffles the list randomly. The proof goes through, the green check appears, and you have certified a property that any list of the right length already has.

In every one of these cases the proof is valid. The whole apparatus looks correct. And the checker is structurally incapable of complaining, because from its point of view nothing is wrong: the proof really does establish the statement. The problem is that the statement was never the one you meant.

---

## Soundness, completeness, and why specs are where things break

It helps to name the two properties a good spec needs. A spec is **sound** if it never rejects a correct answer, and it is **complete** if it never accepts a wrong one. Too-tight specs violate soundness. Too-loose and vacuous specs violate completeness. A spec that is both sound and complete is the one you actually wanted: it draws the line in exactly the right place, admitting every correct answer and excluding every incorrect one.

The reason this matters in practice is that writing a sound and complete spec turns out to be harder than writing correct code. On Verina, a benchmark of programming tasks where each task ships with both code and a formal specification, the best general model writes code that is correct about 73 percent of the time, but writes specifications that are both sound and complete only about 52 percent of the time. Read that again, because it is the crux. The model is better at solving the problem than at saying what the problem is. The specification, not the implementation, is the weaker link, and it is precisely the link a proof checker is blind to.

This should not be surprising once you sit with it. Code has a behavior you can run, test, and watch fail. A specification is a claim about all possible behaviors at once, and the failure modes are subtractive: a missing clause, a dropped assumption, a quantifier in the wrong place. Nothing crashes. Nothing throws. The spec just quietly means less, or means something adjacent to what you intended, and every downstream proof inherits the error while looking impeccable.

---

## Three levels of trust

The easiest way to see where Popper fits is to lay out three levels of trust and ask what each one still misses.

| Stack | What you get | What it still misses |
|---|---|---|
| A model writes or explains a proof | Fluent, plausible text | Any guarantee at all; a skipped case or off-by-one is invisible |
| Model plus a prover (Lean or AXLE) | A real proof that matches the statement | Whether the statement is faithful to your intent |
| Model plus a prover plus Popper | A proof, plus an active attempt to break the statement | It does not itself prove or certify |

At the first level you have a model that produces something that reads well and carries no warranty. A wrong step or a missing edge case is invisible unless an expert reads it line by line. This is the level most people are at when they ask a chatbot to "prove" something, and it is worth being honest that fluent text and a correct result are different things.

The second level is a genuine step up. A prover like Lean or AXLE gives you a real proof that matches the statement, mechanically checked. But you are still trusting the statement, and the statement is exactly where the 52 percent failure rate lives. A vacuous spec proves instantly. A too-tight spec blocks your correct code. The prover will not warn you, because there is nothing wrong with the proof.

The third level adds a screen in front of the prover. Before you spend effort proving the statement, an independent process actively tries to break it. If it cannot, you have more reason to trust the spec. If it can, you get the exact input that breaks it, which tells you what to fix. Popper is that screen. It is not a competitor to the prover and it does not produce proofs. It answers a different question: is this the right statement to prove in the first place.

---

## Why breaking is cheaper than proving

There is a practical, economic argument for running the screen before the prover, and it comes down to the asymmetry between proving and breaking.

Proving is expensive. A proof search, whether run by a human or an agent like AxiomProver, is heavy compute. It explores a large space of possible proof steps, backtracks, and can run for a long time before it either succeeds or gives up. Breaking, by contrast, is cheap. To show a statement is wrong you only need one input that violates it, and finding that input is often a matter of sampling a few thousand random cases or evaluating a handful of known witnesses. The cost difference is not marginal; it is the difference between a search and a spot check.

That asymmetry creates two concrete situations where screening first saves real work. The first is a vacuous or too-weak spec. A prover pointed at "for all `x`, true" returns a clean proof almost immediately, and you walk away thinking you verified something when you verified nothing. A screen catches that before any proof compute is spent, so you never burn a proof certifying a statement that guarantees nothing. The second is a wrong or too-strong spec. A prover pointed at a statement that is actually false, because the spec has a bug, can grind through a large search budget failing to prove it, with no way to tell "this is hard" apart from "this is impossible." A falsifier finds the breaking input quickly and says, in effect, fix the spec, so the compute goes to repairing the statement instead of hunting for a proof that cannot exist.

The clean way to say it: the prover answers "can this be proved, and here is the proof," and its blind spot is the statement itself. The screen answers "is this the right statement," and its job is to spend a little cheap compute so the expensive compute lands on statements worth proving.

---

## How the falsifier actually works

Popper exposes a single oracle interface over two engines. Whatever the engine, it takes a statement, tries to break it, and reports one of six verdicts: FAITHFUL when it could not break the statement within its budget, FALSIFIED when it found a clear counterexample, UNSOUND when the spec rejects a known-correct answer, INCOMPLETE when the spec accepts a known-wrong answer, VACUOUS when the spec accepts even nonsense, and INCONCLUSIVE when there simply was not enough signal to decide, reported honestly rather than guessed. That last verdict matters more than it looks. A tool that pretends to certainty it does not have is worse than one that admits when it cannot tell.

The **math engine** handles inequalities and identities, the bread and butter of mathematical statements, by numerical sampling. A statement carries an assumption and a conclusion, for example "if these quantities form a probability distribution, then this divergence is non-negative." The engine samples thousands of random cases and checks whether the conclusion holds every time the assumption does. This is a Monte-Carlo check, which is a precise way of saying "try a great many random inputs and see if anything breaks." The power of the approach is that the common failure modes of a hand-written statement, a dropped assumption or a flipped inequality, almost always break on some random input. If you forget to require that a vector of probabilities sums to one, the sampler will eventually draw a vector that does not, and the supposedly non-negative quantity will go negative right in front of you. The engine runs locally, with no prover and no API key, which is part of the point: the cheap screen should not depend on the expensive machinery.

Here is what its output looks like when faithful statements survive and broken ones get a witness:

```
kl_nonneg_DROPPED_norm    sum q = 3.53 (not 1) => KL = -1.15 < 0
dpi_DROPPED_markov        I(X;Z) = 0.83 > I(X;Y) = 0.05  (Z leaks X directly)
entropy_convex_WRONG      H(mix) = 1.08 > the average 1.06  (entropy is concave)
```

Each line is a real counterexample. The first shows a Kullback-Leibler divergence going negative because the normalization assumption was dropped, so the sampled `q` does not sum to one. The second shows a data-processing inequality violated because the Markov assumption was removed, letting `Z` leak information about `X` directly. The third catches a statement that claimed entropy is convex when it is concave, by exhibiting a mixture whose entropy exceeds the average. None of these requires a prover. They are arithmetic, and arithmetic is enough to refute a false universal claim with a single witness.

The **code engine** handles specifications of programs, where the question is whether the spec draws the right boundary around correct behavior. Each task comes with one known-correct answer and several known-wrong ones. Popper asks AXLE to evaluate the spec against each. If the spec rejects the correct answer, it is too tight, and the verdict is UNSOUND. If it accepts a wrong answer, it is too loose, and the verdict is INCOMPLETE. If it accepts even nonsense, it is VACUOUS. This is the live half of the system, running real Lean through AXLE on real Verina tasks, and it is how Popper catches the 52-percent problem at its source.

---

## The counterexample is the repair signal

A pass-or-fail bit tells you that something is wrong. A counterexample tells you what is wrong, and that difference is what turns Popper from a critic into a collaborator. When the engine hands back the exact input that breaks a spec, that input doubles as a repair hint, because it points straight at the missing clause or the flipped direction. Popper's repair loop takes the witness, adjusts the statement, and checks again, iterating until the statement holds up or the budget runs out.

```
sort_by_length        VACUOUS    -> FAITHFUL   (add sortedness and permutation)
max_lower_bound_only  INCOMPLETE -> FAITHFUL   (also require out in {a, b})
abs_strictly_positive UNSOUND    -> FAITHFUL   (relax > 0 to >= 0 for abs(0))
```

Read these as before-and-after stories. The first spec defined sorting by length alone, which is vacuous; the repair adds the two clauses sorting actually needs, that the output is ordered and is a permutation of the input. The second spec required only that the maximum be a lower bound on both arguments, which is incomplete; the repair also requires the output to actually be one of the two values. The third spec demanded a strictly positive absolute value, which is unsound because it rejects `abs(0) = 0`; the repair relaxes the inequality. In each case the counterexample is not just a complaint, it is the diff.

---

## The numbers

I wanted a measurement, not just an argument, so I built a benchmark of 346 statements where the ground truth is known ahead of time: whether each statement is faithful, and if not, what kind of bug it has. Then I ran three checkers over the same set and counted how many of the unfaithful statements each one caught.

| checker | unfaithful caught | false alarms | gives a counterexample |
|---|---|---|---|
| Popper | 176/178 (99%) | 0/165 (0%) | yes, every time |
| Proof checker (AXLE/Lean) | 0/178 (0%) | 0/165 (0%) | no |
| LLM judge (a model reads the spec and guesses) | runnable live | runnable live | no |

The proof checker catches none of the bad specs, and that is the headline, not an embarrassment. Every spec in the set is valid as far as the proof is concerned, so catching a bad spec is simply not a thing a proof checker does. Pointing it at this benchmark is like asking a spell-checker to catch a factual error: correct spelling, wrong fact, and the tool has nothing to say. The LLM judge, a model that reads the spec and guesses whether it is faithful, can sometimes be right, but it never hands you the input that breaks the spec, so even when it guesses correctly it cannot show its work or drive a repair.

The two specs Popper misses are worth being honest about, because they reveal exactly where the method's edge is. They are the rarest subtle bugs, ones that fail on roughly one input in ten thousand. At a sampling budget of two thousand draws, the Monte-Carlo engine sometimes does not happen to hit the bad input. The fix is not a different algorithm, it is more draws, and the benchmark records this as a sweep:

| draws per statement | math recall | subtle bugs caught |
|---|---|---|
| 100 | 98% | 6/10 |
| 500 | 99% | 8/10 |
| 2,000 | 99% | 8/10 |
| 10,000 | 100% | 10/10 |
| 50,000 | 100% | 10/10 |

The trend is exactly what you would expect from a sampler: rare events get caught once you draw enough samples to hit them. By ten thousand draws the engine catches everything, including the one-in-ten-thousand bugs, and the cost is still trivial compared to a proof search. This is the honest shape of a Monte-Carlo method, and I would rather show the curve than quote the single number that flatters the tool.

---

## A live benchmark built from your own questions

The offline benchmark measures whether the detection engine catches planted bugs. The website measures something different and, to me, more interesting: how the full agent does on real questions. The site opens on a live demo where you can send math or logic claims to the Popper agent, which tries to break each claim through AXLE and shows the real result. After ten messages, a benchmark unlocks that is built from your conversation rather than from a canned set.

Three systems get compared: the Popper agent, which is a strong model paired with AXLE and the falsificationist reasoning; the same model on its own with no tools; and AXLE by itself, the bare prover with no agent around it. A separate evaluator agent decides the true answer, treating what AXLE found in your chat as authoritative, and grades all three. The grades become metrics, accuracy, F1, Matthews correlation coefficient, counterexample yield, and average quality, one row per system. The ten graded results are then bootstrapped to five hundred resamples to put honest confidence intervals on each number, which is a way of saying "resample the ten results many times for a tighter estimate," not "invent five hundred new data points."

Seeing AXLE on its own is the instructive part. It can refute a false claim when the counterexample is small, but it cannot confirm a true one, because confirming requires a proof and the bare engine is not running the proof search. That gap, between refuting the false and confirming the true, is exactly the space the agent fills, and watching the three systems spread out on real questions makes the argument better than any fixed number could.

---

## A note on honesty

Popper breaks statements; it does not certify them. A FAITHFUL verdict means it could not find a counterexample within its budget, not that none exists. Proving that no counterexample exists is undecidable in general, so I do not claim it, and I built the INCONCLUSIVE verdict precisely so the tool can admit when it cannot decide rather than guess. What Popper does catch is the common, real failure that the 52-percent number is made of: a dropped assumption, a flipped direction, a spec that is too loose or too tight or says nothing at all. Lean and AXLE remain the final word on the proof itself. Popper just makes sure that final word is spent on a statement worth proving, and when the statement is not worth proving, it hands you the concrete input that shows you why.
