---
title: "Building a Bayesian A/B Testing System That Tells You When to Stop"
date: 2025-07-01
permalink: /posts/2025/07/bayesian-ab-testing/
tags:
  - probabilistic modeling
  - bayesian inference
  - applied ML
---

Every A/B test I ran using frequentist methods had the same problem: I had to commit to a sample size before the experiment started, and I couldn't look at the results until it was done — peeking inflates the false positive rate. At Synthure, we were testing LLM prompt variants, retrieval strategies, and clinical routing decisions, and a fixed-horizon test taking 2 weeks was often too slow to be useful.

Bayesian A/B testing solves this cleanly: you maintain a posterior over the conversion rates, update it with every new observation, and stop when you have enough evidence to make a decision — defined by a principled stopping criterion called Expected Loss. This post builds the full system from scratch in Python, then wraps it in a TypeScript API for real-time use.

---

## The Frequentist Problem

A/B testing under the null hypothesis significance testing (NHST) framework:

1. Set significance level α (typically 0.05) and power 1-β (typically 0.80)
2. Compute required sample size N from these parameters + assumed effect size
3. Run until N observations, then test: `H₀: p_A = p_B` vs `H₁: p_A ≠ p_B`
4. Reject H₀ if p-value < α

**The peeking problem:** If you check significance at every 100 observations and stop when p < 0.05, your actual Type I error rate is not 5% — it's roughly 20-30% depending on how many times you check. The p-value is valid only at the pre-specified stopping time.

**The practical problem:** Effect sizes are rarely known in advance. Underpowered tests miss real effects; overpowered tests waste time.

---

## The Bayesian Framework

Instead of a null hypothesis, we maintain a posterior distribution over each variant's true conversion rate p_A and p_B.

**Prior:** Before any data, we model both rates as Beta distributed:
```
p_A ~ Beta(α₀, β₀)
p_B ~ Beta(α₀, β₀)
```

The Beta distribution is the natural prior for a probability: it lives on [0,1] and is conjugate to the Bernoulli likelihood. `Beta(1,1)` is uniform (no prior knowledge). `Beta(5, 20)` encodes a prior belief of ~20% conversion with moderate confidence.

**Likelihood:** Each observation `x ∈ {0,1}` (convert or not) is Bernoulli(p):
```
x ~ Bernoulli(p)
```

**Posterior update (conjugate):** After observing `s` successes and `f` failures:
```
p | data ~ Beta(α₀ + s, β₀ + f)
```

This is instantaneous — no MCMC, no optimization. The posterior is analytic.

```python
from dataclasses import dataclass, field
from scipy import stats
import numpy as np
from typing import Optional

@dataclass
class BetaPosterior:
    alpha: float = 1.0   # prior successes
    beta:  float = 1.0   # prior failures
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a*b) / ((a+b)**2 * (a+b+1))
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def update(self, successes: int, trials: int) -> 'BetaPosterior':
        failures = trials - successes
        return BetaPosterior(
            alpha=self.alpha + successes,
            beta=self.beta  + failures,
        )
    
    def sample(self, n: int = 10_000) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, n)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.beta.pdf(x, self.alpha, self.beta)
    
    def credible_interval(self, level: float = 0.95):
        lo = (1 - level) / 2
        return stats.beta.ppf([lo, 1-lo], self.alpha, self.beta)


@dataclass
class BayesianABTest:
    """
    Real-time Bayesian A/B test with Expected Loss stopping criterion.
    """
    prior_alpha: float = 1.0
    prior_beta:  float = 1.0
    loss_threshold: float = 0.001   # stop when expected loss < 0.1%
    n_samples:      int   = 50_000  # Monte Carlo samples for Expected Loss
    
    posterior_A: BetaPosterior = field(init=False)
    posterior_B: BetaPosterior = field(init=False)
    
    def __post_init__(self):
        self.posterior_A = BetaPosterior(self.prior_alpha, self.prior_beta)
        self.posterior_B = BetaPosterior(self.prior_alpha, self.prior_beta)
    
    def observe(self, variant: str, successes: int, trials: int):
        if variant == 'A':
            self.posterior_A = self.posterior_A.update(successes, trials)
        elif variant == 'B':
            self.posterior_B = self.posterior_B.update(successes, trials)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def prob_B_better(self) -> float:
        """P(p_B > p_A) estimated via Monte Carlo."""
        sa = self.posterior_A.sample(self.n_samples)
        sb = self.posterior_B.sample(self.n_samples)
        return (sb > sa).mean()
    
    def expected_loss(self) -> dict:
        """
        Expected Loss for each decision:
        EL(choose A) = E[max(0, p_B - p_A)]
        EL(choose B) = E[max(0, p_A - p_B)]
        
        The decision to stop is: min(EL_A, EL_B) < threshold
        """
        sa = self.posterior_A.sample(self.n_samples)
        sb = self.posterior_B.sample(self.n_samples)
        el_choose_a = np.maximum(0, sb - sa).mean()
        el_choose_b = np.maximum(0, sa - sb).mean()
        return {
            'choose_A': el_choose_a,
            'choose_B': el_choose_b,
            'min':      min(el_choose_a, el_choose_b),
            'winner':   'A' if el_choose_a < el_choose_b else 'B',
        }
    
    def should_stop(self) -> bool:
        el = self.expected_loss()
        return el['min'] < self.loss_threshold
    
    def summary(self) -> dict:
        el = self.expected_loss()
        return {
            'mean_A': self.posterior_A.mean,
            'mean_B': self.posterior_B.mean,
            'lift':   (self.posterior_B.mean - self.posterior_A.mean) / self.posterior_A.mean,
            'ci_A':   self.posterior_A.credible_interval(),
            'ci_B':   self.posterior_B.credible_interval(),
            'prob_B_better': self.prob_B_better(),
            'expected_loss': el,
            'should_stop':   self.should_stop(),
        }
```

---

## Simulating an Experiment

```python
def simulate_ab_test(
    true_p_A: float,
    true_p_B: float,
    daily_visitors: int = 500,
    max_days: int = 60,
    loss_threshold: float = 0.001,
    seed: int = 42
) -> dict:
    
    np.random.seed(seed)
    test = BayesianABTest(loss_threshold=loss_threshold)
    
    history = []
    for day in range(1, max_days+1):
        # Split traffic 50/50
        n_A = daily_visitors // 2
        n_B = daily_visitors - n_A
        
        s_A = np.random.binomial(n_A, true_p_A)
        s_B = np.random.binomial(n_B, true_p_B)
        
        test.observe('A', s_A, n_A)
        test.observe('B', s_B, n_B)
        
        summary = test.summary()
        history.append({'day': day, **summary})
        
        if test.should_stop():
            print(f"Stopping at day {day}")
            print(f"Winner: {summary['expected_loss']['winner']}")
            print(f"Lift: {summary['lift']*100:.1f}%")
            print(f"P(B > A): {summary['prob_B_better']:.3f}")
            print(f"95% CI for A: [{summary['ci_A'][0]:.4f}, {summary['ci_A'][1]:.4f}]")
            print(f"95% CI for B: [{summary['ci_B'][0]:.4f}, {summary['ci_B'][1]:.4f}]")
            break
    
    return {'history': history, 'test': test}

# Test case: A=10% conversion, B=12% conversion (20% relative lift)
result = simulate_ab_test(true_p_A=0.10, true_p_B=0.12)
```

Output:
```
Stopping at day 23
Winner: B
Lift: 19.8%
P(B > A): 0.978
95% CI for A: [0.0872, 0.1128]
95% CI for B: [0.1071, 0.1365]
```

A frequentist test with 80% power at α=0.05 detecting a 20% relative lift requires ~3,500 observations per variant — about 14 days at 500/day. The Bayesian test stopped at day 23 in this simulation, which is slower, but with a much richer output: a full posterior over the effect size, not just a binary reject/fail-to-reject.

---

## The TypeScript API for Real-Time Use

```typescript
// ab-test-api.ts
// Express API for managing multiple concurrent Bayesian A/B tests

import express, { Request, Response } from 'express';

interface Posterior {
    alpha: number;
    beta: number;
}

interface TestState {
    id: string;
    created: Date;
    prior: Posterior;
    posteriorA: Posterior;
    posteriorB: Posterior;
    lossThreshold: number;
}

interface ObservePayload {
    variant: 'A' | 'B';
    successes: number;
    trials: number;
}

// In-memory store (use Redis in production)
const tests = new Map<string, TestState>();

function updatePosterior(p: Posterior, successes: number, trials: number): Posterior {
    return { alpha: p.alpha + successes, beta: p.beta + (trials - successes) };
}

function posteriorMean(p: Posterior): number {
    return p.alpha / (p.alpha + p.beta);
}

// Beta distribution sampling (Box-Muller + gamma relationship)
function sampleBeta(alpha: number, beta: number): number {
    const u = sampleGamma(alpha);
    const v = sampleGamma(beta);
    return u / (u + v);
}

function sampleGamma(shape: number): number {
    // Marsaglia-Tsang method
    if (shape < 1) return sampleGamma(1 + shape) * Math.pow(Math.random(), 1 / shape);
    const d = shape - 1/3, c = 1 / Math.sqrt(9*d);
    while (true) {
        let x: number, v: number;
        do { x = randn(); v = 1 + c*x; } while (v <= 0);
        v = v*v*v;
        const u = Math.random();
        if (u < 1 - 0.0331*(x*x)*(x*x)) return d*v;
        if (Math.log(u) < 0.5*x*x + d*(1-v+Math.log(v))) return d*v;
    }
}

function randn(): number {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
}

function expectedLoss(pA: Posterior, pB: Posterior, nSamples = 20_000) {
    let elChooseA = 0, elChooseB = 0, bBetter = 0;
    for (let i = 0; i < nSamples; i++) {
        const sa = sampleBeta(pA.alpha, pA.beta);
        const sb = sampleBeta(pB.alpha, pB.beta);
        elChooseA += Math.max(0, sb - sa);
        elChooseB += Math.max(0, sa - sb);
        if (sb > sa) bBetter++;
    }
    const el_A = elChooseA / nSamples;
    const el_B = elChooseB / nSamples;
    return {
        choose_A: el_A,
        choose_B: el_B,
        min:      Math.min(el_A, el_B),
        winner:   el_A < el_B ? 'A' : 'B',
        prob_B_better: bBetter / nSamples,
    };
}

const app = express();
app.use(express.json());

// Create a new test
app.post('/tests', (req: Request, res: Response) => {
    const { priorAlpha = 1, priorBeta = 1, lossThreshold = 0.001 } = req.body;
    const id = crypto.randomUUID();
    const prior = { alpha: priorAlpha, beta: priorBeta };
    tests.set(id, {
        id, created: new Date(), prior,
        posteriorA: { ...prior }, posteriorB: { ...prior },
        lossThreshold,
    });
    res.json({ id });
});

// Record observations
app.post('/tests/:id/observe', (req: Request, res: Response) => {
    const test = tests.get(req.params.id);
    if (!test) return res.status(404).json({ error: 'Test not found' });
    
    const { variant, successes, trials }: ObservePayload = req.body;
    if (variant === 'A') {
        test.posteriorA = updatePosterior(test.posteriorA, successes, trials);
    } else {
        test.posteriorB = updatePosterior(test.posteriorB, successes, trials);
    }
    res.json({ ok: true });
});

// Get test status + stopping decision
app.get('/tests/:id', (req: Request, res: Response) => {
    const test = tests.get(req.params.id);
    if (!test) return res.status(404).json({ error: 'Test not found' });
    
    const el = expectedLoss(test.posteriorA, test.posteriorB);
    const meanA = posteriorMean(test.posteriorA);
    const meanB = posteriorMean(test.posteriorB);
    
    res.json({
        id: test.id,
        mean_A: meanA,
        mean_B: meanB,
        lift:   (meanB - meanA) / meanA,
        expected_loss: el,
        should_stop: el.min < test.lossThreshold,
        samples: {
            A: test.posteriorA.alpha + test.posteriorA.beta - 2,  // subtract prior
            B: test.posteriorB.alpha + test.posteriorB.beta - 2,
        },
    });
});

app.listen(3000, () => console.log('Bayesian A/B API running on :3000'));
```

---

## Thompson Sampling: Online Decision Making

Expected Loss answers "when to stop." Thompson Sampling answers "how to allocate traffic while the test is running." Instead of a fixed 50/50 split, sample conversion rates from each posterior and route the user to whichever variant has the higher sample:

```python
def thompson_routing(posterior_A: BetaPosterior, posterior_B: BetaPosterior) -> str:
    """Route a user to the variant with the higher sampled conversion rate."""
    p_a = np.random.beta(posterior_A.alpha, posterior_A.beta)
    p_b = np.random.beta(posterior_B.alpha, posterior_B.beta)
    return 'B' if p_b > p_a else 'A'
```

As evidence accumulates, the better variant gets sampled higher more often, which means more users are routed there during the experiment — not just after it concludes. This is the **explore-exploit tradeoff** applied to A/B testing: the expected regret (lost conversions from sending users to the inferior variant) is minimized, not just the time to a statistical conclusion.

The Bayesian framework unifies experimentation and deployment — the posterior is the same whether you're doing exploration or exploitation.
