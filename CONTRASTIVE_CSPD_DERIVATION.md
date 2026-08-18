# Contrastive CSPD: outcome-conditioned posteriors and local policy-gradient equivalence

## 1. Setup

Fix the behavior policy `pi_k` during one rollout iteration. The verifier reward used by the current PPO run is

```text
R in {-1, +1}.
```

Define the signed value and action value

```math
V(s) = E_{\pi_k}[R \mid s],
\qquad
Q(s,a) = E_{\pi_k}[R \mid s,a].
```

The corresponding probabilities of eventual success are

```math
p(s) = P(R=+1\mid s)=\frac{1+V(s)}{2},
\qquad
p(s,a)=P(R=+1\mid s,a)=\frac{1+Q(s,a)}{2}.
```

For deterministic token concatenation, `Q(s,a) = V(s+a)`.

## 2. Two Bayesian action posteriors

Condition the behavior policy's next action on the final binary outcome.

### 2.1 Success posterior

```math
q^+(a\mid s)
=P(A=a\mid s,R=+1)
=\frac{\pi_k(a\mid s)p(s,a)}{p(s)}
=\pi_k(a\mid s)\frac{1+Q(s,a)}{1+V(s)}.
```

Interpretation: among behavior-policy trajectories that eventually succeed, `q+` is the next-token distribution at prefix `s`.

### 2.2 Failure posterior

```math
q^-(a\mid s)
=P(A=a\mid s,R=-1)
=\frac{\pi_k(a\mid s)[1-p(s,a)]}{1-p(s)}
=\pi_k(a\mid s)\frac{1-Q(s,a)}{1-V(s)}.
```

Interpretation: among behavior-policy trajectories that eventually fail, `q-` is the next-token distribution at the same prefix.

Their ratio has a useful Bayes-factor interpretation:

```math
\log\frac{q^+(a\mid s)}{q^-(a\mid s)}
=\log\frac{1+Q(s,a)}{1-Q(s,a)}
-\log\frac{1+V(s)}{1-V(s)}.
```

It measures the action's success log-odds relative to the state-level success log-odds. Positive values identify actions associated with success; negative values identify actions associated with failure.

## 3. Signed-KL objective

The most direct contrastive extension is

```math
L_{\pm}(\theta)
=p(s)D_{KL}(q^+\Vert\pi_\theta)
-[1-p(s)]D_{KL}(q^-\Vert\pi_\theta).
```

Because the posteriors and their weights are frozen,

```math
p(s)q^+(a\mid s)-[1-p(s)]q^-(a\mid s)
=\pi_k(a\mid s)Q(s,a).
```

Therefore, up to terms independent of `theta`,

```math
L_{\pm}(\theta)
=-\sum_a\pi_k(a\mid s)Q(s,a)\log\pi_\theta(a\mid s)+C.
```

At `theta = theta_k`, let

```math
G_k(a\mid s)=\nabla_\theta\log\pi_\theta(a\mid s)\vert_{\theta_k}.
```

Then

```math
-\nabla L_{\pm}(\theta_k)
=\sum_a\pi_k(a\mid s)Q(s,a)G_k(a\mid s).
```

The score-function identity gives

```math
\sum_a\pi_k(a\mid s)V(s)G_k(a\mid s)=0,
```

so

```math
-\nabla L_{\pm}(\theta_k)
=\sum_a\pi_k(a\mid s)[Q(s,a)-V(s)]G_k(a\mid s).
```

Thus Signed-KL has the correct local actor-critic gradient under an exact critic, full action support, on-policy states, and frozen targets.

### 3.1 Why bare Signed-KL is unsafe

If `Q(s,a) < 0`, the theta-dependent term is

```math
|\pi_k(a\mid s)Q(s,a)|\log\pi_\theta(a\mid s),
```

which approaches negative infinity as `pi_theta(a|s)` approaches zero. Bare Signed-KL therefore has no finite lower bound. Its local gradient is meaningful, but unrestricted multi-step minimization can collapse probability on predicted-failure actions and amplify critic errors.

## 4. Proximal Signed-KL and its relation to ordinary CSPD

Add a frozen-behavior forward KL:

```math
L_{\pm,\beta}(\theta)
=L_{\pm}(\theta)+\beta D_{KL}(\pi_k\Vert\pi_\theta).
```

The theta-dependent action weight becomes

```math
\pi_k(a\mid s)[\beta+Q(s,a)].
```

For `Q in [-1,1]`, choosing `beta >= 1` makes all weights nonnegative and defines a valid target

```math
q_\beta(a\mid s)
=\frac{\pi_k(a\mid s)[\beta+Q(s,a)]}{\beta+V(s)}.
```

Hence

```math
L_{\pm,\beta}(\theta)
=[\beta+V(s)]D_{KL}(q_\beta\Vert\pi_\theta)+C.
```

The target is a conservative mixture of behavior and success conditioning:

```math
q_\beta
=\frac{\beta-1}{\beta+V}\pi_k
+\frac{1+V}{\beta+V}q^+.
```

At `beta = 1`, `q_beta = q+`. Therefore ordinary success-posterior CSPD is, up to a factor of two and a theta-independent constant, Signed-KL plus the smallest globally safe behavior forward KL.

## 5. Bounded pairwise contrastive CSPD

Define the success-failure pair posterior

```math
\rho(a^+,a^-\mid s)=q^+(a^+\mid s)q^-(a^-\mid s).
```

It describes an independently drawn next action from successful behavior trajectories and one from failed behavior trajectories at the same prefix.

Use a behavior-relative policy-change score

```math
h_\theta(a\mid s)
=\log\frac{\pi_\theta(a\mid s)}{\pi_k(a\mid s)},
```

and pairwise margin

```math
\Delta_\theta(a^+,a^-\mid s)
=h_\theta(a^+\mid s)-h_\theta(a^-\mid s).
```

For the signed reward `R in {-1,+1}`, define

```math
L_{C\text{-}CSPD}(\theta)
=[1-V(s)^2]
E_{(a^+,a^-)\sim\rho}
\left[\operatorname{softplus}(-\Delta_\theta)\right].
```

This objective is nonnegative and bounded below by zero.

### 5.1 Local gradient equivalence

At `theta = theta_k`, all behavior-relative scores and margins are zero. Since

```math
-\nabla_\theta\operatorname{softplus}(-\Delta_\theta)\vert_{\theta_k}
=\frac{1}{2}[G_k(a^+\mid s)-G_k(a^-\mid s)],
```

we have

```math
-\nabla L_{C\text{-}CSPD}(\theta_k)
=\frac{1-V(s)^2}{2}
\left[E_{q^+}G_k-E_{q^-}G_k\right].
```

Pointwise,

```math
q^+(a\mid s)-q^-(a\mid s)
=\frac{2\pi_k(a\mid s)[Q(s,a)-V(s)]}{1-V(s)^2}.
```

Consequently,

```math
-\nabla L_{C\text{-}CSPD}(\theta_k)
=\sum_a\pi_k(a\mid s)[Q(s,a)-V(s)]G_k(a\mid s).
```

This is exactly the signed-reward actor-critic policy gradient. For a `{0,1}` reward objective, the prefactor is half as large because its advantage is `(Q-V)/2` in signed-value coordinates.

Behavior-relative rather than raw log-probability margins are necessary: they guarantee `Delta = 0` at the frozen reference, which supplies the exact factor `1/2` in the local derivative.

## 6. Top-K plus tail implementation

For each retained action `a_i`, define complementary success and failure masses

```math
m_i^+=\pi_k(a_i\mid s)\frac{1+Q(s,a_i)}{2},
\qquad
m_i^-=\pi_k(a_i\mid s)\frac{1-Q(s,a_i)}{2}.
```

They satisfy `m_i+ + m_i- = pi_k(a_i|s)`. Let

```math
\pi_T=1-\sum_i\pi_k(a_i\mid s).
```

Estimate and project only the success tail mass,

```math
m_T^+
=\operatorname{clip}\left(
\frac{1+V(s)}{2}-\sum_i m_i^+,
0,
\pi_T
\right),
```

and define failure mass by complement,

```math
m_T^-=\pi_T-m_T^+.
```

This preserves `m_T+ + m_T- = pi_T`. On the coarsened `K+1` support, set

```math
Z^+=\sum_bm_b^+,
\qquad Z^-=\sum_bm_b^-=1-Z^+,
\qquad q_b^\pm=\frac{m_b^\pm}{Z^\pm}.
```

The all-pairs bounded objective is

```math
L^{K+1}_{C\text{-}CSPD}
=4Z^+Z^-
\sum_{i,j}q_i^+q_j^-
\operatorname{softplus}(-\Delta_{ij}).
```

With `K=8`, the coarsened support has nine buckets and only 81 pairs, so all pairs can be summed exactly instead of sampled. This removes pair-sampling variance. The equivalence is exact only for the internally consistent coarsened model; top-K candidate omission and critic error remain approximation sources relative to the full-vocabulary policy gradient.

## 7. Degenerate cases and safeguards

- If `p(s)=0` or `p(s)=1`, one outcome posterior is undefined, but the pairwise prefactor is zero. Skip that state or use the continuous limiting value.
- Clip critic predictions to `[-1,1]` before converting them to outcome probabilities, but log the unclipped fraction as a calibration diagnostic.
- Detach `pi_k`, `V`, `Q`, `q+`, `q-`, and all posterior weights during actor optimization.
- Refresh the reference and posteriors every rollout iteration. The equivalence is local, not a statement about unrestricted optimization epochs.
- A high-entropy deterministic prefix selector changes the state distribution. Full state-distribution equivalence requires all prefixes or an unbiased prefix sampler with the appropriate inclusion correction.
- PPO currently whitens GAE over response tokens. To compare gradient scale, log raw and whitened GAE statistics and, if desired, multiply the whole CSPD loss by a frozen batch scalar. Do not standardize probabilities inside the posterior.

## 8. Recommended experimental matrix

1. `PPO-whiten`: current PPO/GAE baseline.
2. `CSPD-prob`: corrected success-probability posterior.
3. `CSPD-positive`: the old accidental positive-value gated objective, retained as an ablation.
4. `Signed-CSPD`: bare Signed-KL for short, tightly constrained diagnostics only.
5. `C-CSPD`: bounded all-pairs contrastive posterior objective.

For the same rollout batch and frozen actor, report gradient cosine similarity, norm ratio, raw GAE standard deviation, posterior mass ratio, tail projection fraction, and critic explained variance. Direction disagreement should be fixed before tuning a scalar loss coefficient.
