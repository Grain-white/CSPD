# CSPD GT posterior audit

## Purpose

This probe tests the hypothesis that the actor gradually fits a biased CSPD target.  It uses a fixed panel of
Math prefixes for all saved checkpoints and estimates the policy-specific success posterior by Monte Carlo:

\[
m^{GT}(a\mid s)=P_{a,\tau\sim\pi}(a, R(\tau)=1\mid s)=\pi(a\mid s)Q^{GT}(s,a),
\qquad
q^{GT}(a\mid s)=\frac{m^{GT}(a\mid s)}{\sum_bm^{GT}(b\mid s)}.
\]

Each sampled continuation is assigned to its first-token bucket (top-8 or tail) and scored by the same verl Math
verifier used in training.  This estimates success mass directly and avoids dividing by small action counts.
The posterior uses raw successful first-token frequencies. Prefixes with zero successes have no identifiable conditional posterior, so posterior-distance metrics are missing; raw value and success-mass metrics remain valid.

## Experimental controls

- Prefix panel: generated once by the base Qwen3-1.7B; high-entropy response prefixes are frozen across all runs.
- Actor: existing Math75K PPO or CSPD checkpoint at steps 50, 100, 150, 200, 250, 300.
- Critic cross-over: every actor sample is evaluated with both the PPO critic and CSPD critic at the same step.
- Policy sampling: temperature 1, no top-k/top-p truncation, because the CSPD equations use the full softmax policy.
- Default pilot: 3 fixed prefixes, 32 continuations per prefix, top-K=8, at most 1024 new tokens.

## Metric table

| Group | Metric | Definition | Diagnostic meaning |
|---|---|---|---|
| GT support | `v_gt` / `n_success` | Monte-Carlo success rate / successful samples | Reliability of the conditional posterior estimate |
| State critic | `v_critic`, `v_abs_error`, `v_brier` | clipped `(V_signed+1)/2` vs MC success rate | State-value calibration |
| Top-K critic | `top_mass_l1` | `sum_i abs(pi_i Qcritic_i - mGT_i)` | Error before tail projection and renormalization |
| Action critic | `q_mae_observed`, `q_rank_corr_observed` | critic Q vs empirical Q on sampled top actions | Candidate calibration and ordering; noisier than success mass |
| Tail | `tail_mass_gt` | successful samples whose first token is outside top-K, divided by N | True grouped tail success mass |
| Tail | `tail_mass_raw`, `tail_mass_projected` | `Vcritic - sum_top pi Qcritic`, then clamp to `[0, pi_tail]` | Residual-tail quality and projection effect |
| Tail | `tail_mass_abs_error_raw/projected` | absolute error against GT tail success mass | Whether projection improves the tail estimate |
| Feasibility | `tail_projection` | raw residual outside `[0, pi_tail]` | Bellman inconsistency rate |
| Renorm | `posterior_mass`, `weight_mass_abs_gap`, `state_to_mass_ratio` | projected mass total and its mismatch with V | Detects `target` renormalized by one mass but loss weighted by another |
| Mass target | `mass_l1_projected` | `L1(m_projected, m_GT)` | Error in the projected unnormalized success masses, before the loss's extra `V/M` scaling |
| Mass target | `mass_l1_loss_target` | `L1(Vcritic*q_CSPD, m_GT)` | Error in the actual target mass entering the implemented loss |
| Update direction | `update_cosine_gt`, `update_dot_gt` | alignment of `Vcritic(q_CSPD-pi)` with `m_GT-V_GT*pi` | Direct test of whether the CSPD logit update points with or against the verifier-grounded update |
| Update magnitude | `update_norm_cspd`, `update_norm_gt`, `update_norm_ratio_gt`, `update_l2_error_gt` | norms and error of the same two grouped-logit update vectors | Separates bad direction from a direction that is merely over/under-scaled |
| Posterior | `kl/js/tv_gt_cspd` | grouped GT posterior vs CSPD target | Direct target bias |
| Actor | `kl/js/tv_gt_actor` | grouped GT posterior vs current actor | Policy distance from desired posterior |
| Actor | `kl/js/tv_cspd_actor` | CSPD target vs current actor | Current CSPD update pressure |
| Relative | `cspd_kl_gain_over_actor` | `KL(GT||actor)-KL(GT||CSPD)` | Positive means CSPD target is closer to GT than the actor; negative means the update points away from GT |
| Value gating | `v_weighted_kl_gt_cspd` | `Vcritic * KL(GT||CSPD)` | Value-weighted target bias actually entering the proposal loss |
| Value gating | `v_weighted_kl_cspd_actor` | `Vcritic * KL(CSPD||actor)` | Approximate CSPD update pressure after the state-value gate |
| Value gating | `harmful_update_pressure` | `Vcritic * max(0, KL(GT||CSPD)-KL(GT||actor))` | Pressure from targets worse than leaving the actor unchanged |
| Value gating | `beneficial_update_pressure` | `Vcritic * max(0, KL(GT||actor)-KL(GT||CSPD))` | Pressure from targets that improve on the current actor |
| Value gating | `net_update_pressure` | beneficial minus harmful pressure | Signed local proxy for whether CSPD points toward GT |
| Curriculum | `v_bin` | critic V buckets `[0,.25), [.25,.5), [.5,.75), [.75,1]` | Whether useful/bad posterior targets concentrate in different confidence regions |

## Decisive patterns

1. If `KL(GT||CSPD)` rises while MATH-500 falls, the target distribution itself drifts away from GT.
2. If `KL(CSPD||actor)` falls, `KL(GT||actor)` rises, and performance falls, the actor-learning-biased-target hypothesis is supported.
3. If replacing the CSPD critic with the PPO critic materially fixes posterior KL on the same CSPD actor samples, critic counterfactual Q is the main cause.
4. If both critics give similar posterior KL while `top_mass_l1` is modest but tail/renorm metrics are bad, post-processing is the main cause.
5. If `cspd_kl_gain_over_actor < 0`, a CSPD step is locally worse than leaving the actor unchanged on that prefix panel.
6. To test Math75K vs DAPO difficulty, use `mass_l1_loss_target`, update cosine, and update norm as the primary
   evidence.  Raw conditional-posterior KL is secondary: because `q=m/V`, a fixed mass error is mechanically amplified
   when `V` is small, and the effective Monte-Carlo sample size is only about `N*V_GT`.
7. If DAPO and Math both have poor update cosine but DAPO's update norm is much smaller, the difference is value
   gating.  If DAPO cosine is positive while Math becomes negative, the datasets produce genuinely different directions.

## Current top-K versus proposal two-stage selection

The training implementation currently retains policy-probability top-8 directly.  Proposal Section 3.2/E instead
defines two stages: propose `K0` actions by policy probability, score all of them with the critic, then retain `K=8`
by unnormalized posterior mass `pi(a|s) * Q(s,a)`.  The selection probe compares `K0 in {16,32,64}` with:

| Metric | Meaning |
|---|---|
| `overlap_fraction`, `replacement_count`, `jaccard` | How much the two retained token sets differ |
| `proposal_success_mass_gain` | Extra critic-predicted success mass captured by posterior-mass reranking |
| `proposal_pi_mass_change` | Policy probability sacrificed to select higher-value actions |
| `current/proposal_tail_projection` | Whether reranking changes residual-tail feasibility |
| `update_cosine_current_proposal` | Difference between the two actual grouped-logit directions |
| `update_cosine_*_full_k0` | Which top-8 approximation is closer to scoring all K0 actions individually |
| `update_norm_ratio_proposal_current` | Whether two-stage selection changes update scale as well as direction |

Because the pilot has few prefixes and finite Bernoulli samples, it is a diagnostic rather than a publication-quality
estimate.  Any conclusion should be rerun with more prefixes/samples and bootstrap confidence intervals.
