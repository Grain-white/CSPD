# CSPD diagnostic results (2026-08-20)

## Scope and definitions

All checkpoint probes are read-only.  The pilot uses 3 fixed prefixes, 32 verifier-scored continuations per prefix,
top-8 actions plus one tail bucket, and existing Qwen3-1.7B checkpoints.  Math75K uses steps 50-300; DAPO uses
steps 50-250.  DAPO has both a natural low-success panel and a successful-rollout panel.  The latter has enough
successful continuations for conditional-posterior comparisons.

The primary diagnostics are in success-mass and logit-update space:

```text
m_GT[i]     = empirical P(first bucket i and eventual success)
m_loss[i]   = V_critic * q_CSPD[i]
u_GT[i]     = m_GT[i] - V_GT * pi[i]
u_CSPD[i]   = m_loss[i] - V_critic * pi[i]
```

`u_GT` and `u_CSPD` are compared using cosine, dot product, norms, norm ratio, L2 error, per-bucket sign agreement,
push precision/recall, and the fraction of CSPD update absolute mass assigned to coordinates with the wrong sign.
The detail CSVs also store the full 9-coordinate vectors and token strings as JSON columns.  This is a direct
direction test, not an inference from `V` and conditional-posterior KL.

## Mass and update-direction results

### Per checkpoint

| dataset | step | V_GT | V_critic | L1 actual loss mass | update cosine | CSPD norm | GT norm | norm ratio | top-8 sign agreement | wrong-sign CSPD mass | tail projection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Math75K | 50 | .7812 | .6416 | .3535 | .3483 | .1067 | .1008 | 1.1775 | .4583 | .3461 | 1.0000 |
| Math75K | 100 | .7812 | .7120 | .3094 | .3780 | .0528 | .1400 | .4367 | .5000 | .4025 | .6667 |
| Math75K | 150 | .8438 | .6257 | .3820 | -.1719 | .0482 | .1426 | .3121 | .3333 | .6190 | .6667 |
| Math75K | 200 | .8542 | .5322 | .4394 | .0798 | .0398 | .1719 | .2312 | .5000 | .3924 | 1.0000 |
| Math75K | 250 | .8958 | .6715 | .3686 | .0565 | .0392 | .1145 | .3532 | .3750 | .3259 | 1.0000 |
| Math75K | 300 | .8542 | .7673 | .2592 | .0780 | .0196 | .1017 | .1880 | .4583 | .5137 | .6667 |
| DAPO-success | 50 | .6667 | .2720 | .5547 | -.3046 | .0591 | .0863 | .6889 | .2917 | .4813 | .3333 |
| DAPO-success | 100 | .6250 | .1729 | .5518 | -.1290 | .0379 | .1225 | .3187 | .3750 | .2895 | .3333 |
| DAPO-success | 150 | .5729 | .1777 | .5024 | -.4152 | .0377 | .0874 | .5523 | .5417 | .6987 | .3333 |
| DAPO-success | 200 | .6042 | .1667 | .5540 | -.1645 | .0385 | .1166 | .3755 | .5417 | .5311 | .3333 |
| DAPO-success | 250 | .6042 | .0514 | .5742 | .6683 | .0171 | .1541 | .2120 | .5417 | .1129 | .3333 |

The DAPO natural panel has only 0.3-1.7 successful samples per 32-sample prefix average.  Its conditional posterior
and cosine are not reliable enough to interpret.  Its raw success-mass metrics remain available in the CSV.

### Common-step means (50-250)

| dataset | L1 projected mass | L1 actual loss mass | update cosine | CSPD norm | GT norm | norm ratio | top-8 sign agreement | wrong-sign CSPD mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Math75K | .3484 | .3706 | .1381 | .0573 | .1340 | .5021 | .4333 | .4172 |
| DAPO-success | .5231 | .5474 | -.0690 | .0381 | .1134 | .4295 | .4583 | .4227 |

DAPO's target is not more accurate: it has larger success-mass error and slightly negative mean direction on this
fixed success panel.  Its CSPD update norm is about 33.5% smaller than Math (`.0381` vs `.0573`), so value/scale
gating is real, but the wrong-sign fraction is almost identical (`.423` vs `.417`).  The result supports a weaker
claim - harmful updates are partly attenuated on DAPO - not the stronger claim that DAPO learns from a more accurate
posterior.  Since DAPO nevertheless gains slightly, this 3-prefix success-conditioned panel is not a causal or
representative estimate of the whole on-policy training distribution.

### Concrete per-action direction example

For Math step 150, prefix `row1-p0` has cosine `-0.639`.  The critic target reverses 6 of the 8 top-token directions:

| bucket | pi | m_GT | m_loss | u_GT | u_CSPD | sign match |
|---|---:|---:|---:|---:|---:|:---:|
| `Equ` | .3562 | .3438 | .1212 | +.0878 | -.0249 | no |
| `Set` | .1906 | .1250 | .0824 | -.0120 | +.0042 | no |
| `Substitute` | .1310 | .0938 | .0657 | -.0004 | +.0119 | no |
| `Use` | .1156 | .0000 | .0549 | -.0831 | +.0075 | no |
| `Express` | .0901 | .0625 | .0371 | -.0022 | +.0001 | no |
| `Elim` | .0546 | .0938 | .0251 | +.0545 | +.0027 | yes |
| `Combine` | .0331 | .0000 | .0213 | -.0238 | +.0077 | no |
| `Sub` | .0058 | .0000 | .0025 | -.0041 | +.0002 | no |
| tail | .0230 | .0000 | .0000 | -.0165 | -.0094 | yes |

Thus a low or negative cosine has concrete meaning: CSPD suppresses an empirically successful token such as `Equ`
and raises unsuccessful alternatives such as `Use`.  The same vectors for every tested prefix are stored in
`detail-actor-*.csv` under `update_gt_by_bucket`, `update_cspd_by_bucket`, and `update_sign_match_by_bucket`.

## Current top-8 versus proposal's two-stage selection

The current training implementation selects actor-probability top-8 and only scores those actions.  Proposal Section
3.2/E specifies: propose policy top-K0, score all K0 actions, then retain top-8 by `pi(a|s) * Q(s,a)`.  K0 values
16, 32, and 64 were tested.  K0=32 and K0=64 are already nearly saturated.

### Mean across checkpoints and prefixes (K0=64)

| dataset | overlap@8 | replacements / 8 | policy-mass change | predicted success-mass ratio | current/proposal cosine | proposal/current norm |
|---|---:|---:|---:|---:|---:|---:|
| Math75K | .9792 | .1667 | .0000 | 1.0004 | .9993 | 1.0004 |
| DAPO-natural | .5833 | 3.3333 | -.2087 | 1.3205 | .7177 | .5636 |
| DAPO-success | .6917 | 2.4667 | -.0824 | 1.1451 | .7184 | .7086 |

This approximation mismatch cannot explain the Math75K decline: on Math it changes almost nothing.  It matters a
great deal on DAPO, where posterior reranking exchanges 2.5-3.3 of 8 tokens and materially changes both direction
and scale.  Two-stage selection should therefore be implemented for proposal fidelity, but it is not a demonstrated
fix for the observed Math failure.

An additional caution: on DAPO, posterior-mass top-8 is not consistently closer to the grouped full-K0 update than
policy top-8.  At K0=64 the cosine-to-full-K0 difference averages `-0.160` on the natural panel and `-0.092` on the
success panel.  Maximizing captured positive mass `pi*Q` is not the same as minimizing policy-gradient direction
error, which depends on signed advantage `Q-V` and the tail derivative.  A useful ablation is candidate retention by
`pi*abs(Q-V)` or a stratified mix of positive and negative `pi*(Q-V)`, while keeping the proposal-defined `pi*Q`
variant as the canonical method.

## GSM8K corrected 0/1 run

The latest job used `cspd_reward_range=01`, ran 50 steps, and deliberately set `SAVE_FREQ=-1`.  It completed but
saved no actor/critic checkpoint, so GT posterior and update cosine cannot be reconstructed after the fact.  The log
dynamics can still be compared with the older erroneous `pm1` mapping and PPO.

| run | step 0 | step 10 | step 20 | step 30 | step 40 | step 50 | delta 0-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSPD, corrected 0/1 | .7441 | .7407 | .7227 | .7092 | .6751 | .6308 | -.1133 |
| CSPD, old pm1 mapping | .7411 | .7485 | .7521 | .7672 | .7671 | .7638 | +.0227 |
| PPO | .7494 | .7521 | .7708 | .7750 | .7788 | .7877 | +.0383 |

| run / phase | actor grad norm | state success | raw-tail abs mass | mean V/M | tail projection |
|---|---:|---:|---:|---:|---:|
| corrected, steps 1-10 | 43.51 | .5201 | .3743 | 1854.68 | .9472 |
| corrected, steps 11-30 | 12.30 | .7231 | .2150 | 140.59 | .9819 |
| corrected, steps 31-50 | 5.99 | .6561 | .1371 | 1.79 | .9862 |
| old pm1, steps 1-10 | 17.30 | .7489 | .2584 | 211.76 | .9613 |
| old pm1, steps 11-30 | 5.77 | .8598 | .1085 | 1.03 | .9755 |
| old pm1, steps 31-50 | 3.33 | .8764 | .0645 | 1.01 | .9803 |
| PPO, steps 1-10 | 1.18 | n/a | n/a | n/a | n/a |
| PPO, steps 11-30 | 1.27 | n/a | n/a | n/a | n/a |
| PPO, steps 31-50 | 1.20 | n/a | n/a | n/a | n/a |

The corrected run exposes the same early-large-gradient / later-annealed-gradient pattern more strongly than Math.
Accuracy continues to fall while the actor gradient becomes smaller, which is consistent with cumulative fitting of
an early biased target rather than late numerical explosion.  The enormous early mean `V/M` is heavy-tailed (some
prefixes have nearly zero projected posterior mass), and the corrected mapping has much larger residual-tail error.

The old mapping was still mathematically wrong for a 0/1 critic.  Its apparent improvement is plausibly an accidental
regularizer: `(value+1)/2` adds a 0.5 offset, makes action success values more alike, keeps the target closer to the
behavior policy, and substantially damps actor gradients.  It should be treated as a smoothing ablation, not evidence
that the posterior formula intended the pm1 transform.

## Overall interpretation and next tests

1. The relation between `V` and raw posterior KL is insufficient.  Small `V` mechanically amplifies conditional
   posterior error because `q=m/V`; mass and update-vector diagnostics are the primary evidence.
2. Math's mean update cosine becomes negative at step 150 and stays near zero afterward while MATH-500 continues to
   decline.  The actor norm falls from `.107` to `.020`, supporting cumulative learning of biased directions.
3. DAPO does not have a more accurate tested posterior; it mostly has a smaller CSPD update norm.  The slight DAPO
   gain must come from unmeasured on-policy prefixes, beneficial directions outside the tiny panel, regularization,
   or ordinary run variance.
4. Two-stage candidate selection is essential for proposal fidelity but cannot explain Math's drop because it is
   almost identical to current top-8 there.  It substantially changes DAPO and deserves a controlled ablation.
5. The strongest implementation-level suspect is the combination of cold critic values, residual-tail projection,
   and renormalization: the implemented target mass is `(V/M) * m_projected`.  GSM8K shows extreme early `V/M` and
   very large actor gradients after the correct 0/1 mapping exposes low/zero critic outputs.
6. Highest-priority controlled fixes are: warm up or confidence-gate CSPD and fall back to PPO; use the projected mass
   directly (`weight=M`) or cap `V/M`; log update cosine against sampled-action/MC estimates; and save GSM8K
   checkpoints every 10 steps so its directions can be audited rather than inferred from aggregate logs.

## Files

- Mass/direction details: `output/gt-posterior-math75k-pilot`, `output/gt-posterior-dapo17k-pilot`, and
  `output/gt-posterior-dapo17k-success`.
- Top-K details and summaries: `output/topk-selection-comparison`.
- GSM8K extracted dynamics: `output/gsm8k_training_dynamics.csv`.
- Reproducible probes: `analysis/gt_posterior_probe.py`, `analysis/topk_selection_probe.py`, and their salloc scripts.

The three-prefix checkpoint probes are diagnostic pilots, not paper-quality estimates.  A decisive study should use
32-64 frozen on-policy prefixes per checkpoint, bootstrap confidence intervals, and saved GSM8K checkpoints.
