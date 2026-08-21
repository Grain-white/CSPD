# GT posterior audit results

## Setup

- Existing checkpoints only; no training was restarted or modified.
- Math75K: CSPD and PPO actors at steps 50/100/150/200/250/300, crossed with both critics.
- DAPO-Math17K: CSPD checkpoints at steps 50/100/150/200/250 (no matching saved PPO checkpoint exists).
- Fixed panel, 3 prefixes, 32 full-softmax continuations per prefix, top-K=8, 1024-token cap.
- `q_GT` is the raw first-token distribution among verifier-successful continuations. Zero-success prefixes have missing posterior metrics.
- Two DAPO panels: natural generated high-entropy prefixes (low V, posterior often unidentified) and successful-rollout high-entropy prefixes (enough successes for posterior comparison).

## Math75K CSPD actor + CSPD critic

| dataset   |   step |   math500_mean12 |   v_gt |   v_critic |   posterior_valid |   kl_gt_cspd |   kl_gt_actor |   v_weighted_tv_gt_cspd |   harmful_update_pressure |   net_update_pressure |   tail_projection |   top_mass_l1 |
|:----------|-------:|-----------------:|-------:|-----------:|------------------:|-------------:|--------------:|------------------------:|--------------------------:|----------------------:|------------------:|--------------:|
| math75k   |     50 |           0.6753 | 0.7812 |     0.6416 |            1.0000 |       0.6312 |        0.1362 |                  0.1358 |                    0.3311 |               -0.3311 |            1.0000 |        0.2711 |
| math75k   |    100 |           0.6668 | 0.7812 |     0.7120 |            1.0000 |       0.8018 |        0.2630 |                  0.1419 |                    0.2427 |               -0.2356 |            0.6667 |        0.2863 |
| math75k   |    150 |           0.6690 | 0.8438 |     0.6257 |            1.0000 |       0.1917 |        0.1841 |                  0.1194 |                    0.0106 |                0.0008 |            0.6667 |        0.3768 |
| math75k   |    200 |           0.6633 | 0.8542 |     0.5322 |            1.0000 |       0.1450 |        0.1786 |                  0.0947 |                    0.0059 |                0.0240 |            1.0000 |        0.3995 |
| math75k   |    250 |           0.6612 | 0.8958 |     0.6715 |            1.0000 |       0.2682 |        0.1195 |                  0.1157 |                    0.1131 |               -0.1079 |            1.0000 |        0.3303 |
| math75k   |    300 |           0.6575 | 0.8542 |     0.7673 |            1.0000 |       0.1124 |        0.0946 |                  0.1094 |                    0.0110 |               -0.0081 |            0.6667 |        0.2321 |

## DAPO natural generated low-V panel

| dataset   |   step |   v_gt |   v_critic |   posterior_valid |   top_mass_l1 |   tail_projection |
|:----------|-------:|-------:|-----------:|------------------:|--------------:|------------------:|
| dapo17k   |     50 | 0.0208 |     0.1582 |            0.3333 |        0.1135 |            0.3333 |
| dapo17k   |    100 | 0.0104 |     0.0882 |            0.3333 |        0.0708 |            0.6667 |
| dapo17k   |    150 | 0.0208 |     0.1299 |            0.3333 |        0.0690 |            0.3333 |
| dapo17k   |    200 | 0.0521 |     0.1696 |            0.3333 |        0.0971 |            0.3333 |
| dapo17k   |    250 | 0.0208 |     0.0430 |            0.3333 |        0.0331 |            0.6667 |

Posterior distances are not interpreted here: only one of three prefixes has any successful continuation at each checkpoint.

## DAPO successful-rollout panel

| dataset         |   step |   math500_mean12 |   v_gt |   v_critic |   posterior_valid |   kl_gt_cspd |   kl_gt_actor |   v_weighted_tv_gt_cspd |   harmful_update_pressure |   net_update_pressure |   tail_projection |   top_mass_l1 |
|:----------------|-------:|-----------------:|-------:|-----------:|------------------:|-------------:|--------------:|------------------------:|--------------------------:|----------------------:|------------------:|--------------:|
| dapo17k_success |     50 |           0.6833 | 0.6667 |     0.2720 |            1.0000 |       3.3081 |        0.3078 |                  0.1118 |                    0.0501 |               -0.0501 |            0.3333 |        0.3468 |
| dapo17k_success |    100 |           0.6903 | 0.6250 |     0.1729 |            1.0000 |       4.1054 |        0.5733 |                  0.0812 |                    0.0387 |               -0.0387 |            0.3333 |        0.3532 |
| dapo17k_success |    150 |           0.6893 | 0.5729 |     0.1777 |            1.0000 |       3.0256 |        0.5029 |                  0.0776 |                    0.1471 |               -0.1471 |            0.3333 |        0.3052 |
| dapo17k_success |    200 |           0.6947 | 0.6042 |     0.1667 |            1.0000 |       3.2269 |        0.6036 |                  0.0788 |                    0.2152 |               -0.2152 |            0.3333 |        0.4000 |
| dapo17k_success |    250 |           0.6947 | 0.6042 |     0.0514 |            1.0000 |       3.2638 |        0.3267 |                  0.0160 |                    0.0587 |               -0.0568 |            0.3333 |        0.3540 |

## Common-step means (50-250)

|                               |   Math75K |   DAPO-success |   DAPO/Math |
|:------------------------------|----------:|---------------:|------------:|
| v_gt                          |    0.8313 |         0.6146 |      0.7393 |
| v_critic                      |    0.6366 |         0.1681 |      0.2641 |
| kl_gt_cspd                    |    0.4076 |         3.3860 |      8.3077 |
| kl_gt_actor                   |    0.1763 |         0.4629 |      2.6257 |
| v_weighted_tv_gt_cspd         |    0.1215 |         0.0731 |      0.6014 |
| harmful_update_pressure       |    0.1407 |         0.1020 |      0.7247 |
| net_update_pressure           |   -0.1300 |        -0.1016 |      0.7817 |
| top_mass_l1                   |    0.3328 |         0.3519 |      1.0572 |
| tail_projection               |    0.8667 |         0.3333 |      0.3846 |
| weight_mass_abs_gap           |    0.0730 |         0.0271 |      0.3717 |
| tail_mass_abs_error_projected |    0.0155 |         0.1712 |     11.0143 |

## Math cross-critic control

| critic_method   |   v_abs_error |   top_mass_l1 |   kl_gt_cspd |   v_weighted_tv_gt_cspd |   harmful_update_pressure |   tail_projection |
|:----------------|--------------:|--------------:|-------------:|------------------------:|--------------------------:|------------------:|
| cspd            |        0.1841 |        0.3160 |       0.3584 |                  0.1195 |                    0.1191 |            0.8333 |
| ppo             |        0.2574 |        0.3266 |       0.3521 |                  0.1117 |                    0.1344 |            0.9444 |

## V-bias correlation within each 3-prefix panel

| dataset         |   step |   corr_v_kl |
|:----------------|-------:|------------:|
| math75k         |     50 |     -0.0214 |
| math75k         |    100 |     -0.9435 |
| math75k         |    150 |     -0.6797 |
| math75k         |    200 |      0.1785 |
| math75k         |    250 |      0.5650 |
| math75k         |    300 |     -0.6652 |
| dapo17k_success |     50 |     -0.9288 |
| dapo17k_success |    100 |     -0.9223 |
| dapo17k_success |    150 |     -0.9851 |
| dapo17k_success |    200 |     -0.9973 |
| dapo17k_success |    250 |     -0.9326 |

## Findings

1. DAPO's raw conditional-posterior KL is 8.3x Math, but this is not evidence by itself that its gradient target is
   8.3x worse.  Since `q=m/V`, small `V` mechanically amplifies mass error in posterior space, and low success counts
   make the Monte-Carlo conditional posterior noisier.  Mass-space and update-vector diagnostics are the primary test.
2. DAPO's critic value is only 26.4% of Math (0.168 vs 0.637). The gradient-relevant bounded proxy `V*TV` is 39.9% lower, and harmful pressure is 27.5% lower.
3. On DAPO, V and posterior KL are strongly negatively correlated at every checkpoint (r=-0.92 to -1.00), but that
   correlation is partly built into the normalization and must not be read as independent evidence of adaptive gating.
4. DAPO projected-tail absolute error is 0.171 vs 0.016 on Math.  This too is not sufficient to infer gradient harm:
   the loss uses `V*q`, and its actual direction and magnitude must be compared with `m_GT-V_GT*pi`.
5. Replacing the CSPD critic by the PPO critic on the identical Math actor/prefixes does not systematically fix the target. Mean KL and harmful pressure are similar; step 100 favors PPO critic while step 300 favors CSPD critic. This argues against a CSPD-critic-specific failure.
6. Math has the largest harmful pressure at steps 50 and 100 (0.331 and 0.243), when actor gradients are also largest; pressure mostly anneals afterward. Continued later performance decline is therefore compatible with cumulative fitting of early biased targets, not with late numerical instability.
7. DAPO MATH-500 rises from 0.6833 (step 50) to 0.6947 (steps 200/250). The probe explains why biased targets need not hurt there (value gating), but does not prove that CSPD itself causes the small gain: the fixed-panel net proxy remains slightly negative and only three prefixes were tested.

## Limitations

This is a diagnostic pilot, not a publication-quality estimate. Three prefixes are enough to expose the consistent DAPO V-KL anticorrelation but not to estimate dataset-wide causal effects. A final paper result should use at least 32-64 fixed prefixes, bootstrap confidence intervals, and saved on-policy prefix panels at each checkpoint.
