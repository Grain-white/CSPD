#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MATH500 = {
    "cspd": {50: 0.675333, 100: 0.666833, 150: 0.669000, 200: 0.663333, 250: 0.661167, 300: 0.657500},
    "ppo": {50: 0.676167, 100: 0.679500, 150: 0.685667, 200: 0.681167, 250: 0.679167, 300: 0.692333},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    rows = []
    for path in sorted(args.output_dir.glob("summary-actor-*-step*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no summary json files")
    df = pd.DataFrame(rows)
    if "dataset" not in df:
        df["dataset"] = "math75k"
    df["math500_mean12"] = [MATH500.get(a, {}).get(int(s)) if d == "math75k" else None for d, a, s in zip(df.dataset, df.actor_method, df.step)]
    df = df.sort_values(["dataset", "actor_method", "critic_method", "step"])
    csv_path = args.output_dir / "gt_posterior_metrics.csv"
    df.to_csv(csv_path, index=False)
    cols = [
        "dataset", "actor_method", "critic_method", "step", "math500_mean12", "v_gt", "v_critic", "v_abs_error",
        "top_mass_l1", "tail_mass_gt", "tail_mass_raw", "tail_projection", "weight_mass_abs_gap",
        "renorm_amplification", "mass_l1_projected", "mass_l1_loss_target", "update_cosine_gt",
        "update_dot_gt", "update_norm_cspd", "update_norm_gt", "update_norm_ratio_gt", "update_l2_error_gt",
        "update_sign_agreement_gt", "update_sign_agreement_topk_gt", "harmful_coordinate_fraction",
        "push_precision_gt", "push_recall_gt",
        "kl_gt_cspd", "kl_gt_actor", "kl_cspd_actor", "cspd_kl_gain_over_actor", "tv_gt_cspd",
        "v_weighted_kl_gt_cspd", "v_weighted_tv_gt_cspd", "v_weighted_kl_cspd_actor", "harmful_update_pressure",
        "beneficial_update_pressure", "net_update_pressure",
        "q_mae_observed", "q_rank_corr_observed", "n_success",
    ]
    md = "# GT posterior probe results\n\n" + df[cols].to_markdown(index=False, floatfmt=".4f") + "\n"
    (args.output_dir / "gt_posterior_metrics.md").write_text(md, encoding="utf-8")
    print(csv_path)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
