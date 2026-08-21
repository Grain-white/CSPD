#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    paths = sorted(args.output_dir.glob("detail-*.csv"))
    if not paths:
        raise SystemExit("no top-K selection detail files")
    detail = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    detail.to_csv(args.output_dir / "topk_selection_detail.csv", index=False)
    keys = ["dataset", "actor_method", "critic_method", "step", "retain_k", "proposal_k0"]
    numeric = [column for column in detail.columns if column not in keys and pd.api.types.is_numeric_dtype(detail[column])]
    summary = detail.groupby(keys, as_index=False)[numeric].mean()
    summary.to_csv(args.output_dir / "topk_selection_summary.csv", index=False)
    overall_keys = ["dataset", "actor_method", "critic_method", "retain_k", "proposal_k0"]
    overall = detail.groupby(overall_keys, as_index=False)[numeric].mean()
    overall.to_csv(args.output_dir / "topk_selection_overall.csv", index=False)

    columns = [
        "dataset", "proposal_k0", "overlap_fraction", "replacement_count", "proposal_pi_mass_change",
        "proposal_success_mass_gain", "proposal_success_mass_ratio", "current_tail_projection",
        "proposal_tail_projection", "update_cosine_current_proposal", "update_norm_ratio_proposal_current",
        "update_cosine_current_full_k0", "update_cosine_proposal_full_k0", "proposal_cosine_gain_to_full_k0",
    ]
    per_step_columns = ["dataset", "step"] + columns[2:]
    k0_max = int(detail.proposal_k0.max())
    per_step = summary[summary.proposal_k0 == k0_max][per_step_columns]
    report = (
        "# Current top-K vs proposal two-stage selection\n\n"
        "The current implementation keeps policy-probability top-8 directly.  The proposal first considers "
        "policy top-K0 and then retains top-8 by predicted success mass `pi(a|s) * Q(s,a)`.\n\n"
        "## Mean across checkpoints and prefixes\n\n"
        + overall[columns].to_markdown(index=False, floatfmt=".4f")
        + f"\n\n## Per-checkpoint comparison at K0={k0_max}\n\n"
        + per_step.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    (args.output_dir / "TOPK_SELECTION_RESULTS.md").write_text(report, encoding="utf-8")
    print(overall[columns].to_string(index=False))


if __name__ == "__main__":
    main()
