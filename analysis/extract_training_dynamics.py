#!/usr/bin/env python3
"""Extract a compact metric time series from verl's text training log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
METRICS = [
    "val-core/openai/gsm8k/acc/mean@4",
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/entropy",
    "cspd_loss",
    "cspd_value_weight",
    "cspd_target_entropy",
    "diagnostics/cspd_state_success/mean",
    "diagnostics/cspd_state_success/std",
    "diagnostics/cspd_successor_success/mean",
    "diagnostics/cspd_successor_success/std",
    "diagnostics/cspd_signed_candidate_advantage/mean",
    "diagnostics/cspd_signed_candidate_advantage/std",
    "diagnostics/cspd_signed_candidate_advantage/abs_mean",
    "diagnostics/cspd_behavior_tail_probability/mean",
    "diagnostics/cspd_raw_tail_mass/abs_mean",
    "diagnostics/cspd_projected_tail_mass/mean",
    "diagnostics/cspd_posterior_mass/mean",
    "diagnostics/cspd_state_to_mass_ratio/mean",
    "diagnostics/cspd_state_to_mass_ratio/std",
    "diagnostics/cspd_state_to_mass_ratio/max",
    "diagnostics/cspd_tail_projection_fraction",
    "critic/grad_norm",
    "critic/vf_loss",
    "critic/vpred_mean",
    "critic/vf_explained_var",
    "critic/score/mean",
    "critic/values/mean",
    "critic/returns/mean",
    "diagnostics/raw_gae/std",
    "response_length/mean",
]


def extract(path: Path, label: str) -> list[dict]:
    rows: dict[int, dict] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw)
        match = re.search(r"\bstep:(\d+)\s+-", line)
        if not match:
            continue
        step = int(match.group(1))
        row = rows.setdefault(step, {"run": label, "step": step})
        for metric in METRICS:
            value = re.search(re.escape(metric) + r":" + NUMBER, line)
            if value:
                row[metric] = float(value.group(1))
    return [rows[step] for step in sorted(rows)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, action="append", required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.log) != len(args.label):
        raise SystemExit("--log and --label counts differ")
    rows = []
    for path, label in zip(args.log, args.label, strict=True):
        rows.extend(extract(path, label))
    frame = pd.DataFrame(rows).sort_values(["run", "step"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
