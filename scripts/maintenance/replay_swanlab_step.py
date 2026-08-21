#!/usr/bin/env python3
"""Replay one numeric verl console-metric line into an existing SwanLab run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--workspace", default="grainrain")
    parser.add_argument("--project", default="CSPD-PPO-comparison")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_metrics(path: Path, step: int) -> dict[str, float]:
    marker = f"training/global_step:{step}"
    matched = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if marker in line and f"step:{step} - " in line:
                matched = ANSI_RE.sub("", line)
    if matched is None:
        raise RuntimeError(f"No metric line for step {step} in {path}")

    payload = matched.split(f"step:{step} - ", 1)[1]
    metrics: dict[str, float] = {}
    for field in payload.split(" - "):
        if ":" not in field:
            continue
        key, raw_value = field.rsplit(":", 1)
        try:
            metrics[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    if metrics.get("training/global_step") != float(step):
        raise RuntimeError(f"Parsed line does not identify step {step}")
    return metrics


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.log, args.step)
    print(
        f"parsed {len(metrics)} metrics for step {args.step}; "
        f"mean@4={metrics.get('val-core/openai/gsm8k/acc/mean@4')}"
    )
    if args.dry_run:
        return

    import swanlab

    run = swanlab.init(
        workspace=args.workspace,
        project=args.project,
        id=args.run_id,
        resume="must",
    )
    run.log(metrics, step=args.step)
    run.finish()


if __name__ == "__main__":
    main()
