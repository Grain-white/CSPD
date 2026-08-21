#!/usr/bin/env python3
"""Rebuild GSM8K verl parquet with a stronger #### instruction for Qwen3."""

from __future__ import annotations

from pathlib import Path

import datasets
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "gsm8k"

# Stronger than upstream verl: Qwen3 otherwise prefers \\boxed{} and ignores ####.
INSTR = (
    "Solve the problem step by step. "
    "You MUST end your response with a line of the exact form: #### <number> "
    "(four hash marks, a space, then the final numeric answer). "
    "Do NOT use \\boxed{}."
)


def extract_gt(answer: str) -> str:
    # GSM8K answers end with #### <gt>
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().replace(",", "")


def to_row(example, idx: int, split: str):
    q = example["question"].strip()
    gt = extract_gt(example["answer"])
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": f"{q}\n\n{INSTR}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": q,
            "answer": example["answer"],
            "instruction": INSTR,
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Prefer local cache / mirror if present
    ds = datasets.load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        rows = [to_row(ex, i, split) for i, ex in enumerate(ds[split])]
        path = OUT / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"Wrote {path} n={len(rows)}")
        print("sample prompt tail:", rows[0]["prompt"][0]["content"][-120:])


if __name__ == "__main__":
    main()
