#!/usr/bin/env python3
"""One-GPU pass@1 probe for candidate PPO train sets (Qwen3-1.7B)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sdpo_verl"))

from verl.utils.reward_score import gsm8k as gsm8k_score  # noqa: E402
from verl.utils.reward_score.math_dapo import compute_score, last_boxed_only_string  # noqa: E402

MATH_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _as_text_prompt(prompt_field) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, list):
        parts = []
        for m in prompt_field:
            if isinstance(m, dict) and m.get("content"):
                parts.append(str(m["content"]))
        return "\n\n".join(parts)
    return str(prompt_field)


def load_rows(name: str, path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(path)
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]
    rows = []
    for i in idxs:
        r = df.iloc[i].to_dict()
        if name == "gsm8k":
            q = r["question"]
            gt = gsm8k_score.extract_solution(r["answer"], method="strict")
            if gt is None:
                continue
            rows.append({"problem": q, "ground_truth": gt, "scorer": "gsm8k"})
        elif name == "deepscaler":
            rm = r.get("reward_model") or {}
            if isinstance(rm, str):
                rm = json.loads(rm)
            gt = rm.get("ground_truth")
            prompt = r.get("prompt")
            # pandas may yield numpy arrays of chat dicts
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            problem = _as_text_prompt(prompt)
            if isinstance(prompt, list):
                user = [
                    m
                    for m in prompt
                    if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
                ]
                if user:
                    problem = user[-1]["content"]
            rows.append({"problem": problem, "ground_truth": gt, "scorer": "math"})
        elif name == "orz":
            rows.append(
                {
                    "problem": r["problem"],
                    "ground_truth": str(r["answer"]).strip(),
                    "scorer": "math",
                }
            )
        elif name == "numina":
            sol = r.get("solution") or ""
            boxed = last_boxed_only_string(sol)
            if boxed is None:
                # fall back: last #### style / last line
                m = re.findall(r"####\s*(.+)", sol)
                gt = m[-1].strip() if m else None
            else:
                # strip \boxed{...}
                gt = boxed
                if gt.startswith("\\boxed{") and gt.endswith("}"):
                    gt = gt[len("\\boxed{") : -1]
            if not gt:
                continue
            rows.append({"problem": r["problem"], "ground_truth": gt, "scorer": "math"})
        else:
            raise ValueError(name)
    return rows


def score_one(scorer: str, completion: str, gt: str) -> bool:
    if scorer == "gsm8k":
        # Prefer boxed if present, else ####, else whole string via gsm8k flexible on pred
        pred = None
        boxed = last_boxed_only_string(completion)
        if boxed:
            pred = boxed[len("\\boxed{") : -1] if boxed.startswith("\\boxed{") else boxed
        if pred is None:
            pred = gsm8k_score.extract_solution(completion, method="flexible")
        if pred is None:
            return False
        # numeric compare via gsm8k.compute_score expects solution with #### for strict;
        # compare extracted strings loosely.
        try:
            return abs(float(str(pred).replace(",", "")) - float(str(gt).replace(",", ""))) < 1e-6
        except Exception:
            return str(pred).strip() == str(gt).strip()
    out = compute_score(completion, gt)
    return bool(out.get("acc"))


@torch.inference_mode()
def generate_batch(model, tok, problems: list[str], max_new: int, temp: float, top_p: float, batch_size: int):
    outs = []
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        texts = []
        for p in chunk:
            msgs = [
                {"role": "system", "content": MATH_SYSTEM},
                {"role": "user", "content": p},
            ]
            texts.append(
                tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        do_sample = temp > 0
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temp if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for i in range(gen.size(0)):
            outs.append(tok.decode(gen[i, prompt_len:], skip_special_tokens=True))
        print(f"  generated {min(start + batch_size, len(problems))}/{len(problems)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "output" / "probe_passrate.json"))
    args = ap.parse_args()

    datasets = {
        "gsm8k": ROOT / ".." / "d-opsd" / "outputs" / "datasets" / "gsm8k" / "main" / "train-00000-of-00001.parquet",
        "deepscaler": Path("/WORK/PUBLIC/alex_work/Meiqi.Gu/TreePO/data/DeepScaler.train.parquet"),
        "orz": ROOT / "data" / "probe_sets" / "orz_math_57k" / "data" / "train-00000-of-00001.parquet",
        "numina": ROOT / "data" / "probe_sets" / "numina_cot" / "data" / "train-00000-of-00005.parquet",
    }
    # gsm8k path: prefer sibling then absolute
    gsm_candidates = [
        Path("/home/fit/alex1/WORK/Meiqi.Gu/d-opsd/outputs/datasets/gsm8k/main/train-00000-of-00001.parquet"),
        datasets["gsm8k"],
    ]
    for p in gsm_candidates:
        if p.exists():
            datasets["gsm8k"] = p
            break

    print("Loading model", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    summary = {"model": args.model, "n": args.n, "temperature": args.temperature, "datasets": {}}
    for name, path in datasets.items():
        path = Path(path)
        if not path.exists():
            print(f"[skip] {name}: missing {path}", flush=True)
            continue
        rows = load_rows(name, path, args.n, args.seed)
        print(f"\n=== {name}: {len(rows)} samples from {path} ===", flush=True)
        completions = generate_batch(
            model,
            tok,
            [r["problem"] for r in rows],
            max_new=args.max_new,
            temp=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )
        hits = 0
        for r, c in zip(rows, completions):
            ok = score_one(r["scorer"], c, r["ground_truth"])
            hits += int(ok)
        acc = hits / max(len(rows), 1)
        summary["datasets"][name] = {
            "path": str(path),
            "n": len(rows),
            "pass_at_1": acc,
            "correct": hits,
        }
        print(f"{name}: pass@1={acc:.4f} ({hits}/{len(rows)})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\nWrote", out, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
