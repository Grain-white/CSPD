#!/usr/bin/env python3
"""1-GPU debug: align GSM8K verifier with upstream verl (#### + gsm8k.compute_score)."""

from __future__ import annotations

import json
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
from verl.utils.reward_score.math_dapo import last_boxed_only_string  # noqa: E402

MODEL = "/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B"
TRAIN = ROOT / "data" / "gsm8k" / "train.parquet"

# Upstream verl preprocess instruction (examples/data_preprocess/gsm8k.py)
VERL_INSTR = 'Let\'s think step by step and output the final answer after "####".'
# Stronger variant if base model ignores ####
STRICT_INSTR = (
    'Solve the problem step by step. '
    'You MUST end your response with a line of the exact form: #### <number> '
    '(four hash marks, a space, then the final numeric answer). '
    'Do NOT use \\boxed{}.'
)


def has_hash(ans: str) -> bool:
    return bool(re.search(r"####\s*-?[0-9\.\,]+", ans[-400:]))


def has_boxed(ans: str) -> bool:
    return last_boxed_only_string(ans) is not None


@torch.inference_mode()
def gen_one(model, tok, user_content: str, max_new: int = 1024) -> str:
    msgs = [{"role": "user", "content": user_content}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def eval_prompt_variant(name: str, rows, model, tok, make_prompt):
    n = len(rows)
    stats = {
        "name": name,
        "n": n,
        "strict_acc": 0,  # #### or boxed fallback (training scorer)
        "strict_hash_only_acc": 0,  # raw #### extract only
        "flexible_acc": 0,
        "has_####": 0,
        "has_boxed": 0,
        "examples": [],
    }
    for i, r in enumerate(rows):
        q = r["extra_info"]["question"] if isinstance(r.get("extra_info"), dict) else None
        if not q:
            # fall back: strip from prompt content
            content = r["prompt"][0]["content"]
            q = content.rsplit(VERL_INSTR, 1)[0].strip()
        gt = r["reward_model"]["ground_truth"]
        prompt = make_prompt(q)
        ans = gen_one(model, tok, prompt)
        # compute_score(method=strict) now also accepts \\boxed{} fallback
        scored = gsm8k_score.compute_score(ans, gt, method="strict")
        strict_only = gsm8k_score.extract_solution(ans, method="strict")
        strict_only_ok = (
            1.0
            if strict_only is not None
            and gsm8k_score._answers_equal(strict_only, gt)
            else 0.0
        )
        flex = gsm8k_score.compute_score(ans, gt, method="flexible")
        stats["strict_acc"] += int(scored == 1.0)
        stats["strict_hash_only_acc"] += int(strict_only_ok == 1.0)
        stats["flexible_acc"] += int(flex == 1.0)
        stats["has_####"] += int(has_hash(ans))
        stats["has_boxed"] += int(has_boxed(ans))
        if i < 3:
            stats["examples"].append(
                {
                    "gt": gt,
                    "scored": scored,
                    "strict_hash_only": strict_only_ok,
                    "flex": flex,
                    "tail": ans[-200:],
                }
            )
        print(
            f"[{name}] {i+1}/{n} scored={scored} hash_only={strict_only_ok} flex={flex} "
            f"####={has_hash(ans)} boxed={has_boxed(ans)}",
            flush=True,
        )
    for k in ("strict_acc", "strict_hash_only_acc", "flexible_acc", "has_####", "has_boxed"):
        stats[k] = stats[k] / n
    return stats


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    df = pd.read_parquet(TRAIN)
    rng = random.Random(42)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    rows = [df.iloc[i].to_dict() for i in idxs[:n]]

    print("Loading", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    variants = [
        (
            "verl_default_####",
            lambda q: f"{q} {VERL_INSTR}",
        ),
        (
            "strict_force_####",  # == rebuilt train.parquet instruction
            lambda q: f"{q}\n\n{STRICT_INSTR}",
        ),
        (
            "boxed_math_style",
            lambda q: (
                "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
                + q
            ),
        ),
    ]

    all_stats = []
    for name, fn in variants:
        print(f"\n======== {name} ========", flush=True)
        all_stats.append(eval_prompt_variant(name, rows, model, tok, fn))

    out = ROOT / "output" / "gsm8k_verifier_debug.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print("\nSUMMARY", flush=True)
    for s in all_stats:
        print(
            f"{s['name']}: scored={s['strict_acc']:.3f} "
            f"hash_only={s['strict_hash_only_acc']:.3f} flex={s['flexible_acc']:.3f} "
            f"####rate={s['has_####']:.3f} boxedrate={s['has_boxed']:.3f}",
            flush=True,
        )
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
