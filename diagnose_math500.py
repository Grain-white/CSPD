import glob
import importlib.util
import json
import os
import re

import pandas as pd

_score_path = "/home/fit/alex1/WORK/Meiqi.Gu/CSPD/sdpo_verl/verl/utils/reward_score/math_dapo.py"
_score_spec = importlib.util.spec_from_file_location("cspd_math_dapo", _score_path)
_score_module = importlib.util.module_from_spec(_score_spec)
_score_spec.loader.exec_module(_score_module)
compute_score = _score_module.compute_score


DATA = "/WORK/PUBLIC/alex_work/Meiqi.Gu/SDPO/datasets/dapo_hf/processed/math-500.eval.parquet"
SWAN = "/home/fit/alex1/WORK/Meiqi.Gu/CSPD/swanlog"


def prompt_text(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return "\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in value)
    return str(value)


df = pd.read_parquet(DATA)
print("rows", len(df), "columns", list(df.columns))
print("optional verifiers", {name: bool(importlib.util.find_spec(name)) for name in ("math_verify", "sympy", "latex2sympy2_extended")})
print("first record", df.iloc[0].to_dict())

tables = sorted(glob.glob(os.path.join(SWAN, "run-20260815_*", "media", "echarts", "000-*.json")))
print("step0 tables", tables)
for table in tables:
    payload = json.load(open(table, encoding="utf-8"))
    print("TABLE", table)
    matched_count = 0
    corrected_count = 0
    logged_correct_count = 0
    for i, row in enumerate(payload.get("rowData", [])[:16]):
        question = row.get("input", "")
        output = row.get("output", "")
        matches = []
        for _, record in df.iterrows():
            raw = prompt_text(record.get("prompt", ""))
            # The logged input includes the rendered chat template; a stable
            # 120-character question fragment is sufficient for alignment.
            # Match on the user content, excluding the system prefix.
            fragment = re.sub(r"\s+", " ", raw).strip()
            fragment = fragment.replace("You are a helpful assistant. ", "")[:100]
            if fragment and fragment in re.sub(r"\s+", " ", question):
                matches.append(record)
        gt = None
        if matches:
            reward_model = matches[0].get("reward_model", {})
            gt = reward_model.get("ground_truth") if isinstance(reward_model, dict) else reward_model
        recomputed = compute_score(output, gt) if gt is not None else None
        if gt is not None:
            matched_count += 1
            corrected_count += int(bool(recomputed["acc"]))
            logged_correct_count += int(row.get("score") == 1.0)
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", output)
        print(json.dumps({
            "i": i,
            "logged_score": row.get("score"),
            "matched": len(matches),
            "gt": gt,
            "last_boxed_simple": boxed[-1] if boxed else None,
            "recomputed": recomputed,
            "output_tail": output[-500:],
        }, ensure_ascii=False, default=str))
    print("SUMMARY", {"matched": matched_count, "logged_correct": logged_correct_count,
                      "boxed_first_correct": corrected_count,
                      "boxed_first_acc": corrected_count / matched_count if matched_count else None})
