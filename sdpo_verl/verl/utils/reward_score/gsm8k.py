# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def _extract_boxed_numeric(solution_str: str):
    """Fallback for models (e.g. Qwen3) that emit \\boxed{n} instead of #### n."""
    try:
        from .math_dapo import last_boxed_only_string, remove_boxed
    except Exception:
        return None
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        inner = remove_boxed(boxed)
    except Exception:
        return None
    # Prefer a numeric token inside the box (strip $, commas, simple latex)
    inner = inner.replace(",", "").replace("$", "").strip()
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", inner)
    if not nums:
        return inner if inner else None
    return nums[-1]


def _answers_equal(pred, ground_truth) -> bool:
    if pred is None:
        return False
    p = str(pred).replace(",", "").replace("$", "").strip()
    g = str(ground_truth).replace(",", "").replace("$", "").strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer

    Notes:
        For method='strict', if no #### answer is found we also accept \\boxed{...}.
        Qwen-family chat models often ignore the verl #### instruction and box instead;
        without this fallback, training reward collapses to ~0 despite correct math.
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None and method == "strict":
        answer = _extract_boxed_numeric(solution_str)
    if answer is None:
        return 0
    if _answers_equal(answer, ground_truth):
        return score
    # Format present but wrong: keep upstream format_score semantics for ####-style extracts.
    return format_score
