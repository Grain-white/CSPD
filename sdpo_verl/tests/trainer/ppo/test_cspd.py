import torch

from verl import DataProto

from verl.trainer.ppo.cspd import build_success_posterior, cspd_forward_kl, make_successor_batch, select_prefixes
from verl.utils.reward_score.math_dapo import compute_score


def test_select_prefixes_respects_mask():
    entropy = torch.tensor([[0.1, 0.9, 0.8, 99.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    assert select_prefixes(entropy, mask, 2).tolist() == [[False, True, True, False]]


def test_topk_tail_posterior():
    target, weight = build_success_posterior(
        torch.log(torch.tensor([[[0.4, 0.3]]])), torch.tensor([[[0.6, -0.6]]]),
        torch.tensor([[0.0]]), torch.tensor([[True]]))
    mass = torch.tensor([0.32, 0.06, 0.12])
    torch.testing.assert_close(target[0, 0], mass / mass.sum())
    torch.testing.assert_close(weight, torch.tensor([[0.5]]))


def test_negative_ppo_value_still_has_nonzero_success_weight():
    _, weight = build_success_posterior(
        torch.log(torch.tensor([[[0.4, 0.3]]])), torch.tensor([[[-0.5, -0.5]]]),
        torch.tensor([[-0.5]]), torch.tensor([[True]]))
    torch.testing.assert_close(weight, torch.tensor([[0.25]]))


def test_posterior_diagnostics_expose_projection_and_mass_ratio():
    target, weight, diagnostics = build_success_posterior(
        torch.log(torch.tensor([[[0.4, 0.3]]])),
        torch.tensor([[[1.0, 1.0]]]),
        torch.tensor([[-0.5]]),
        torch.tensor([[True]]),
        return_diagnostics=True,
    )
    torch.testing.assert_close(weight, torch.tensor([[0.25]]))
    assert diagnostics["tail_projection_mask"].item()
    assert diagnostics["posterior_mass"].item() > diagnostics["state_success"].item()
    assert diagnostics["state_to_mass_ratio"].item() < 1.0
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 1))


def test_math_reward_accepts_answer_and_boxed_formats():
    assert compute_score("Therefore, Answer: 42", "42")["acc"]
    assert compute_score(r"Therefore, $\boxed{42}$", "42")["acc"]
    assert compute_score("### Final Answer:\n\n$$\n\\boxed{42}\n$$", "42")["acc"]
    assert compute_score("### Final Answer:\n\n$$\n\\boxed{0.5}\n$$", r"\frac{1}{2}")["acc"]


def test_forward_kl_prefers_matching_distribution():
    target = torch.tensor([[[0.2, 0.3, 0.5]]])
    args = (torch.ones(1, 1), torch.tensor([[True]]))
    matched, _ = cspd_forward_kl(torch.log(target[..., :2]), target, *args)
    shifted, _ = cspd_forward_kl(torch.log(torch.tensor([[[0.4, 0.1]]])), target, *args)
    assert matched < shifted


def test_successor_batch_compacts_left_padding_and_scores_candidate_position():
    batch = DataProto.from_dict(tensors={
        "input_ids": torch.tensor([[0, 0, 11, 12, 21, 22, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0]]),
        "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3, 0]]),
        "responses": torch.tensor([[21, 22, 0]]),
        "response_mask": torch.tensor([[1, 1, 0]]),
    })
    selected = torch.tensor([[False, True, False]])
    candidates = torch.tensor([[[31, 32], [41, 42], [51, 52]]])
    successor, _, _ = make_successor_batch(batch, selected, candidates, pad_token_id=0)
    assert successor.batch["input_ids"].tolist() == [[11, 12, 21, 41, 0], [11, 12, 21, 42, 0]]
    assert successor.batch["response_mask"].tolist() == [[1], [1]]
