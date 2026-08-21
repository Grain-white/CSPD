"""Regression tests for the BCE success critic and two-stage CSPD selection."""

import math

import torch

from verl.trainer.ppo.core_algos import compute_binary_value_loss
from verl.trainer.ppo.cspd import rerank_posterior_candidates


def test_binary_value_loss_uses_logits_and_returns_probabilities():
    logits = torch.zeros((1, 2))
    returns = torch.tensor([[0.0, 1.0]])
    mask = torch.ones_like(returns)

    loss, probabilities, brier, target_oob = compute_binary_value_loss(logits, returns, mask)

    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    torch.testing.assert_close(probabilities, torch.full_like(probabilities, 0.5))
    assert math.isclose(brier.item(), 0.25, rel_tol=1e-6)
    assert target_oob.item() == 0.0


def test_two_stage_selection_retains_posterior_mass_topk():
    candidate_logp = torch.log(torch.tensor([[[0.50, 0.30, 0.15, 0.05]]]))
    candidate_ids = torch.tensor([[[10, 11, 12, 13]]])
    successor_values = torch.tensor([[[0.10, 0.20, 0.90, 1.00]]])
    selected = torch.tensor([[True]])

    logp, ids, values, positions, mass = rerank_posterior_candidates(
        candidate_logp,
        candidate_ids,
        successor_values,
        selected,
        retain_k=2,
        reward_range="01",
    )

    torch.testing.assert_close(mass, torch.tensor([[[0.05, 0.06, 0.135, 0.05]]]))
    torch.testing.assert_close(positions, torch.tensor([[[2, 1]]]))
    torch.testing.assert_close(ids, torch.tensor([[[12, 11]]]))
    torch.testing.assert_close(values, torch.tensor([[[0.90, 0.20]]]))
    torch.testing.assert_close(logp.exp(), torch.tensor([[[0.15, 0.30]]]))


def test_unselected_prefix_keeps_policy_topk_order():
    candidate_logp = torch.log(torch.tensor([[[0.50, 0.30, 0.15, 0.05]]]))
    candidate_ids = torch.tensor([[[10, 11, 12, 13]]])
    successor_values = torch.tensor([[[0.10, 0.20, 0.90, 1.00]]])

    _, ids, _, positions, _ = rerank_posterior_candidates(
        candidate_logp,
        candidate_ids,
        successor_values,
        torch.tensor([[False]]),
        retain_k=2,
        reward_range="01",
    )

    torch.testing.assert_close(positions, torch.tensor([[[0, 1]]]))
    torch.testing.assert_close(ids, torch.tensor([[[10, 11]]]))


def test_baseline_tail_uses_zero_advantage_mass_and_total_weight():
    from verl.trainer.ppo.cspd import build_success_posterior

    behavior_logp = torch.tensor([[[0.4, 0.3]]]).log()
    successor_values = torch.tensor([[[0.8, 0.2]]])
    state_values = torch.tensor([[0.5]])
    selected = torch.tensor([[True]])

    target, weight, diagnostics = build_success_posterior(
        behavior_logp,
        successor_values,
        state_values,
        selected,
        reward_range="01",
        tail_mode="baseline",
        return_diagnostics=True,
    )

    expected_masses = torch.tensor([0.32, 0.06, 0.15])
    expected_total = expected_masses.sum()
    torch.testing.assert_close(target[0, 0], expected_masses / expected_total)
    torch.testing.assert_close(weight[0, 0], expected_total)
    torch.testing.assert_close(diagnostics["implied_tail_success"][0, 0], torch.tensor(0.5))
    assert not diagnostics["tail_projection_mask"][0, 0]
