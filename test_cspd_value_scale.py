"""Regression checks for CSPD's {-1, +1} critic-to-success conversion."""

import math

import torch

from verl.trainer.ppo.cspd import build_success_posterior


def test_signed_critic_values_are_mapped_to_success_probabilities():
    behavior_logp = torch.log(torch.tensor([[[0.5, 0.5]]]))
    successor_values = torch.tensor([[[-0.5, 0.5]]])
    state_values = torch.tensor([[0.0]])
    selected = torch.tensor([[True]])

    target, value_weight = build_success_posterior(
        behavior_logp, successor_values, state_values, selected
    )

    torch.testing.assert_close(target, torch.tensor([[[0.25, 0.75, 0.0]]]))
    torch.testing.assert_close(value_weight, torch.tensor([[0.5]]))


def test_out_of_range_predictions_are_clipped_after_affine_mapping():
    behavior_logp = torch.log(torch.tensor([[[0.5, 0.5]]]))
    successor_values = torch.tensor([[[-3.0, 3.0]]])
    state_values = torch.tensor([[1.0]])
    selected = torch.tensor([[True]])

    target, value_weight = build_success_posterior(
        behavior_logp, successor_values, state_values, selected
    )

    assert math.isclose(target.sum().item(), 1.0)
    torch.testing.assert_close(target, torch.tensor([[[0.0, 1.0, 0.0]]]))
    torch.testing.assert_close(value_weight, torch.tensor([[1.0]]))
