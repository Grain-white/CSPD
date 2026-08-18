"""Compatibility-level regression checks for the root ``cspd`` import."""

import torch

from cspd import build_success_posterior


def test_root_import_uses_signed_critic_value_scale():
    behavior_logp = torch.log(torch.tensor([[[0.25, 0.25]]]))
    successor_values = torch.tensor([[[-1.0, 1.0]]])
    state_values = torch.tensor([[0.0]])
    selected_mask = torch.tensor([[True]])

    target, value_weight = build_success_posterior(
        behavior_logp,
        successor_values,
        state_values,
        selected_mask,
    )

    torch.testing.assert_close(target, torch.tensor([[[0.0, 0.5, 0.5]]]))
    torch.testing.assert_close(value_weight, torch.tensor([[0.5]]))
