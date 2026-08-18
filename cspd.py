"""Compatibility imports for the canonical CSPD implementation.

The experiment launcher prepends ``sdpo_verl`` to ``PYTHONPATH``. Keep this
module only so older commands importing ``cspd`` resolve to exactly the same
implementation used by training.
"""

from verl.trainer.ppo.cspd import (
    build_success_posterior,
    cspd_forward_kl,
    make_successor_batch,
    select_prefixes,
)

__all__ = [
    "build_success_posterior",
    "cspd_forward_kl",
    "make_successor_batch",
    "select_prefixes",
]
