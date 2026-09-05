# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[newton] Clear MuJoCo's solver warm start for envs that reset.

MuJoCo seeds each constraint solve from the previous step's accelerations
(``qacc_warmstart``; ``forward.py`` copies ``qacc`` into it after every step and
``solver.py`` starts from it). Nothing on the Isaac Lab reset path touches it, so
an env that is reset after a blow-up starts its very next solve from
accelerations of 1e14 -- and the clean joint state the reset just wrote is torn
apart again within the same step. Measured: root velocity reads exactly 0 (the
reset landed) while joint velocities are back at 1e12 on the same step, every
step, until the NaN reaches the observations and rsl-rl aborts.

Zero ``qacc_warmstart`` (and ``qacc``) for the worlds being reset, right after
Isaac Lab has written their state. Enable with ``AGILE_NEWTON_RESET_WARMSTART=1``.
"""

from __future__ import annotations

import os

import torch

_SENTINEL = "_agile_newton_reset_warmstart_patch_applied"


def clear_solver_warmstart(env_ids, warn_owner=None) -> bool:
    """Zero the per-world solver memory (``qacc_warmstart``, ``qacc``, ``qfrc_constraint``)
    of the given worlds. Used after every env reset and by the fallen-state collector
    after it re-spawns a blown-up env: a world that went non-finite otherwise stays
    non-finite through any number of state writes (measured on Isaac Lab develop:
    one env re-read NaN joints right after each reset until the process ended).

    Returns:
        True if the buffers were cleared.
    """
    try:
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        mjd = NewtonManager._solver.mjw_data
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        if ids.numel() == 0:
            return True
        ids = ids.to(device=wp.device_to_torch(mjd.qacc_warmstart.device), dtype=torch.long)
        for name in ("qacc_warmstart", "qacc", "qfrc_constraint"):
            arr = getattr(mjd, name, None)
            if arr is None:
                continue
            t = wp.to_torch(arr)
            # per-world rows only; flat constraint-row arrays (efc.*) are rebuilt every step
            if t.dim() >= 2 and t.shape[0] >= int(ids.max()) + 1:
                t[ids] = 0.0
        return True
    except Exception as exc:  # never break a reset over this
        if warn_owner is not None and not getattr(warn_owner, "_agile_warmstart_warned", False):
            print(f"[newton] reset warm-start clear failed: {exc}", flush=True)
            warn_owner._agile_warmstart_warned = True
        return False


def apply_newton_reset_warmstart_patch() -> bool:
    if os.environ.get("AGILE_NEWTON_RESET_WARMSTART", "0") != "1":
        return False
    # wrap the most-derived reset the env actually runs (the RL env overrides
    # the base one), and the base as well for non-RL envs
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    if getattr(ManagerBasedRLEnv, _SENTINEL, False):
        return False
    original_reset_idx = ManagerBasedRLEnv._reset_idx

    def _reset_idx_with_warmstart_clear(self, env_ids):
        out = original_reset_idx(self, env_ids)
        clear_solver_warmstart(env_ids, warn_owner=self)
        return out

    ManagerBasedRLEnv._reset_idx = _reset_idx_with_warmstart_clear
    setattr(ManagerBasedRLEnv, _SENTINEL, True)
    print("[newton] MuJoCo warm start cleared on env reset", flush=True)
    return True


apply_newton_reset_warmstart_patch()
