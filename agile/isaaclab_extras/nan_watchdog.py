# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the first non-finite simulation quantity, with one step of history.

Enable with ``AGILE_NAN_WATCHDOG=1``. Wraps ``ManagerBasedRLEnv.step`` and checks
the articulation state after every env step. On the first non-finite value it
prints which quantity went first, which env, and that env's state on the
*previous* step -- which is what says whether the divergence came in through the
joints, the floating base, or the contacts.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_SENTINEL = "_agile_nan_watchdog_applied"

_FIELDS = (
    "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
    "joint_pos", "joint_vel", "joint_acc", "applied_torque",
)


def _as_torch(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if hasattr(value, "torch"):
        value = value.torch
    return value if isinstance(value, torch.Tensor) else None


def apply_nan_watchdog_patch() -> bool:
    if os.environ.get("AGILE_NAN_WATCHDOG", "0") != "1":
        return False

    from isaaclab.envs import ManagerBasedRLEnv

    if getattr(ManagerBasedRLEnv, _SENTINEL, False):
        return False

    original_step = ManagerBasedRLEnv.step
    state = {"n": 0, "prev": None}

    def step_with_watchdog(self, action):
        out = original_step(self, action)
        state["n"] += 1

        robot = self.scene["robot"]
        snap = {}
        first_bad = None
        for name in _FIELDS:
            t = _as_torch(getattr(robot.data, name, None))
            if t is None:
                continue
            snap[name] = t.detach().clone()
            if first_bad is None and not torch.isfinite(t).all():
                first_bad = name

        # contact sensor, if present
        sensor = self.scene.sensors.get("contact_forces") if hasattr(self.scene, "sensors") else None
        if sensor is not None:
            cf = _as_torch(getattr(sensor.data, "net_forces_w", None))
            if cf is not None:
                snap["contact_net_forces_w"] = cf.detach().clone()
                if first_bad is None and not torch.isfinite(cf).all():
                    first_bad = "contact_net_forces_w"

        if first_bad is not None:
            bad = ~torch.isfinite(snap[first_bad])
            envs = bad.any(dim=tuple(range(1, bad.ndim))) if bad.ndim > 1 else bad
            e = int(envs.nonzero()[0])
            print("\n" + "=" * 78, flush=True)
            print(f"[watchdog] FIRST NON-FINITE at env-step {state['n']}", flush=True)
            print(f"[watchdog] leading quantity: {first_bad}", flush=True)
            print(f"[watchdog] envs affected: {int(envs.sum())} / {envs.numel()}   first env: {e}", flush=True)
            print(f"[watchdog] episode_length_buf[{e}] = {int(self.episode_length_buf[e])} "
                  f"(0 means it reset this step)", flush=True)
            print("\n[watchdog] all quantities THIS step (env %d):" % e, flush=True)
            for k, v in snap.items():
                fin = "finite" if torch.isfinite(v[e]).all() else "NON-FINITE"
                print(f"    {k:24s} {fin:11s} absmax={v[e][torch.isfinite(v[e])].abs().max().item()
                      if torch.isfinite(v[e]).any() else float('nan'):.4g}", flush=True)
            prev = state["prev"]
            if prev is not None:
                print(f"\n[watchdog] PREVIOUS step (env {e}) -- the last good state:", flush=True)
                for k, v in prev.items():
                    if e < v.shape[0]:
                        vv = v[e]
                        print(f"    {k:24s} absmax={vv.abs().max().item():.4g}", flush=True)
            print("=" * 78 + "\n", flush=True)

        state["prev"] = snap
        return out

    ManagerBasedRLEnv.step = step_with_watchdog
    setattr(ManagerBasedRLEnv, _SENTINEL, True)
    return True
