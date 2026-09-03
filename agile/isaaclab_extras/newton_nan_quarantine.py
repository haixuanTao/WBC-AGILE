# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[newton] Quarantine an env whose state went non-finite instead of aborting.

MuJoCo-Warp's constraint solve very occasionally returns NaN from an ordinary
looking state -- measured on the 10k run: one env in 4096, after 7043 iterations,
from a step with joint speeds under 5 rad/s and a 2.9 kN contact. rsl-rl aborts
the whole run on any NaN in the observations. One env in ~7e8 env-steps is not
worth a run; reset that env, mark it done, and hand rsl-rl finite numbers.

Hooks ``RslRlVecEnvWrapper.step``. After each step, any env with a non-finite
robot state or observation is reset through the env's own ``_reset_idx`` (so the
usual reset events run and, with AGILE_NEWTON_RESET_WARMSTART, the solver warm
start is cleared), its observation rows are replaced with zeros, its reward
with zero and its done flag set. The event is logged with a running count so the
rate stays visible. Enable with ``AGILE_NEWTON_NAN_QUARANTINE=1``.
"""

from __future__ import annotations

import os

import torch

_SENTINEL = "_agile_newton_nan_quarantine_applied"
COUNT = {"events": 0, "envs": 0}


def _as_torch(v):
    if v is None:
        return None
    if hasattr(v, "torch"):
        v = v.torch
    return v if isinstance(v, torch.Tensor) else None


def apply_newton_nan_quarantine_patch() -> bool:
    if os.environ.get("AGILE_NEWTON_NAN_QUARANTINE", "0") != "1":
        return False
    try:
        from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    except Exception:
        return False
    if getattr(RslRlVecEnvWrapper, _SENTINEL, False):
        return False
    original_step = RslRlVecEnvWrapper.step

    def step_with_quarantine(self, actions):
        obs, rew, dones, extras = original_step(self, actions)
        env = self.unwrapped
        robot = env.scene["robot"]
        bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"):
            t = _as_torch(getattr(robot.data, name, None))
            if t is not None:
                bad |= ~torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1)
        groups = list(obs.keys()) if hasattr(obs, "keys") else []
        for g in groups:
            t = _as_torch(obs[g])
            if t is not None and t.shape[0] == env.num_envs:
                bad |= ~torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1)
        if bad.any():
            ids = bad.nonzero().flatten()
            COUNT["events"] += 1
            COUNT["envs"] += int(ids.numel())
            print(f"[nan-quarantine] event {COUNT['events']}: {int(ids.numel())} env(s) non-finite "
                  f"(first {int(ids[0])}), reset and masked; total envs so far {COUNT['envs']}", flush=True)
            env._reset_idx(ids)
            # the reset events may not write joint state (no dataset attached, or an
            # event that returned early) -- make sure the env leaves with finite state
            n = ids.numel()
            rp = _as_torch(robot.data.default_root_state)[ids, :7].clone()
            rp[:, :3] += env.scene.env_origins[ids]
            robot.write_root_pose_to_sim(rp, env_ids=ids)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
            jp = _as_torch(robot.data.default_joint_pos)[ids].clone()
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
            for g in groups:
                t = _as_torch(obs[g])
                if t is not None and t.shape[0] == env.num_envs:
                    t[ids] = 0.0
            rew = rew.clone(); rew[ids] = 0.0
            dones = dones.clone(); dones[ids] = True
            # any NaN that leaked into the extras' time-out flag
            to = extras.get("time_outs") if isinstance(extras, dict) else None
            if isinstance(to, torch.Tensor) and to.shape[0] == env.num_envs:
                to[ids] = False
        return obs, rew, dones, extras

    RslRlVecEnvWrapper.step = step_with_quarantine
    setattr(RslRlVecEnvWrapper, _SENTINEL, True)
    print("[newton] NaN quarantine armed on RslRlVecEnvWrapper.step", flush=True)
    return True


apply_newton_nan_quarantine_patch()
