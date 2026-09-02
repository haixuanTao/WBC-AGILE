# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reinstate PhysX's DOF velocity clamp on the Newton backend.

AGILE configures ``velocity_limit_sim`` for every G1/T1 joint group. On PhysX
that becomes a hard in-solver clamp -- ``isaaclab_physx``'s articulation calls
``root_view.set_dof_max_velocities()``, which is why a PhysX rollout sits at
*exactly* the configured limit rather than wherever the dynamics would take it.

No Newton solver implements a joint velocity limit: the support matrix in
``newton/solvers.py`` lists ``joint_velocity_limit`` as unsupported for all six
solvers, and ``solver_mujoco.py`` states it outright ("MuJoCo doesn't have
velocity limit"). Switching the backend therefore silently drops a constraint
the task depends on, and the sim diverges -- with substepping alone, runs died
after 1-57 iterations depending on ``num_substeps`` and env count.

This patch clamps joint velocities to the configured limits immediately after
each physics step and writes them back. PhysX clamps *inside* the solve and this
clamps just after it, so the two are not bit-identical -- but both bound the
per-step joint velocity to the same configured limit, which is what the task's
reward and termination terms assume.

Disable with ``AGILE_NEWTON_VEL_CLAMP=0``.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_SENTINEL = "_agile_newton_vel_clamp_applied"
_CACHE_ATTR = "_agile_newton_vel_limit"

# Optional accounting: how much velocity the clamp actually removes.
# Enable with AGILE_NEWTON_VEL_CLAMP_STATS=1 and read via get_clamp_stats().
STATS = {"steps": 0, "elems": 0, "clamped": 0, "removed_abs": 0.0, "max_excess": 0.0}


def get_clamp_stats() -> dict:
    """Snapshot of clamp engagement since process start."""
    s = dict(STATS)
    s["clamped_fraction"] = (s["clamped"] / s["elems"]) if s["elems"] else 0.0
    s["removed_per_step"] = (s["removed_abs"] / s["steps"]) if s["steps"] else 0.0
    return s


def _as_torch(value: Any) -> torch.Tensor | None:
    """Unwrap a ProxyArray / warp array to a torch tensor."""
    if value is None:
        return None
    if hasattr(value, "torch"):
        value = value.torch
    return value if isinstance(value, torch.Tensor) else None


def _velocity_limits(articulation: Any) -> torch.Tensor | None:
    """Per-joint velocity limits, cached on the articulation. Shape (num_envs, num_joints)."""
    cached = getattr(articulation, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    limits = _as_torch(getattr(articulation.data, "joint_vel_limits", None))
    if limits is None or limits.numel() == 0:
        return None
    # A non-finite or non-positive limit means "unconstrained" -> never clamp that joint.
    limits = limits.detach().abs().clone()
    limits[~torch.isfinite(limits)] = float("inf")
    limits[limits <= 0.0] = float("inf")
    setattr(articulation, _CACHE_ATTR, limits)
    return limits


_STATS_ON = False


def apply_newton_joint_velocity_clamp_patch() -> bool:
    """Patch :meth:`InteractiveScene.update` to clamp joint velocities post-step.

    Returns:
        True if the patch was applied, False if it was disabled or already present.
    """
    if os.environ.get("AGILE_NEWTON_VEL_CLAMP", "1") == "0":
        return False

    global _STATS_ON
    _STATS_ON = os.environ.get("AGILE_NEWTON_VEL_CLAMP_STATS", "0") == "1"

    from isaaclab.scene import InteractiveScene

    if getattr(InteractiveScene, _SENTINEL, False):
        return False

    original_update = InteractiveScene.update

    _limit_kd_state = {"applied": False}

    def _apply_limit_damping_once():
        """AGILE_NEWTON_LIMIT_KD: override Newton's joint-limit spring damping (builder
        default kd=10 for ke=1e4 -- under-damped; a limp robot rebounds off its own
        joint limits). Applied once, after the model exists."""
        if _limit_kd_state["applied"]:
            return
        _limit_kd_state["applied"] = True
        kd_env = os.environ.get("AGILE_NEWTON_LIMIT_KD")
        if not kd_env:
            return
        try:
            import warp as wp
            from isaaclab_newton.physics import NewtonManager
            m = NewtonManager.get_model()
            kd = wp.to_torch(m.joint_limit_kd); mask = kd != 0
            kd[mask] = float(kd_env)
            flags = None
            for path in ("newton", "newton._src.solvers.flags", "newton._src.solvers.solver"):
                try:
                    mod = __import__(path, fromlist=["SolverNotifyFlags"]); flags = mod.SolverNotifyFlags; break
                except Exception:
                    continue
            if flags is not None and hasattr(NewtonManager, "_model_changes"):
                NewtonManager._model_changes.add(flags.JOINT_DOF_PROPERTIES)
            print(f"[newton] joint_limit_kd set to {float(kd_env):g} on {int(mask.sum())} dofs")
        except Exception as exc:  # never break the sim over a tuning knob
            print(f"[newton] AGILE_NEWTON_LIMIT_KD not applied: {exc}")

    _prio_state = {"applied": False}

    def _apply_geom_priority_once():
        """AGILE_NEWTON_ROBOT_GEOM_PRIORITY: give every non-terrain geom a MuJoCo
        contact priority so the pair friction takes the ROBOT geom's (randomised) mu.
        MuJoCo's default pair rule is max(mu_a, mu_b) -> with terrain mu=1.0 every foot
        gets friction 1.0; PhysX with AGILE's 'multiply' combine gets 1.0*mu_foot.
        Priority makes the two agree."""
        if _prio_state["applied"]:
            return
        _prio_state["applied"] = True
        pr = os.environ.get("AGILE_NEWTON_ROBOT_GEOM_PRIORITY")
        if not pr:
            return
        try:
            import gc, numpy as np, warp as wp, mujoco
            solver = next((o for o in gc.get_objects() if type(o).__name__ == "SolverMuJoCo"), None)
            mj, mw = solver.mj_model, solver.mjw_model
            names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(mj.ngeom)]
            prio = wp.to_torch(mw.geom_priority)          # (nworld, ngeom)
            idx = [i for i, n in enumerate(names) if "terrain" not in n.lower() and "ground" not in n.lower()]
            prio[:, idx] = int(pr)
            mj.geom_priority[idx] = int(pr)
            print(f"[newton] geom_priority={pr} on {len(idx)} robot geoms (terrain stays 0)")
        except Exception as exc:
            print(f"[newton] AGILE_NEWTON_ROBOT_GEOM_PRIORITY not applied: {exc}")

    def update_with_velocity_clamp(self, dt: float):
        _apply_geom_priority_once()
        _apply_limit_damping_once()
        original_update(self, dt)
        for articulation in self.articulations.values():
            limits = _velocity_limits(articulation)
            if limits is None:
                continue
            joint_vel = _as_torch(articulation.data.joint_vel)
            if joint_vel is None:
                continue
            clamped = torch.clamp(joint_vel, -limits, limits)
            if _STATS_ON:
                excess = (joint_vel.abs() - limits).clamp_(min=0.0)
                finite = torch.isfinite(excess)
                excess = torch.where(finite, excess, torch.zeros_like(excess))
                STATS["steps"] += 1
                STATS["elems"] += excess.numel()
                STATS["clamped"] += int((excess > 0).sum())
                STATS["removed_abs"] += float(excess.sum())
                STATS["max_excess"] = max(STATS["max_excess"], float(excess.max()))
                # periodic engagement report: how often the guard actually fires
                if STATS["steps"] % 500 == 0:
                    frac = STATS["clamped"] / max(STATS["elems"], 1)
                    print(f"[clamp-stats] steps={STATS['steps']} engaged on {100*frac:.4f}% of joint samples, "
                          f"max excess so far {STATS['max_excess']:.1f} rad/s", flush=True)
            # `_mask` (not `_index`) is the graph-capture-safe write path, and the
            # Newton config runs with use_cuda_graph=True.
            articulation.write_joint_velocity_to_sim_mask(velocity=clamped.contiguous())

    InteractiveScene.update = update_with_velocity_clamp
    setattr(InteractiveScene, _SENTINEL, True)
    return True
