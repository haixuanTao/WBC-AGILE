# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[newton] Give MuJoCo its own joint-limit constraint instead of a stiff spring.

Newton converts a joint limit into MuJoCo's ``solref_limit`` using the negative
"direct" convention ``(-limit_ke, -limit_kd)`` whenever ``limit_ke > 0`` -- and the
builder default is ke=1e4, kd=10: a very stiff, almost undamped spring. A joint
slamming into that at tens of rad/s produces an enormous constraint impulse; the
link is flung, penetrates something else, and the constraint forces run away
(measured: ``qfrc_constraint`` 1e8-1e9, penetration -12 m, NaN within 40 steps
once the post-step velocity clamp is removed).

MuJoCo's native limit constraint uses a time-constant ``solref = (timeconst,
dampratio)``; the default ``(0.02, 1.0)`` is critically damped and stable by
construction. This patch writes that form into ``jnt_solref`` for every joint
Newton had given an explicit spring, and zeroes the Newton-side ``joint_limit_ke``
so the JOINT_DOF_PROPERTIES update kernel does not put the spring back.

Enable with ``AGILE_NEWTON_LIMIT_SOLREF=timeconst,dampratio`` (e.g. ``0.02,1``).
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

_SENTINEL = "_agile_newton_joint_limit_solref_patch_applied"


def apply_newton_joint_limit_solref_patch() -> bool:
    spec = os.environ.get("AGILE_NEWTON_LIMIT_SOLREF", "")
    if not spec:
        return False
    try:
        timeconst, dampratio = (float(x) for x in spec.split(","))
    except ValueError:
        print(f"[newton] AGILE_NEWTON_LIMIT_SOLREF={spec!r} is not 'timeconst,dampratio'; ignored")
        return False
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return False
    if getattr(NewtonManager, _SENTINEL, False):
        return False

    original_initialize_solver = NewtonManager.initialize_solver.__func__

    @classmethod
    def initialize_solver_with_limit_solref(cls):
        original_initialize_solver(cls)
        solver, model = cls._solver, cls._model
        mjw = getattr(solver, "mjw_model", None)
        jnt_map = getattr(solver, "mjc_jnt_to_newton_dof", None)
        if mjw is None or jnt_map is None or not hasattr(mjw, "jnt_solref"):
            print("[newton] limit solref: solver has no jnt_solref / joint map; not applied")
            return
        ke = model.joint_limit_ke.numpy()
        jm = jnt_map.numpy()
        solref = mjw.jnt_solref.numpy()  # (nworld, njnt, 2)
        n = 0
        for w in range(solref.shape[0]):
            for j in range(solref.shape[1]):
                d = int(jm[w, j])
                if d >= 0 and ke[d] > 0.0:
                    solref[w, j] = (timeconst, dampratio)
                    n += 1
        mjw.jnt_solref.assign(wp.array(solref, dtype=wp.vec2, device=mjw.jnt_solref.device))
        # keep the update kernel from restoring the explicit spring
        ke_zero = ke.copy()
        ke_zero[:] = 0.0
        model.joint_limit_ke.assign(wp.array(ke_zero, dtype=wp.float32, device=model.device))
        print(f"[newton] joint-limit solref set to ({timeconst:g}, {dampratio:g}) on {n} joints; explicit limit spring disabled")

    NewtonManager.initialize_solver = initialize_solver_with_limit_solref
    setattr(NewtonManager, _SENTINEL, True)
    return True


apply_newton_joint_limit_solref_patch()
