# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[newton] Put the DC-motor torque-speed curve back, inside the solver.

AGILE's ``DelayedDCMotor`` limits the torque a joint can produce as a function of
its speed -- a linear four-quadrant curve that reaches zero torque at
``velocity_limit`` -- and that curve is *the* physical mechanism that stops a
motor from over-speeding. Switching Newton to solver-side (implicit) PD
actuators dropped it: the drive got a flat ``effort_limit`` clip and nothing
else, and a post-step velocity clamp was bolted on to keep joints under their
rated speed. Overwriting a velocity after the solve does not conserve momentum;
it removes energy for free, every step, and a policy trained against it learns
to lean on a brake no real robot has.

MuJoCo already has the right primitive: ``jnt_actfrcrange`` clamps the *total*
actuator force on a joint, per world, per step, and MuJoCo-Warp evaluates it in
``fwd_actuation``. Newton fills it with ``(-effort_limit, +effort_limit)``. This
patch rewrites it every physics step from the DC curve at the joint's current
velocity::

    cap(qd) = clip(sat * (1 - |qd| / vel), 0, eff),   range = [-cap, +cap]

the DC curve's available torque at the joint's speed, applied symmetrically
(see the kernel for why the four-quadrant form is unsafe with a one-step lag). It runs as a Newton post-actuator callback so it lives inside the
captured CUDA graph, before the substeps.

Enable with ``AGILE_NEWTON_DC_ENVELOPE=1``; pair it with
``AGILE_NEWTON_VEL_CLAMP=0`` so the clamp is not also in the loop. Saturation
torques per joint group come from the explicit ``DelayedDCMotorCfg`` groups in
``unitree_g1.py`` (``saturation_effort``); effort and velocity limits are read
from the Newton model, so they follow whatever the asset configured.
"""

from __future__ import annotations

import os
import re

import numpy as np
import warp as wp

_SENTINEL = "_agile_newton_dc_envelope_patch_applied"

# saturation_effort of the G1 height-tracking DelayedDCMotorCfg groups
# (AGILE_NEWTON_IMPLICIT_ACTUATORS=0 view of unitree_g1.py). Joints matching no
# group keep their plain effort-limit range.
_SATURATION_GROUPS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (180.0, (r".*_hip_.*_joint", r".*_knee_joint")),
    (80.0, (r".*_ankle_.*_joint",)),
    (120.0, (r"waist_.*_joint",)),
    (130.0, (r".*_shoulder_.*_joint", r".*_elbow_joint", r".*_wrist_.*_joint")),
)


@wp.kernel(enable_backward=False)
def _dc_envelope_kernel(
    mjc_jnt_to_newton_dof: wp.array2d(dtype=wp.int32),
    jnt_dofadr: wp.array(dtype=wp.int32),
    joint_qd: wp.array(dtype=wp.float32),
    sat: wp.array(dtype=wp.float32),
    eff: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.float32),
    base_damping: wp.array2d(dtype=wp.float32),
    vel_limit_gain: float,
    # outputs
    jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
    dof_damping: wp.array2d(dtype=wp.float32),
):
    world, jnt = wp.tid()
    dof = mjc_jnt_to_newton_dof[world, jnt]
    if dof < 0:
        return
    s = sat[dof]
    if s <= 0.0:
        return
    e = eff[dof]
    v = vel[dof]
    if v <= 0.0 or e <= 0.0:
        return
    qd = wp.abs(joint_qd[dof])
    # Drive side (symmetric, lag-safe): the motor cannot push a joint past its
    # rated speed. Both directions capped by the curve at |qd|; the
    # four-quadrant form lags one step and feeds a sign-alternating oscillation.
    cap = wp.clamp(s * (1.0 - qd / v), 0.0, e)
    jnt_actfrcrange[world, jnt] = wp.vec2(-cap, cap)
    # Brake side: above no-load speed a DC motor is a generator, braking with a
    # torque that rises linearly to the effort limit. Expressed as passive
    # damping so MuJoCo integrates it implicitly -- it always opposes the
    # *current* velocity and cannot go unstable. This is what stops a joint
    # from running away once the drive cap has gone to zero.
    md = jnt_dofadr[jnt]
    brake = 0.0
    if qd > v:
        brake = wp.min(s * (qd / v - 1.0), e) / qd
        # PhysX enforces `velocity_limit_sim` unconditionally, in-solver. The
        # generator brake above saturates at the effort limit, and a joint that
        # is still being driven past its rated speed by the rest of the chain
        # can end up a dozen turns outside its range, where MuJoCo's soft limit
        # asks for an acceleration proportional to the violation and the solve
        # goes non-finite. Add a damping band that rises steeply above the
        # rated speed -- implicit, so any magnitude is stable -- which is the
        # faithful port of PhysX's joint velocity limit.
        if vel_limit_gain > 0.0:
            band = vel_limit_gain * e / v * (qd - v) / qd
            brake = wp.max(brake, band)
    dof_damping[world, md] = base_damping[world, md] + brake


def _joint_name(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _build_dof_params(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-Newton-DOF (sat, eff, vel) arrays; returns how many DOFs got a curve."""
    n = int(model.joint_dof_count)
    sat = np.zeros(n, dtype=np.float32)
    eff = model.joint_effort_limit.numpy().astype(np.float32)
    vel = model.joint_velocity_limit.numpy().astype(np.float32)
    qd_start = model.joint_qd_start.numpy()
    dof_dim = model.joint_dof_dim.numpy()
    matched = 0
    for j, label in enumerate(model.joint_label):
        name = _joint_name(label)
        s = 0.0
        for value, patterns in _SATURATION_GROUPS:
            if any(re.fullmatch(p, name) for p in patterns):
                s = value
                break
        if s <= 0.0:
            continue
        start = int(qd_start[j])
        width = int(dof_dim[j][0] + dof_dim[j][1])
        sat[start : start + width] = s
        matched += width
    return sat, eff, vel, matched


def apply_newton_dc_motor_envelope_patch() -> bool:
    if os.environ.get("AGILE_NEWTON_DC_ENVELOPE", "0") != "1":
        return False
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return False
    if getattr(NewtonManager, _SENTINEL, False):
        return False

    original_initialize_solver = NewtonManager.initialize_solver.__func__

    @classmethod
    def initialize_solver_with_envelope(cls):
        original_initialize_solver(cls)
        solver = cls._solver
        model = cls._model
        mjw = getattr(solver, "mjw_model", None)
        jnt_map = getattr(solver, "mjc_jnt_to_newton_dof", None)
        if mjw is None or jnt_map is None or not hasattr(mjw, "jnt_actfrcrange"):
            print("[newton] DC envelope: solver has no jnt_actfrcrange / joint map; not applied")
            return
        sat, eff, vel, matched = _build_dof_params(model)
        device = model.device
        sat_wp = wp.array(sat, dtype=wp.float32, device=device)
        eff_wp = wp.array(eff, dtype=wp.float32, device=device)
        vel_wp = wp.array(vel, dtype=wp.float32, device=device)
        nworld, njnt = mjw.jnt_actfrcrange.shape
        mjd = getattr(solver, "mjw_data", None)
        if mjd is not None:  # ground truth for the solver's constraint/contact capacity (overflow => dropped constraints)
            print(f"[newton] solver capacity: nworld={nworld} njmax={int(mjd.njmax)} naconmax={int(mjd.naconmax)}"
                  f" (~{int(mjd.naconmax) // max(nworld, 1)} contacts/world)", flush=True)
        if not hasattr(mjw, "jnt_dofadr") or not hasattr(mjw, "dof_damping"):
            print("[newton] DC envelope: solver has no jnt_dofadr / dof_damping; not applied")
            return
        base_damping = wp.clone(mjw.dof_damping)   # Newton's own passive damping, kept underneath the brake
        # AGILE_NEWTON_VEL_LIMIT_GAIN: torque per unit over-speed at 2x rated speed, in
        # multiples of the effort limit (0 disables the band; 50 = a hard limit in practice)
        vel_limit_gain = float(os.environ.get("AGILE_NEWTON_VEL_LIMIT_GAIN", "50"))

        def _apply_envelope():
            wp.launch(
                _dc_envelope_kernel,
                dim=(nworld, njnt),
                inputs=[jnt_map, mjw.jnt_dofadr, cls._state_0.joint_qd, sat_wp, eff_wp, vel_wp, base_damping, vel_limit_gain],
                outputs=[mjw.jnt_actfrcrange, mjw.dof_damping],
                device=device,
            )

        cls.register_post_actuator_callback(_apply_envelope)
        print(
            f"[newton] DC-motor torque-speed envelope active on {matched}/{model.joint_dof_count} DOFs "
            f"({nworld} worlds x {njnt} joints) via jnt_actfrcrange + generator braking via dof_damping"
            + (f", velocity-limit band gain {vel_limit_gain:g}" if vel_limit_gain > 0 else "")
        )

    NewtonManager.initialize_solver = initialize_solver_with_envelope
    setattr(NewtonManager, _SENTINEL, True)
    return True


apply_newton_dc_motor_envelope_patch()
