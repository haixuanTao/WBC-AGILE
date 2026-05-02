# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Single-axis robustness sweeps for sim-to-sim evaluation.

Each axis belongs to one of four categories (see --list-axes for the full menu):

    dr           Domain randomization  -- mass, friction, damping
    actuator     Control gains/limits  -- kp_scale, kd_scale, torque_cap_scale
    init_state   Pose at rollout start -- base_yaw/pitch/roll/height, joint_noise
    disturbance  In-loop pushes        -- push

Examples:
    # List every axis grouped by category:
    python scripts/sim2mujoco_sweep.py --list-axes

    # Run an entire category (recommended thematic sweep):
    python scripts/sim2mujoco_sweep.py --checkpoint <ckpt> --config <cfg> \
        --mjcf <scene.xml> --category dr --output /tmp/dr.json

    # Run all 12 axes in one shot:
    python scripts/sim2mujoco_sweep.py --checkpoint <ckpt> --config <cfg> \
        --mjcf <scene.xml> --category all --output /tmp/all.json

    # Custom values on a single axis (still ad-hoc):
    python scripts/sim2mujoco_sweep.py --checkpoint <ckpt> --config <cfg> \
        --mjcf <scene.xml> --axes base_height_dz_m \
        --values 0,0.3,0.5,0.55,0.6,0.8 --n-trials 16 --duration 48 \
        --output /tmp/height_long.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sim2mujoco_watcher import (  # noqa: E402
    SCHEDULES,
    PolicyWrapper,
    _init_base_pose,
    _run_single_rollout,
    build_once,
)

# ---------------------------------------------------------------------------
# Axis dispatch — one applier per family, params per axis (Isaac EventTerm style).
# ---------------------------------------------------------------------------

_AXIS_FIELDS = {
    "body_mass": "body_mass",
    "geom_friction": "geom_friction",
    "dof_damping": "dof_damping",
    "actuator_gear": "actuator_gear",
}


def scale_field(sim, value: float, originals: dict, *, field: str) -> None:
    """Multiply a snapshotted mj_model array by ``value``."""
    getattr(sim.mj_model, field)[...] = originals[field] * value
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def scale_kp(sim, value: float, originals: dict) -> None:
    """Position actuator kp lives in gain[0] and bias[1]; scale them together."""
    sim.mj_model.actuator_gainprm[:, 0] = originals["actuator_gainprm"][:, 0] * value
    sim.mj_model.actuator_biasprm[:, 1] = originals["actuator_biasprm"][:, 1] * value
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def scale_kd(sim, value: float, originals: dict) -> None:
    """Actuator velocity gain (kv) lives in bias[2]."""
    sim.mj_model.actuator_biasprm[:, 2] = originals["actuator_biasprm"][:, 2] * value
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def init_orient_deg(sim, value: float, *, axis: str, rng: np.random.Generator) -> float:
    """Initial-state init_fn: rotate the base by ``value`` degrees about ``axis``."""
    pelvis_z = _init_base_pose(sim, init_quat_wxyz=ROBOT_CFG.init_quat_wxyz)
    t = float(value) * np.pi / 180.0
    if axis == "yaw":
        sim.mj_data.qpos[3:7] = [np.cos(t / 2), 0.0, 0.0, np.sin(t / 2)]
    elif axis == "pitch":
        sim.mj_data.qpos[3:7] = [np.cos(t / 2), 0.0, np.sin(t / 2), 0.0]
    elif axis == "roll":
        sim.mj_data.qpos[3:7] = [np.cos(t / 2), np.sin(t / 2), 0.0, 0.0]
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return pelvis_z


def init_height(sim, value: float, *, rng: np.random.Generator) -> float:
    pelvis_z = _init_base_pose(sim, init_quat_wxyz=ROBOT_CFG.init_quat_wxyz)
    sim.mj_data.qpos[2] += float(value)
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return pelvis_z


def init_joint_noise(sim, value: float, *, rng: np.random.Generator) -> float:
    pelvis_z = _init_base_pose(sim, init_quat_wxyz=ROBOT_CFG.init_quat_wxyz)
    if value > 0:
        n = sim.mj_model.nq - 7
        sim.mj_data.qpos[7 : 7 + n] += rng.normal(0.0, float(value), size=n)
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return pelvis_z


def push_hook(magnitude: float) -> Callable:
    """Build a watcher-compatible push hook with a fixed magnitude.

    The push axis runs through the watcher's hard-eval gate, which couples
    push impulses with DR randomization. So a push-axis sweep tests the
    policy under coupled (push + DR) — same semantics as --hard-eval but
    with push magnitude scanned across the axis values.
    """
    times = (3.0, 6.0, 9.0)

    def _hook(sim, rng, t_now, t_prev):
        if magnitude <= 0:
            return
        for pt in times:
            if t_prev <= pt < t_now:
                angle = rng.uniform(0, 2 * np.pi)
                sim.mj_data.qvel[0] += magnitude * np.cos(angle)
                sim.mj_data.qvel[1] += magnitude * np.sin(angle)

    return _hook


# ---------------------------------------------------------------------------
# Robot/task config — values that are robot-specific (init quat, tracking body
# name, fall-height threshold) or task-specific (command schedules). Defaults
# match the LeRobot velocity policy used in this branch; override via
# --robot-config <PATH>.
# ---------------------------------------------------------------------------


@dataclass
class RobotCfg:
    init_quat_wxyz: tuple[float, float, float, float] = (0.9946, 0.0523, 0.0872, 0.0)
    tracking_body_name: str = "torso_subassembly"
    fall_height_m: float = 0.35
    # If None, the watcher's module-level SCHEDULES is used.
    schedules: dict[str, list] | None = None


ROBOT_CFG = RobotCfg()


def load_robot_config(path: Path) -> None:
    """Update the module-global ROBOT_CFG from a JSON file. Unset keys keep defaults."""
    d = json.loads(path.read_text())
    if "init_quat_wxyz" in d:
        q = d["init_quat_wxyz"]
        if len(q) != 4:
            raise ValueError(f"init_quat_wxyz must be 4 floats, got {q}")
        ROBOT_CFG.init_quat_wxyz = tuple(float(x) for x in q)
    if "tracking_body_name" in d:
        ROBOT_CFG.tracking_body_name = str(d["tracking_body_name"])
    if "fall_height_m" in d:
        ROBOT_CFG.fall_height_m = float(d["fall_height_m"])
    if "schedules" in d:
        ROBOT_CFG.schedules = {k: [tuple(s) for s in v] for k, v in d["schedules"].items()}


# ---------------------------------------------------------------------------
# Axis configs (Isaac EventTerm-style: func + params).
# ---------------------------------------------------------------------------


@dataclass
class AxisCfg:
    """Declarative spec for one perturbation axis (Isaac EventTerm-style)."""

    description: str = ""
    training_range: tuple[float, float] | None = None
    default_values: tuple[float, ...] = ()
    # `func` is dispatched in `run_sweep`; either a model-side scale or an init_fn factory.
    func: Callable | None = None
    params: dict = field(default_factory=dict)
    # `kind` selects how the applier is called: "model" (scale_*), "init" (per-rollout init_fn),
    # or "push" (in-loop hook).
    kind: str = "model"
    # `category` groups axes for thematic sweeps and report layout.
    category: str = "other"


# Display order for categories — used by the report and by --category=all.
CATEGORY_ORDER = ("dr", "actuator", "init_state", "disturbance")
CATEGORY_LABELS = {
    "dr": "Domain Randomization (sim-parameter scales)",
    "actuator": "Actuator (control gains and limits)",
    "init_state": "Initial state (pose / joint perturbations at rollout start)",
    "disturbance": "Disturbance (in-loop pushes)",
}


AXES: dict[str, AxisCfg] = {
    # --- Domain randomization (DR): sim-parameter scales randomized in training ---
    "mass": AxisCfg(
        category="dr",
        description="Body mass scale",
        training_range=(0.85, 1.15),
        default_values=(0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0),
        func=scale_field,
        params={"field": "body_mass"},
    ),
    "friction": AxisCfg(
        category="dr",
        description="Geom friction scale",
        training_range=(0.5, 2.0),
        default_values=(0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5),
        func=scale_field,
        params={"field": "geom_friction"},
    ),
    "damping": AxisCfg(
        category="dr",
        description="Joint dof_damping scale",
        training_range=(0.8, 2.0),
        default_values=(0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5),
        func=scale_field,
        params={"field": "dof_damping"},
    ),
    # --- Actuator: deploy-time control gains / torque limits ---
    "kp_scale": AxisCfg(
        category="actuator",
        description="Actuator stiffness scale",
        training_range=(0.9, 1.1),
        default_values=(0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5),
        func=scale_kp,
    ),
    "kd_scale": AxisCfg(
        category="actuator",
        description="Actuator velocity-gain scale",
        training_range=(0.8, 2.0),
        default_values=(0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
        func=scale_kd,
    ),
    "torque_cap_scale": AxisCfg(
        category="actuator",
        description="Actuator gear scale (caps motor torque)",
        training_range=None,
        default_values=(0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0),
        func=scale_field,
        params={"field": "actuator_gear"},
    ),
    # --- Initial state: pose / joint perturbations at rollout start ---
    "base_yaw_deg": AxisCfg(
        category="init_state",
        description="Initial base yaw rotation",
        training_range=(-180.0, 180.0),
        default_values=(-180, -90, -60, -30, 0, 30, 60, 90, 180),
        func=init_orient_deg,
        params={"axis": "yaw"},
        kind="init",
    ),
    "base_pitch_deg": AxisCfg(
        category="init_state",
        description="Initial base pitch",
        training_range=(-20.0, 20.0),
        default_values=(-30, -15, -7, 0, 7, 15, 30),
        func=init_orient_deg,
        params={"axis": "pitch"},
        kind="init",
    ),
    "base_roll_deg": AxisCfg(
        category="init_state",
        description="Initial base roll",
        training_range=(-20.0, 20.0),
        default_values=(-30, -15, -7, 0, 7, 15, 30),
        func=init_orient_deg,
        params={"axis": "roll"},
        kind="init",
    ),
    "base_height_dz_m": AxisCfg(
        category="init_state",
        description="Initial base height offset",
        training_range=None,
        default_values=(-0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        func=init_height,
        kind="init",
    ),
    "joint_noise_std": AxisCfg(
        category="init_state",
        description="Std of gaussian noise on initial joint angles",
        training_range=None,
        default_values=(0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30),
        func=init_joint_noise,
        kind="init",
    ),
    # --- Disturbance: in-loop perturbations ---
    "push": AxisCfg(
        category="disturbance",
        description="Magnitude of horizontal velocity injection at t=3,6,9 s",
        training_range=(0.0, 0.5),
        default_values=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5),
        kind="push",
    ),
}


def axes_in_category(category: str) -> list[str]:
    """Return axis names belonging to one category, in declaration order."""
    return [name for name, cfg in AXES.items() if cfg.category == category]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


_SNAPSHOT_FIELDS = (
    "body_mass",
    "geom_friction",
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_gear",
)


def _snapshot(sim) -> dict:
    return {f: getattr(sim.mj_model, f).copy() for f in _SNAPSHOT_FIELDS}


def _restore(sim, originals: dict) -> None:
    for f, v in originals.items():
        getattr(sim.mj_model, f)[...] = v
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def _wrap_init(
    axis_init: Callable | None,
    rng: np.random.Generator,
    noise_std: float,
) -> Callable:
    """Combine an optional axis init_fn with per-trial joint-angle noise."""

    def _fn(sim):
        pelvis_z = (
            axis_init(sim) if axis_init is not None else _init_base_pose(sim, init_quat_wxyz=ROBOT_CFG.init_quat_wxyz)
        )
        if noise_std > 0:
            n = sim.mj_model.nq - 7
            sim.mj_data.qpos[7 : 7 + n] += rng.normal(0.0, noise_std, size=n)
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
        return pelvis_z

    return _fn


def _aggregate(value: float, rollouts: list[dict]) -> dict:
    n = len(rollouts)
    by_sched = {sn: [r for r in rollouts if r["sched"] == sn] for sn in {r["sched"] for r in rollouts}}
    return {
        "value": float(value),
        "n_rollouts": n,
        "fall_rate": sum(r["fell"] for r in rollouts) / n,
        "mean_survival_s": sum(r["survived_s"] for r in rollouts) / n,
        "mean_base_height": sum(r["mean_base_height"] for r in rollouts) / n,
        "per_schedule": {
            sn: {
                "fall_rate": sum(r["fell"] for r in these) / len(these),
                "mean_survival_s": sum(r["survived_s"] for r in these) / len(these),
            }
            for sn, these in by_sched.items()
        },
        "rollouts": rollouts,
    }


def run_sweep(
    *,
    checkpoint: Path,
    config_path: Path,
    mjcf_path: Path,
    axes: list[str],
    values_override: list[float] | None,
    n_trials: int,
    duration_s: float,
    trial_joint_noise_std: float,
    fall_height_m: float,
    device_str: str = "cpu",
    output_path: Path | None = None,
) -> dict:
    """Run requested axis sweeps and return (also write) a result dict."""
    from scripts import sim2mujoco_watcher as wmod

    wmod.SCHEDULE_DURATION_S = duration_s
    device = torch.device(device_str)
    config, sim, obs_proc, act_proc, cmd_prov = build_once(config_path, mjcf_path, device)
    policy = PolicyWrapper.from_config(checkpoint, config, device)
    originals = _snapshot(sim)
    orig_push = wmod._hard_eval_maybe_push

    out = {
        "checkpoint": str(checkpoint),
        "n_trials": n_trials,
        "duration_s": duration_s,
        "trial_joint_noise_std": trial_joint_noise_std,
        "axes": {},
    }

    tmp = Path(tempfile.mkdtemp(prefix="sim2mujoco_sweep_"))
    t0 = time.time()
    try:
        for axis_name in axes:
            cfg = AXES[axis_name]
            values = values_override if values_override is not None else list(cfg.default_values)
            cells: dict[str, dict] = {}

            for value in values:
                _restore(sim, originals)
                if cfg.kind == "model":
                    cfg.func(sim, float(value), originals, **cfg.params)
                elif cfg.kind == "push":
                    wmod._hard_eval_maybe_push = push_hook(float(value))

                schedules = ROBOT_CFG.schedules if ROBOT_CFG.schedules is not None else SCHEDULES
                rollouts: list[dict] = []
                for trial in range(n_trials):
                    for sn, sched in schedules.items():
                        seed = (
                            (1234 + trial * 31)
                            ^ (hash(sn) & 0xFFFFFFFF)
                            ^ (hash(axis_name) & 0xFFFFFFFF)
                            ^ int(float(value) * 10000)
                        ) & 0xFFFFFFFF
                        rng = np.random.default_rng(seed)
                        axis_init = (
                            (lambda _sim, v=value, p=cfg.params, r=rng, fn=cfg.func: fn(_sim, float(v), rng=r, **p))
                            if cfg.kind == "init"
                            else None
                        )
                        init_fn = _wrap_init(axis_init, rng, trial_joint_noise_std)
                        push_active = cfg.kind == "push" and float(value) > 0
                        result = _run_single_rollout(
                            sim=sim,
                            obs_processor=obs_proc,
                            act_processor=act_proc,
                            command_provider=cmd_prov,
                            policy=policy,
                            schedule_name=sn,
                            schedule=sched,
                            video_path=tmp / f"{axis_name}_{value}_{sn}_t{trial}.mp4",
                            fall_height_m=fall_height_m,
                            init_fn=init_fn,
                            hard_eval_originals=originals if push_active else None,
                            hard_eval_rng=rng if push_active else None,
                            tracking_body_name=ROBOT_CFG.tracking_body_name,
                        )
                        rollouts.append(
                            {
                                "sched": sn,
                                "trial": trial,
                                "survived_s": result.survival_time_s,
                                "fell": bool(result.fell),
                                "mean_base_height": float(result.mean_base_height),
                            }
                        )

                cell = _aggregate(value, rollouts)
                cells[str(value)] = cell
                print(
                    f"[sweep] {axis_name:18s} val={float(value):>7.3f} N={cell['n_rollouts']:3d} "
                    f"fall={cell['fall_rate']:.3f} surv={cell['mean_survival_s']:5.2f}s "
                    f"base_h={cell['mean_base_height']:.2f}"
                )

            wmod._hard_eval_maybe_push = orig_push
            out["axes"][axis_name] = {
                "category": cfg.category,
                "training_range": list(cfg.training_range) if cfg.training_range else None,
                "description": cfg.description,
                "values": [float(v) for v in values],
                "results": cells,
            }
            # Incremental save: persist after every axis so a mid-sweep crash
            # leaves us with usable partial data instead of nothing.
            if output_path is not None:
                out["wall_seconds"] = time.time() - t0
                output_path.write_text(json.dumps(out, indent=2))

        out["wall_seconds"] = time.time() - t0
        if output_path is not None:
            output_path.write_text(json.dumps(out, indent=2))
            print(f"saved: {output_path}")
        return out
    finally:
        wmod._hard_eval_maybe_push = orig_push
        _restore(sim, originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_axes() -> None:
    print(f"{'axis':<22} {'category':<14} {'training_range':<22} description")
    print("-" * 110)
    for cat in CATEGORY_ORDER:
        for name in axes_in_category(cat):
            cfg = AXES[name]
            tr = f"[{cfg.training_range[0]}, {cfg.training_range[1]}]" if cfg.training_range else "(not in training)"
            print(f"{name:<22} {cat:<14} {tr:<22} {cfg.description}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-axes", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--mjcf", type=Path)
    p.add_argument(
        "--category",
        choices=(*CATEGORY_ORDER, "all"),
        help="Run every axis in the named category (or 'all' across all four). Mutually exclusive with --axes.",
    )
    p.add_argument("--axes", type=str, help="Comma-separated axis names. Mutually exclusive with --category.")
    p.add_argument("--values", type=str, default=None, help="Comma-separated value override (single axis only).")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--trial-joint-noise-std", type=float, default=0.02)
    p.add_argument(
        "--fall-height",
        type=float,
        default=None,
        help="Pelvis-z below which a rollout counts as fallen. Default: ROBOT_CFG.fall_height_m (LeRobot: 0.35).",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "JSON with optional fields {init_quat_wxyz: [w,x,y,z], tracking_body_name: str, "
            "fall_height_m: float, schedules: {name: [[t,vx,vy,wz,label], ...]}}. "
            "Defaults match the LeRobot velocity policy."
        ),
    )
    args = p.parse_args()

    if args.robot_config is not None:
        load_robot_config(args.robot_config)
    if args.fall_height is None:
        args.fall_height = ROBOT_CFG.fall_height_m

    if args.list_axes:
        _list_axes()
        return

    if (args.category is None) == (args.axes is None):
        p.error("specify exactly one of --category or --axes")

    required = ("checkpoint", "config", "mjcf", "output")
    missing = [n for n in required if getattr(args, n) is None]
    if missing:
        p.error(f"missing required arguments: {missing}")

    if args.category is not None:
        cats = CATEGORY_ORDER if args.category == "all" else (args.category,)
        axes = [a for c in cats for a in axes_in_category(c)]
    else:
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
        if any(a not in AXES for a in axes):
            p.error(f"unknown axes: {[a for a in axes if a not in AXES]}. Run --list-axes.")

    values_override = None
    if args.values is not None:
        if len(axes) != 1:
            p.error("--values is only valid with a single axis.")
        values_override = [float(v) for v in args.values.split(",") if v.strip()]

    run_sweep(
        checkpoint=args.checkpoint,
        config_path=args.config,
        mjcf_path=args.mjcf,
        axes=axes,
        values_override=values_override,
        n_trials=args.n_trials,
        duration_s=args.duration,
        trial_joint_noise_std=args.trial_joint_noise_std,
        fall_height_m=args.fall_height,
        device_str=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
