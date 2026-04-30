"""Run a headless MuJoCo rollout of the pulled policy to verify it survives sim.

Reuses `sim2mujoco_watcher.build_once` + `run_eval`, which accept .onnx as
well as .pt via PolicyWrapper.from_config. A nonzero return code means at
least one rollout schedule fell — the `--out-dir` has the mp4s for inspection.

Pass --init-from-state-log to seed each rollout from the last row of the
bipedal_state_log.csv pulled off the Pi (joint positions, joint velocities,
base orientation). Without the flag, sim starts from the canonical init pose
(slight quat tilt + default joint positions) and the rollout is NOT a test of
hot data — just a generic sim2sim survival check.
"""

from __future__ import annotations

import argparse
import csv as csvlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from scripts.sim2mujoco_watcher import build_once, run_eval

# MuJoCo joint name -> CSV prefix (CSV uses left/right_<joint>, MuJoCo uses <joint>_<side>).
# The CSV's "ankle_pitch"/"ankle_roll" correspond to MJCF's "ankley"/"anklex".
_MJ_TO_CSV = {
    "hipz_right": "right_hipz",
    "hipx_right": "right_hipx",
    "hipy_right": "right_hipy",
    "knee_right": "right_knee",
    "ankley_right": "right_ankle_pitch",
    "anklex_right": "right_ankle_roll",
    "hipz_left": "left_hipz",
    "hipx_left": "left_hipx",
    "hipy_left": "left_hipy",
    "knee_left": "left_knee",
    "ankley_left": "left_ankle_pitch",
    "anklex_left": "left_ankle_roll",
}

# MuJoCo joint name -> index in snapshot.json's joint_state_deg/joint_velocity_rad_s.
# Per `go_to_zero_and_hold.py`: [left(6), right(6)] with per-side order
# (hipz, hipx, hipy, knee, ankle_pitch, ankle_roll).
_MJ_TO_SNAP_IDX = {
    "hipz_left": 0,
    "hipx_left": 1,
    "hipy_left": 2,
    "knee_left": 3,
    "ankley_left": 4,
    "anklex_left": 5,
    "hipz_right": 6,
    "hipx_right": 7,
    "hipy_right": 8,
    "knee_right": 9,
    "ankley_right": 10,
    "anklex_right": 11,
}


def _read_last_row(csv_path: Path) -> dict:
    last = None
    with csv_path.open() as f:
        for row in csvlib.DictReader(f):
            last = row
    if last is None:
        raise RuntimeError(f"empty CSV: {csv_path}")
    return last


def _snap_base_z(sim) -> float:
    min_z = float("inf")
    for gid in range(sim.mj_model.ngeom):
        if sim.mj_model.geom_group[gid] == 2 or sim.mj_model.geom_bodyid[gid] == 0:
            continue
        gpos = sim.mj_data.geom_xpos[gid]
        gmat = sim.mj_data.geom_xmat[gid].reshape(3, 3)
        if sim.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = sim.mj_model.geom_dataid[gid]
            vadr = sim.mj_model.mesh_vertadr[mid]
            vnum = sim.mj_model.mesh_vertnum[mid]
            verts = (gmat @ sim.mj_model.mesh_vert[vadr : vadr + vnum].T).T + gpos
            z = verts[:, 2].min()
        else:
            z = gpos[2] - float(np.abs(gmat) @ sim.mj_model.geom_size[gid])[2]
        min_z = min(min_z, z)
    pelvis_z = float(sim.mj_data.qpos[2] - min_z)
    sim.mj_data.qpos[:3] = (0.0, 0.0, pelvis_z)
    return pelvis_z


def _build_init_from_state_log(csv_path: Path):
    """Return an init_fn(sim) that seeds base orientation + joint pos/vel from
    the last row of bipedal_state_log.csv. Base XY=0, base linear/angular vel=0
    (robot is held stationary on the Pi while we snapshot). Geom-snap adjusts
    base Z so the lowest geom touches z=0."""
    row = _read_last_row(csv_path)
    # cache deg->rad conversion up front; init_fn is called per rollout.
    joint_pos_rad = {mj: np.deg2rad(float(row[f"{csv}_pos_deg"])) for mj, csv in _MJ_TO_CSV.items()}
    joint_vel_rad = {mj: float(row[f"{csv}_vel_rad_s"]) for mj, csv in _MJ_TO_CSV.items()}
    qx = float(row["orientation_quat_x"])
    qy = float(row["orientation_quat_y"])
    qz = float(row["orientation_quat_z"])
    qw = float(row["orientation_quat_w"])

    def init_fn(sim):
        for mj_name, jp in joint_pos_rad.items():
            jid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid == -1:
                raise RuntimeError(f"MJCF has no joint named {mj_name!r}")
            qp_adr = sim.mj_model.jnt_qposadr[jid]
            qv_adr = sim.mj_model.jnt_dofadr[jid]
            sim.mj_data.qpos[qp_adr] = jp
            sim.mj_data.qvel[qv_adr] = joint_vel_rad[mj_name]
        sim.mj_data.qpos[3:7] = (qw, qx, qy, qz)
        sim.mj_data.qvel[:6] = 0.0
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        pelvis_z = _snap_base_z(sim)
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        return pelvis_z

    print(
        f"[init-log] seeded from {csv_path} last row (t={row.get('time_s')}):"
        f" quat_wxyz=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})"
    )
    return init_fn


def _build_init_from_snapshot(snapshot_path: Path):
    """Return an init_fn(sim) that seeds from `capture_snapshot_on_pi.py`-style
    JSON (also written by `go_to_zero_and_hold.py --snapshot-out`). Joint order
    in the JSON is [left(6), right(6)] per `_MJ_TO_SNAP_IDX`."""
    snap = json.loads(Path(snapshot_path).read_text())
    jp_deg = snap["joint_state_deg"]
    jv_rad = snap.get("joint_velocity_rad_s") or [0.0] * 12
    qx, qy, qz, qw = snap["orientation_quaternion_xyzw"]
    joint_pos_rad = {mj: np.deg2rad(float(jp_deg[idx])) for mj, idx in _MJ_TO_SNAP_IDX.items()}
    joint_vel_rad = {mj: float(jv_rad[idx]) for mj, idx in _MJ_TO_SNAP_IDX.items()}

    def init_fn(sim):
        for mj_name, jp in joint_pos_rad.items():
            jid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid == -1:
                raise RuntimeError(f"MJCF has no joint named {mj_name!r}")
            qp_adr = sim.mj_model.jnt_qposadr[jid]
            qv_adr = sim.mj_model.jnt_dofadr[jid]
            sim.mj_data.qpos[qp_adr] = jp
            sim.mj_data.qvel[qv_adr] = joint_vel_rad[mj_name]
        sim.mj_data.qpos[3:7] = (qw, qx, qy, qz)
        sim.mj_data.qvel[:6] = 0.0
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        pelvis_z = _snap_base_z(sim)
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        return pelvis_z

    max_abs = max(abs(d) for d in jp_deg)
    print(
        f"[init-snap] seeded from {snapshot_path} (t={snap.get('time_s')}): "
        f"max |joint|={max_abs:.2f}° quat_wxyz=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})"
    )
    return init_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True, help=".pt or .onnx — PolicyWrapper handles both")
    ap.add_argument("--s2m-config", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("hot_data/sim_rollout"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fall-height", type=float, default=0.35)
    ap.add_argument(
        "--init-from-state-log",
        type=Path,
        default=None,
        help="Seed sim from the last row of this bipedal_state_log.csv",
    )
    ap.add_argument(
        "--init-from-snapshot",
        type=Path,
        default=None,
        help="Seed sim from a capture_snapshot_on_pi.py-style JSON "
        "(also produced by go_to_zero_and_hold.py --snapshot-out)",
    )
    args = ap.parse_args()

    if args.init_from_state_log and args.init_from_snapshot:
        ap.error("pass only one of --init-from-state-log / --init-from-snapshot")

    device = torch.device(args.device)
    config, sim, obs_proc, act_proc, cmd_prov = build_once(args.s2m_config, args.mjcf, device)
    init_fn = None
    if args.init_from_snapshot:
        init_fn = _build_init_from_snapshot(args.init_from_snapshot)
    elif args.init_from_state_log:
        init_fn = _build_init_from_state_log(args.init_from_state_log)
    try:
        result = run_eval(
            config=config,
            sim=sim,
            obs_processor=obs_proc,
            act_processor=act_proc,
            command_provider=cmd_prov,
            checkpoint=args.policy,
            out_dir=args.out_dir,
            device=device,
            fall_height_m=args.fall_height,
            init_fn=init_fn,
        )
    finally:
        sim.close()

    print(
        f"[sim_rollout] fall_rate={result.fall_rate:.2f}  "
        f"mean_survival={result.mean_survival_time_s:.1f}s  "
        f"mp4s -> {args.out_dir}"
    )
    for r in result.rollouts:
        status = "FELL" if r.fell else "OK"
        print(
            f"    {r.name:10s}  survived={r.survival_time_s:5.1f}s  {status}  "
            f"err_vx={r.tracking_err_vx:.3f}  err_vy={r.tracking_err_vy:.3f}  "
            f"err_wz={r.tracking_err_wz:.3f}"
        )

    return 1 if result.fall_rate > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
