"""Sim-to-sim cross-val watcher: polls training checkpoints, runs a short
headless MuJoCo sweep for each new one, logs survival time + velocity tracking
error + a rendered mp4 to the training's W&B run.

Use alongside a live `scripts/train.py` run: it watches the run's checkpoint
directory, picks up every Nth new `model_*.pt`, and pushes per-step metrics
into the same W&B run so the sim-to-sim curve sits next to the training
reward curve.

Examples:
    # One-shot against a single checkpoint (smoke test):
    python scripts/sim2mujoco_watcher.py \\
        --once logs/rsl_rl/velocity_lerobot_no_arms/<run>/model_117750.pt \\
        --config /home/baguette/tmp_eval/config.yaml \\
        --mjcf /home/baguette/tmp_eval/mjcf/scene.xml \\
        --wandb-entity 1ms-ai --wandb-project Velocity-LeRobot-NoArms \\
        --wandb-run-id kuaut9q0

    # Watch mode, attaching to a live training run:
    python scripts/sim2mujoco_watcher.py \\
        --log-dir logs/rsl_rl/velocity_lerobot_no_arms/<run> \\
        --config /home/baguette/tmp_eval/config.yaml \\
        --mjcf /home/baguette/tmp_eval/mjcf/scene.xml \\
        --wandb-entity 1ms-ai --wandb-project Velocity-LeRobot-NoArms \\
        --wandb-run-id kuaut9q0 \\
        --every 1000 --poll-interval 30
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import torch

from agile.sim2mujoco.actions import ActionProcessor
from agile.sim2mujoco.command_provider import (
    VelocityCommandProvider,
    create_command_provider,
)
from agile.sim2mujoco.observations import ObservationProcessor
from agile.sim2mujoco.policy import PolicyWrapper
from agile.sim2mujoco.simulation import MuJocoSimulation
from agile.sim2mujoco.utils import load_config

CKPT_RE = re.compile(r"^model_(\d+)\.pt$")

# Command schedule: (start_time_s, vx, vy, wz, label). Covers stand, forward,
# strafe, yaw — small enough that a reasonable policy should survive the
# whole thing, but aggressive enough to separate good checkpoints from bad.
# Schedule entries are (start_time_s, vx, vy, wz, label). Keep each schedule
# short — we run several per checkpoint so total eval time per ckpt stays
# bounded. Post-fall the loop breaks, so the SCHEDULE_DURATION_S below is an
# upper bound.
SCHEDULE_DURATION_S = 12.0

SCHEDULES: dict[str, list[tuple[float, float, float, float, str]]] = {
    "stand": [
        (0.0, 0.0, 0.0, 0.0, "stand"),
    ],
    "fwd": [
        (0.0, 0.0, 0.0, 0.0, "stand"),
        (2.0, 0.2, 0.0, 0.0, "fwd_slow"),
        (5.0, 0.3, 0.0, 0.0, "fwd_fast"),
        (9.0, 0.0, 0.0, 0.0, "stop"),
    ],
    "strafe": [
        (0.0, 0.0, 0.0, 0.0, "stand"),
        (2.0, 0.0, 0.15, 0.0, "strafe_py"),
        (5.0, 0.0, -0.15, 0.0, "strafe_my"),
        (9.0, 0.0, 0.0, 0.0, "stop"),
    ],
    "yaw": [
        (0.0, 0.0, 0.0, 0.0, "stand"),
        (2.0, 0.0, 0.0, 0.5, "yaw_p"),
        (5.0, 0.0, 0.0, -0.5, "yaw_m"),
        (9.0, 0.0, 0.0, 0.0, "stop"),
    ],
}


# Per-rollout difficulty: domain randomization + scheduled random pushes. Sized
# to be inside the policy's training distribution but stiff enough to surface
# differences between checkpoints that all hit ceiling on the bare-bones eval.
HARD_EVAL_MASS_RANGE = (0.85, 1.15)
HARD_EVAL_FRICTION_RANGE = (0.7, 1.3)
HARD_EVAL_DAMPING_RANGE = (0.7, 1.3)
HARD_EVAL_PUSH_TIMES_S = (3.0, 6.0, 9.0)
HARD_EVAL_PUSH_MAG_RANGE = (0.5, 1.0)


def _hard_eval_capture(sim):
    """Snapshot mj_model arrays we'll randomize; reused across rollouts."""
    return {
        "body_mass": sim.mj_model.body_mass.copy(),
        "geom_friction": sim.mj_model.geom_friction.copy(),
        "dof_damping": sim.mj_model.dof_damping.copy(),
    }


def _hard_eval_apply_dr(sim, originals, rng):
    """Restore originals then resample uniform scales — keeps DR per-rollout."""
    sim.mj_model.body_mass[:] = originals["body_mass"] * rng.uniform(*HARD_EVAL_MASS_RANGE)
    sim.mj_model.geom_friction[:] = originals["geom_friction"] * rng.uniform(*HARD_EVAL_FRICTION_RANGE)
    sim.mj_model.dof_damping[:] = originals["dof_damping"] * rng.uniform(*HARD_EVAL_DAMPING_RANGE)
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def _hard_eval_maybe_push(sim, rng, t_now, t_prev):
    """If a scheduled push time was crossed this step, kick the base velocity
    in a random horizontal direction. The policy has to react online."""
    for pt in HARD_EVAL_PUSH_TIMES_S:
        if t_prev <= pt < t_now:
            mag = rng.uniform(*HARD_EVAL_PUSH_MAG_RANGE)
            angle = rng.uniform(0, 2 * np.pi)
            sim.mj_data.qvel[0] += mag * np.cos(angle)
            sim.mj_data.qvel[1] += mag * np.sin(angle)


@dataclass
class RolloutResult:
    """Metrics + artifact for a single command-schedule rollout."""

    name: str
    survival_time_s: float
    fell: bool
    mean_base_height: float
    tracking_err_vx: float
    tracking_err_vy: float
    tracking_err_wz: float
    num_sim_steps: int
    video_path: str


@dataclass
class EvalResult:
    step: int
    checkpoint: str
    rollouts: list[RolloutResult]

    @property
    def fall_rate(self) -> float:
        return sum(r.fell for r in self.rollouts) / len(self.rollouts)

    @property
    def mean_survival_time_s(self) -> float:
        return float(np.mean([r.survival_time_s for r in self.rollouts]))

    @property
    def mean_tracking_err_vx(self) -> float:
        return float(np.mean([r.tracking_err_vx for r in self.rollouts]))

    @property
    def mean_tracking_err_vy(self) -> float:
        return float(np.mean([r.tracking_err_vy for r in self.rollouts]))

    @property
    def mean_tracking_err_wz(self) -> float:
        return float(np.mean([r.tracking_err_wz for r in self.rollouts]))


def _parse_step(p: Path) -> int | None:
    m = CKPT_RE.match(p.name)
    return int(m.group(1)) if m else None


def _scheduled_cmd(t: float, schedule):
    cur = schedule[0]
    for entry in schedule:
        if entry[0] <= t:
            cur = entry
        else:
            break
    return cur


def _init_base_pose(sim: MuJocoSimulation, init_quat_wxyz=(0.9946, 0.0523, 0.0872, 0.0)):
    """Match record_video.py: tilt base to neutral quat and snap pelvis so
    lowest geom sits at z=0. Keeps the robot from spawning penetrating the floor."""
    sim.mj_data.qpos[3:7] = init_quat_wxyz
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

    min_z = float("inf")
    for gid in range(sim.mj_model.ngeom):
        if sim.mj_model.geom_group[gid] == 2 or sim.mj_model.geom_bodyid[gid] == 0:
            continue
        gpos = sim.mj_data.geom_xpos[gid]
        gmat = sim.mj_data.geom_xmat[gid].reshape(3, 3)
        if sim.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = sim.mj_model.geom_dataid[gid]
            vadr, vnum = sim.mj_model.mesh_vertadr[mid], sim.mj_model.mesh_vertnum[mid]
            verts = (gmat @ sim.mj_model.mesh_vert[vadr : vadr + vnum].T).T + gpos
            z = verts[:, 2].min()
        else:
            z = gpos[2] - float(np.abs(gmat) @ sim.mj_model.geom_size[gid])[2]
        min_z = min(min_z, z)
    pelvis_z = float(sim.mj_data.qpos[2] - min_z)
    sim.mj_data.qpos[:3] = (0.0, 0.0, pelvis_z)
    sim.mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return pelvis_z


def build_once(config_path: Path, mjcf_path: Path, device: torch.device):
    """Build config + a MuJoCo sim + obs/act processors. MjModel is expensive
    to reload, so the watcher reuses one sim across all checkpoints."""
    config = load_config(config_path)
    config["mjcf_path"] = str(mjcf_path)
    sim = MuJocoSimulation(config, device, enable_viewer=False, mjcf_path=mjcf_path)
    obs_processor = ObservationProcessor(config, sim.joint_names, device, command_manager=None)
    command_provider = create_command_provider(config, device, motion_tracker=obs_processor.motion_tracker)
    if isinstance(command_provider, VelocityCommandProvider):
        sim.command_manager = command_provider.manager
        obs_processor.command_manager = command_provider.manager
    act_processor = ActionProcessor(config, sim.joint_names, device)
    return config, sim, obs_processor, act_processor, command_provider


def _run_single_rollout(
    *,
    sim: MuJocoSimulation,
    obs_processor: ObservationProcessor,
    act_processor: ActionProcessor,
    command_provider,
    policy,
    schedule_name: str,
    schedule: list[tuple[float, float, float, float, str]],
    video_path: Path,
    fall_height_m: float,
    video_width: int = 640,
    video_height: int = 360,
    fps: int = 25,
    init_fn=None,
    hard_eval_originals: dict | None = None,
    hard_eval_rng=None,
) -> RolloutResult:
    """Run one rollout on a given command schedule. Terminates early on fall:
    stops stepping the policy + sim, pads the video with the last frame so
    all rollouts produce the same duration (comparable across checkpoints)."""
    sim.reset()
    (init_fn or _init_base_pose)(sim)
    if hard_eval_originals is not None and hard_eval_rng is not None:
        _hard_eval_apply_dr(sim, hard_eval_originals, hard_eval_rng)
    obs_processor.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    control_dt = sim.dt
    num_steps = int(SCHEDULE_DURATION_S / control_dt)

    torso_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_subassembly")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(sim.mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id if torso_id != -1 else 0
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 135.0
    sim.mj_model.vis.global_.offwidth = video_width
    sim.mj_model.vis.global_.offheight = video_height
    renderer = mujoco.Renderer(sim.mj_model, height=video_height, width=video_width)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(video_path), fps=fps, codec="libx264", quality=7)
    frame_stride = max(1, int(round(1.0 / (fps * control_dt))))

    base_heights: list[float] = []
    sq_err_vx, sq_err_vy, sq_err_wz = 0.0, 0.0, 0.0
    n_err_vx, n_err_vy, n_err_wz = 0, 0, 0
    fell = False
    survival_time = SCHEDULE_DURATION_S
    is_velocity = isinstance(command_provider, VelocityCommandProvider)
    zero_actions = torch.zeros(
        act_processor.total_action_dim, device=policy.device if hasattr(policy, "device") else "cpu"
    )

    try:
        for step in range(num_steps):
            t = step * control_dt
            t_prev = (step - 1) * control_dt
            if hard_eval_originals is not None and hard_eval_rng is not None:
                _hard_eval_maybe_push(sim, hard_eval_rng, t, t_prev)
            _, cmd_vx, cmd_vy, cmd_wz, _ = _scheduled_cmd(t, schedule)
            if is_velocity:
                command_provider.manager.set_command(linear_x=cmd_vx, linear_y=cmd_vy, angular_z=cmd_wz)

            sim_state = sim.get_state()
            # After the fall, stop querying the policy (its obs is corrupted
            # by the tumble and can push the sim into NaN). Send zero actions
            # so the robot settles naturally under gravity.
            if fell:
                actions = zero_actions
            else:
                obs = obs_processor.compute(sim_state, noise_scale=0.0)
                with torch.no_grad():
                    actions = policy(obs)
                obs_processor.set_last_action(actions)
            joint_cmd = act_processor.process(actions)
            for _ in range(sim.decimation):
                sim.step(joint_cmd)

            post = sim.get_state()
            bz = float(post.root_pos[2].item())
            base_heights.append(bz)

            # Only accumulate tracking-error stats while the robot is alive —
            # post-fall kinematics aren't informative.
            if not fell:
                meas_vx = float(post.root_lin_vel[0].item())
                meas_vy = float(post.root_lin_vel[1].item())
                meas_wz = float(post.root_ang_vel[2].item())
                if abs(cmd_vx) > 1e-6:
                    sq_err_vx += (cmd_vx - meas_vx) ** 2
                    n_err_vx += 1
                if abs(cmd_vy) > 1e-6:
                    sq_err_vy += (cmd_vy - meas_vy) ** 2
                    n_err_vy += 1
                if abs(cmd_wz) > 1e-6:
                    sq_err_wz += (cmd_wz - meas_wz) ** 2
                    n_err_wz += 1

            if step % frame_stride == 0:
                renderer.update_scene(sim.mj_data, camera=cam)
                writer.append_data(renderer.render())

            if not fell and bz < fall_height_m:
                fell = True
                survival_time = t
    finally:
        writer.close()
        renderer.close()

    steps_ran = num_steps

    return RolloutResult(
        name=schedule_name,
        survival_time_s=survival_time,
        fell=fell,
        mean_base_height=float(np.mean(base_heights)) if base_heights else 0.0,
        tracking_err_vx=float(np.sqrt(sq_err_vx / n_err_vx)) if n_err_vx else 0.0,
        tracking_err_vy=float(np.sqrt(sq_err_vy / n_err_vy)) if n_err_vy else 0.0,
        tracking_err_wz=float(np.sqrt(sq_err_wz / n_err_wz)) if n_err_wz else 0.0,
        num_sim_steps=steps_ran,
        video_path=str(video_path),
    )


def run_eval(
    *,
    config: dict,
    sim: MuJocoSimulation,
    obs_processor: ObservationProcessor,
    act_processor: ActionProcessor,
    command_provider,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    fall_height_m: float = 0.35,
    video_width: int = 640,
    video_height: int = 360,
    fps: int = 25,
    init_fn=None,
    hard_eval: bool = False,
    hard_eval_originals: dict | None = None,
    hard_eval_seed: int | None = None,
) -> EvalResult:
    """Run all SCHEDULES for a checkpoint and return aggregated results."""
    policy = PolicyWrapper.from_config(checkpoint, config, device)
    step = _parse_step(checkpoint) or -1
    rollouts: list[RolloutResult] = []
    for name, schedule in SCHEDULES.items():
        video_path = out_dir / f"eval_{step:08d}_{name}.mp4"
        rollout_rng = (
            np.random.default_rng((hard_eval_seed or 0) ^ hash(name) & 0xFFFFFFFF)
            if hard_eval and hard_eval_originals is not None
            else None
        )
        r = _run_single_rollout(
            sim=sim,
            obs_processor=obs_processor,
            act_processor=act_processor,
            command_provider=command_provider,
            policy=policy,
            schedule_name=name,
            schedule=schedule,
            video_path=video_path,
            fall_height_m=fall_height_m,
            video_width=video_width,
            video_height=video_height,
            fps=fps,
            init_fn=init_fn,
            hard_eval_originals=hard_eval_originals if hard_eval else None,
            hard_eval_rng=rollout_rng,
        )
        rollouts.append(r)
    return EvalResult(step=step, checkpoint=checkpoint.name, rollouts=rollouts)


def _log_wandb(run, result: EvalResult, log_video: bool):
    if run is None:
        return
    import wandb

    payload: dict = {
        "sim_to_sim/iter": result.step,
        # Aggregated across all rollouts.
        "sim_to_sim/fall_rate": result.fall_rate,
        "sim_to_sim/mean_survival_time_s": result.mean_survival_time_s,
        "sim_to_sim/mean_tracking_err_vx": result.mean_tracking_err_vx,
        "sim_to_sim/mean_tracking_err_vy": result.mean_tracking_err_vy,
        "sim_to_sim/mean_tracking_err_wz": result.mean_tracking_err_wz,
    }
    for r in result.rollouts:
        prefix = f"sim_to_sim/{r.name}"
        payload[f"{prefix}/survival_time_s"] = r.survival_time_s
        payload[f"{prefix}/fell"] = int(r.fell)
        payload[f"{prefix}/tracking_err_vx"] = r.tracking_err_vx
        payload[f"{prefix}/tracking_err_vy"] = r.tracking_err_vy
        payload[f"{prefix}/tracking_err_wz"] = r.tracking_err_wz
        if log_video and Path(r.video_path).exists():
            payload[f"{prefix}/video"] = wandb.Video(r.video_path, fps=25, format="mp4")
    # Use the custom `sim_to_sim/iter` step axis — the run's default step is
    # owned by the training process and advances past our checkpoint's
    # iteration, which would otherwise cause W&B to drop us as out-of-order.
    run.log(payload)
    # Also push scalar values into run.summary so the dashboard's auto-generated
    # panels (which only surface metrics that show up in summary) render the
    # sim-to-sim curves alongside the training reward without requiring the
    # user to manually add panels for each metric.
    for key, value in payload.items():
        if isinstance(value, int | float | bool):
            run.summary[key] = value


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--log-dir", type=Path, help="Directory to watch for model_*.pt")
    mode.add_argument("--once", type=Path, help="Run once against a single checkpoint and exit")

    p.add_argument("--config", type=Path, required=True, help="sim2mujoco YAML config")
    p.add_argument("--mjcf", type=Path, required=True, help="MJCF file")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to write mp4s (default: alongside checkpoints)")
    p.add_argument("--device", type=str, default="cpu", help="cpu (default, doesn't fight training for GPU) or cuda")

    p.add_argument("--every", type=int, default=1000, help="Stride in training iters between evals (watch mode)")
    p.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between checkpoint-dir polls")
    p.add_argument("--fall-height", type=float, default=0.35)
    p.add_argument(
        "--hard-eval",
        action="store_true",
        help="Apply per-rollout DR (mass/friction/damping) + scheduled random pushes to differentiate strong checkpoints.",
    )
    p.add_argument(
        "--hard-eval-seed",
        type=int,
        default=42,
        help="Seed for hard-eval RNG so checkpoints face the same DR + push sequence.",
    )

    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-id", type=str, default=None, help="Log into an existing training run (resume=allow)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-video-upload", action="store_true", help="Save mp4 locally but don't push to W&B")

    args = p.parse_args()

    device = torch.device(args.device)
    config, sim, obs_proc, act_proc, cmd_prov = build_once(args.config, args.mjcf, device)
    hard_eval_originals = _hard_eval_capture(sim) if args.hard_eval else None
    if args.hard_eval:
        print(
            f"[hard-eval] enabled: mass {HARD_EVAL_MASS_RANGE} friction {HARD_EVAL_FRICTION_RANGE} damping {HARD_EVAL_DAMPING_RANGE} pushes at {HARD_EVAL_PUSH_TIMES_S}s mag {HARD_EVAL_PUSH_MAG_RANGE}m/s seed={args.hard_eval_seed}"
        )

    wb_run = None
    if not args.no_wandb and args.wandb_run_id:
        import wandb

        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow",
            job_type="sim_to_sim_eval",
        )
        # Define a custom step axis so our logs aren't rejected for being
        # "behind" the training run's global step counter.
        wb_run.define_metric("sim_to_sim/iter")
        wb_run.define_metric("sim_to_sim/*", step_metric="sim_to_sim/iter")
        print(f"[wandb] resumed run {args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")

    def _eval_one(ckpt: Path):
        step = _parse_step(ckpt)
        if step is None:
            return
        out_dir = args.output_dir or (ckpt.parent / "sim_to_sim")
        print(f"[eval] {ckpt.name} …")
        t0 = time.time()
        result = run_eval(
            config=config,
            sim=sim,
            obs_processor=obs_proc,
            act_processor=act_proc,
            command_provider=cmd_prov,
            checkpoint=ckpt,
            out_dir=out_dir,
            device=device,
            fall_height_m=args.fall_height,
            hard_eval=args.hard_eval,
            hard_eval_originals=hard_eval_originals,
            hard_eval_seed=args.hard_eval_seed,
        )
        dt = time.time() - t0
        per_rollout = "  ".join(
            f"{r.name}(survived={r.survival_time_s:.1f}s{' FELL' if r.fell else ''})" for r in result.rollouts
        )
        print(
            f"[eval] step={result.step}  fall_rate={result.fall_rate:.2f}  "
            f"mean_survival={result.mean_survival_time_s:.1f}s  {per_rollout}  "
            f"({dt:.1f}s wall)"
        )
        _log_wandb(wb_run, result, log_video=not args.no_video_upload)
        return result

    try:
        if args.once is not None:
            _eval_one(args.once)
            return

        watch_dir = args.log_dir.resolve()
        print(f"[watch] {watch_dir}  every {args.every} iters  poll {args.poll_interval:.0f}s")
        processed: set[int] = set()
        # Anchor to the latest existing checkpoint so we don't backfill 500 old ones.
        existing = sorted(
            (p for p in watch_dir.glob("model_*.pt") if _parse_step(p) is not None),
            key=lambda p: _parse_step(p) or 0,
        )
        if existing:
            anchor = _parse_step(existing[-1]) or 0
            print(f"[watch] anchor: only evaluating checkpoints with step > {anchor}")
            processed = {s for s in (_parse_step(p) for p in existing) if s is not None}

        while True:
            candidates = []
            for ckpt in watch_dir.glob("model_*.pt"):
                step = _parse_step(ckpt)
                if step is None or step in processed:
                    continue
                if step % args.every != 0:
                    processed.add(step)  # skip but mark seen
                    continue
                # Stability check: size must be stable across a short wait.
                s1 = ckpt.stat().st_size
                time.sleep(0.5)
                if not ckpt.exists() or ckpt.stat().st_size != s1 or s1 == 0:
                    continue
                candidates.append((step, ckpt))

            for step, ckpt in sorted(candidates):
                _eval_one(ckpt)
                processed.add(step)

            time.sleep(args.poll_interval)
    finally:
        if wb_run is not None:
            wb_run.finish()
        sim.close()


if __name__ == "__main__":
    main()
