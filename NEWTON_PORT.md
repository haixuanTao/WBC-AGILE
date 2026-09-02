# WBC-AGILE, Newton backend

Full copy of `/workspace/WBC-AGILE` at commit `6830cf9`, with **only** the physics
engine swapped from PhysX to Newton. Everything else — task configs, robot assets,
terrain, rewards, observations, RSL-RL settings, the venv (Isaac Lab 3.0.0b2 /
Isaac Sim 6.0 / torch 2.11.0+cu128 / rsl-rl 5.4.1) — is byte-identical to the
baseline tree.

Run it exactly like the baseline, from this directory:

    env HOME=/root OMNI_KIT_ACCEPT_EULA=YES ./.venv/bin/python scripts/train.py \
        --task HeightTracking-G1-v0 --num_envs 4096 --headless --max_iterations 30

## What changed (`git diff` against the baseline: 9 files, +31 -27)

**1. Backend selection — 8 task configs.**
`from isaaclab_physx.physics import PhysxCfg` → `from isaaclab_newton.physics import NewtonCfg`,
and `self.sim.physics = PhysxCfg()` → `NewtonCfg()`. Isaac Lab dispatches every
asset/sensor factory off `SimulationContext.physics_manager`, which comes from
`sim.physics.class_type`, so this one field switches articulations, contact
sensors, ray casters and frame views to Newton in one move.

`NewtonCfg()` with no arguments resolves to `MJWarpSolverCfg` — Newton's default
MuJoCo-Warp solver, `use_cuda_graph=True`. No solver tuning was applied, since the
point is to compare stock against stock.

**2. PhysX-only tuning knobs — commented out, not silently dropped.**
`gpu_max_rigid_patch_count` and `solver_type` are PhysX GPU-buffer/solver settings
with no Newton counterpart; they are left in place as `# [newton] PhysX-only knob…`
comments so the baseline's tuning stays visible.

**3. Two reward terms read a PhysX-only view — `agile/rl_env/mdp/rewards/regularization_rewards.py`.**
`relax_if_null_cmd_exp.__init__` and `torque_limits()` both called
`asset.root_physx_view.get_dof_max_forces()`. Newton articulations have no
`root_physx_view`. Both now read `asset.data.joint_effort_limits`, which is
backend-agnostic and **value-identical on PhysX** — the PhysX implementation of
that property is literally `wp.clone(root_view.get_dof_max_forces())`, same
`(num_instances, num_joints)` shape on both backends.

## Known remaining PhysX-only call sites (not on the benchmarked path)

- `agile/rl_env/mdp/events/events.py` — `disable_joints_for_rest` calls
  `root_physx_view.set_dof_actuation_forces`. `HeightTracking-G1-v0` does not use
  this event term, so it was left untouched; tasks that do use it will need a
  Newton equivalent.
- `agile/isaaclab_extras/physx_articulation_com_cache.py` — an Isaac Lab
  3.0.0-beta2 COM-cache backport that patches `isaaclab_physx` classes. Under
  Newton it patches a class that is never instantiated, so it is inert. Whether
  Newton's `ArticulationData` has the same COM-cache bug was not investigated.
- `agile/isaaclab_extras/monkey_patches/contact_sensor_patch.py` uses
  `body_physx_view`, but it is not imported by `monkey_patches/__init__.py`, so it
  is dead code on both trees.

## Stabilization: this task needs substepping under Newton

A pure swap (`NewtonCfg()`, 1 substep) **does not produce a usable run**. At AGILE's
`sim.dt = 1/200` (5 ms), MuJoCo-Warp's Euler integrator is unstable for this 29-DOF
articulation: joint velocities grow exponentially and the run dies with
`observation group 'policy' contains NaN` after 1–11 iterations, sooner at higher
env counts.

Measured with `/workspace/bench/scripts/nan_probe.py` at 1024 envs, 400 env-steps,
`|joint_vel|max` across all envs (joint velocity limits for this robot are 20–37 rad/s):

| config | step 0 | step 25 | step 50 | step 75 | step 375 | outcome |
|---|---:|---:|---:|---:|---:|---|
| PhysX | 32.0 | 37.00 | 37.00 | 37.00 | 37.00 | finite, pinned at the limit |
| Newton, 1 substep, zero actions | 8.0 | 61.7 | 123.7 | 1.7e13 | 5.3e14 | diverged |
| Newton, 1 substep + 0.01 margin | 36.8 | 6.4e11 | — | — | — | NaN at step 35 |
| Newton, 2 substeps | 32.7 | 38.0 | 39.2 | 43.2 | — | finite |
| Newton, 4 substeps | 31.7 | 34.1 | 35.8 | 35.4 | 37.2 | finite, naturally bounded |

It diverges with **zero actions**, so it is not the policy — it is the integrator.
Contact margin (Isaac Lab's documented fix for trimesh terrain) makes it *worse*.
`num_substeps` fixes it: 4 substeps holds velocities in their natural 31–37 rad/s
band for the whole probe.

Set via `AGILE_NEWTON_SUBSTEPS` (default `1`, i.e. no change) and
`AGILE_NEWTON_MARGIN` (default `0.0`, i.e. Newton's own default), both read in
`agile/rl_env/tasks/stand_up/g1/height_tracking_env_cfg.py`. Only that one task
config has the knobs; the other seven are pure swaps.

### Why PhysX survives what Newton does not

Newton implements **no joint velocity limit on any of its six solvers** — see the
support matrix in `newton/solvers.py` (`joint_velocity_limit` is `no` across the
board) and `solver_mujoco.py`, which states the attribute is "not supported" and
carries `# MuJoCo doesn't have velocity limit` above a commented-out read at
line 4310.

PhysX enforces it in-solver: `isaaclab_physx/.../articulation.py:1578` calls
`root_view.set_dof_max_velocities(...)`, which is why the PhysX row above sits at
exactly 37.00 — the configured limit — rather than at whatever the dynamics
produce. AGILE sets `velocity_limit_sim` explicitly for every G1 joint group.

So PhysX's clamp was **masking** the same underlying stiffness by truncating the
growth every step before it could compound. Newton has nothing to truncate it,
which is why the instability is visible there and not on PhysX. Substepping fixes
the cause on Newton; it is not a workaround for the missing clamp.

## Additional knobs and patches (added during the investigation)

All read in `agile/rl_env/tasks/stand_up/g1/height_tracking_env_cfg.py` unless noted;
every default reproduces the pure swap.

| env var | default | effect |
|---|---|---|
| `AGILE_NEWTON_SUBSTEPS` | `1` | `NewtonCfg.num_substeps` |
| `AGILE_NEWTON_MARGIN` | `0.0` | `NewtonShapeCfg.margin` |
| `AGILE_NEWTON_GAP` | `0.01` | `NewtonShapeCfg.gap` (Newton's own default) |
| `AGILE_NEWTON_INTEGRATOR` | `euler` | `MJWarpSolverCfg.integrator` (`euler` / `implicitfast`) |
| `AGILE_NEWTON_CUDA_GRAPH` | `1` | `NewtonCfg.use_cuda_graph` |
| `AGILE_NEWTON_VEL_CLAMP` | `1` | `agile/isaaclab_extras/newton_joint_velocity_clamp.py` -- post-step joint velocity clamp emulating PhysX's in-solver `set_dof_max_velocities`; `AGILE_NEWTON_VEL_CLAMP_STATS=1` to count engagement |
| `AGILE_NEWTON_IMPLICIT_ACTUATORS` | `1` | `unitree_g1.py`: height-tracking robot uses `DelayedImplicitActuatorCfg` (solver-side PD) instead of `DelayedDCMotorCfg`; `0` restores the explicit model. Measured: does **not** stabilise the on-ground task; kept as a switch |
| `AGILE_NEWTON_LIMIT_KD` | unset | `newton_joint_velocity_clamp.py`: override Newton's joint-limit spring damping (builder default 10 for ke=1e4). 200 halves the torque-free rebound; 1000 locks the joints |
| `AGILE_NAN_WATCHDOG` | `0` | `agile/isaaclab_extras/nan_watchdog.py` -- reports the first non-finite sim quantity with one step of history |

**Fallen-states cache joint order** (`agile/rl_env/mdp/events/fallen_state_dataset.py`,
`agile/rl_env/tasks/stand_up/g1/pre_learn.py`): the cache stores `joint_pos`/`joint_vel`
by index in the collecting backend's joint order and carried no names. PhysX and
Newton enumerate joints differently (breadth-first vs depth-first), so a cache
collected on one and replayed on the other is scrambled. `save()` now records
`joint_names`; `load(path, target_joint_names=...)` remaps columns by name, assuming
the legacy PhysX order for caches without names. Verified: the loaded dataset is
identical by name on both engines and 100% within joint limits. This was correct
hygiene but **not** the cause of Newton's divergence -- see below.

## Conclusion of the stability investigation

With a controlled, deterministic, single-joint repro the divergence is an
integrator stability property of *AGILE's explicit Python PD + MuJoCo-Warp Euler at
5 ms on a free-floating 29-DOF humanoid*: joint velocity doubles every physics step
against a correctly computed opposing torque, with zero constraint, passive and
actuator forces on the DOF. It is not a reference-frame, axis, joint-order,
effort-routing, inertia, gravity, contact, armature, timing, CUDA-graph or
hidden-drive defect -- each was measured against PhysX and is identical.
Full evidence in `/workspace/bench/results_newton/FINDINGS_NEWTON.md`.

Remedy (measured, `bench/scripts/pd_solver_side.py`): put the PD **inside the solver**.
The same kp=100 / kd=2.5 on the same joint on the same Newton solver is stable when
written as sim-side joint drives (`write_joint_stiffness/damping_to_sim`, i.e. an
`ImplicitActuatorCfg`) -- perturbed 0.1 rad it returns to zero with a damped
overshoot, growth ratio 0.92 -> 0.77, under plain Euler and under implicitfast --
and doubles every step when computed by AGILE's explicit Python `DelayedDCMotor`.

So the concrete change for Newton is: replace `DelayedDCMotorCfg` with an implicit
actuator for the G1 joint groups. That is **not** made in this tree, because it
changes the task's actuator model (the DC-motor torque-speed saturation curve is
lost, and the command delay must be reimplemented on the target side). It is a
task-design decision, not a port fix. Substepping (`AGILE_NEWTON_SUBSTEPS`) and
the velocity clamp only delay the divergence; they do not remove it.

## Working recipe (task changed) -- measured

`AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast` (velocity clamp
on by default), 1 substep: **1000/1000 iterations, 0 NaN, 1.23 s/iter at 4096 envs
(3.32x PhysX)**, reward -1918 -> -420 over 700 iterations. Caveat: a single contact-
impact spike at iteration 950 (return -54,797; `ground_slam`/`torso_slam` ~400x normal)
degraded the policy over the last ~50 iterations. Contact dynamics remain the open
risk. Launcher: `/workspace/bench/scripts/run_newton_implicit_train.sh <iters> <envs> <substeps> <integrator>`.
