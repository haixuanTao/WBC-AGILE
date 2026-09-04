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

## Contact pathology on the generated terrain (and `AGILE_NEWTON_FLAT_TERRAIN`)

The Newton contact sensor reports large *pulling* (negative) normal forces on the
feet that PhysX never reports — standing still, 217/1600 samples below -1 N with a
worst case of -3432 N against a 324 N robot, versus 0/1600 and -0.6 N on PhysX.

The feet are **not** adhering. Pull up with 2x body weight and they leave the
ground more readily than on PhysX (+3.811 m vs +0.634 m in 1 s). The force is real
but it is not a normal force:

| foot contacts | normal.z | vertical force from normal | from friction | friction share |
|---|---:|---:|---:|---:|
| negative-Fz | median +0.000, 95.5% horizontal | -21.4 N | **-111.4 N** | **83.9%** |
| ordinary ground contacts | median -1.000 | +54.7 N | +30.4 N | 35.8% |

The negative population is a distinct class: horizontal contact normals,
penetration pinned at exactly -5.00 mm, and 84% of the downward force carried by
**friction**. That is the signature of the foot collision spheres catching on the
*internal triangle edges* of the trimesh terrain — an edge contact yields a
sideways normal, and friction on a sideways normal acts vertically.

It matters because `net_forces_w` feeds `ground_slam` / `torso_slam` terminations
and the foot-contact rewards, so the corruption reaches the learner. It is a
plausible source of the contact-impact spike that poisoned the last 50 iterations
of the 1000-iteration run.

**Confirmed by removing the terrain.** Same probe, flat plane
(`bench/scripts/foot_penetration_forensics.py`):

| | trimesh terrain | flat plane |
|---|---:|---:|
| negative-Fz foot contacts | 156 / 1095 (14%) | **none** |
| normal.z median | +0.000 (42-96% horizontal) | **+1.000** (0% horizontal) |
| friction share of vertical force | 36% / 84% | **0.0%** |

`AGILE_NEWTON_FLAT_TERRAIN=1` selects a plane instead of the generated terrain.
Measured: 40/40 iterations, 0 NaN, 1.18 s/iter at 4096 envs. It changes the task
(no rough ground, and the terrain-level curriculum is inert — though that
curriculum never advanced past level 0 in the 1000-iteration run either).

Two related notes:

- **Friction combination.** MuJoCo combines friction with `max` unless geom
  priorities differ, so AGILE's randomised foot mu (0.2-1.5) is clipped from below
  by the terrain's mu=1.0: measured range on foot contacts is 0.897-1.499. Half the
  randomisation is silently gone. Raising the foot geoms' `geom_priority`
  post-compile halves the pulling samples (21.6% -> 10.8%, worst -3100 -> -2117 N)
  but does not move mu below 1.0, so it has to be set before the MJWarp model is
  built to take effect properly.
- **Heightfield is the real fix for keeping rough terrain.** The path exists end to
  end — `ModelBuilder.add_shape_heightfield`, `GeoType.HFIELD` -> `mjGEOM_HFIELD`
  in `solver_mujoco.py`, and `HFIELD x SPHERE` in MuJoCo-Warp's collision table —
  so importing Isaac Lab's generated terrain as a heightfield rather than a trimesh
  would give correct up-normals with no internal edges. Not implemented here.

## What made the motion look wrong (and the fixes)

Recording the first flat-ground policy showed a get-up that did not look like a
robot. Three separate things were behind it, none of them the motors: measured
against the DC-motor torque-speed curve, **0.0%** of applied torque exceeded what
a real motor could deliver at that speed.

1. **The training harness was in the video.** `LiftActionCfg` on `torso_link`
   (5000 N/m, capped at 0.9x body weight) is a training-only action that the
   `adaptive_lift` curriculum decays as the policy succeeds; it was still at 0.70
   at the end of training, and a freshly built env starts it at 1.0.
   `scripts/eval.py` strips it via `prepare_training_only_actions_for_evaluation`;
   the recorder did not. Recorded properly, the same policy goes down and stays
   down (pelvis 0.898 -> 0.089 m for 15 s). The recorder and the envelope probe
   now strip it by default (`--keep-assist` to see the harness).

2. **The post-step velocity clamp is not physics.** With it off, 12/29 joints
   exceed their rated speed (shoulders 1.8x), and with it on they sit *exactly*
   on the limit -- because it overwrites velocity after the solve, which removes
   energy for free every step. A policy trained against it learns to lean on a
   brake no real robot has. Replaced by the torque-speed curve itself, inside the
   solver: `AGILE_NEWTON_DC_ENVELOPE=1` rewrites MuJoCo's per-joint
   `jnt_actfrcrange` every physics step from
   `tau_max(qd) = clip(sat (1 - qd/vel), 0, eff)` (and the mirror for `tau_min`),
   as a graph-captured Newton post-actuator callback. Verified against the
   solver's own `qfrc_actuator`: at 29.8 rad/s on a hip (limit 32) the range is
   [-88, 54.4] and the applied force is 54.4. Effort limits themselves were
   already enforced in-solver (`qfrc_actuator` peaks at exactly the limit).

3. **Newton's joint-limit spring was what the clamp had been hiding.** With the
   clamp off, bang-bang actions NaN the sim within ~40 control steps: actuator
   force stays pinned at the effort limit while the *constraint* force runs to
   1e8-1e10 and links tunnel metres into the ground. Newton hands MuJoCo each
   joint limit as an explicit spring, `solref_limit = (-ke, -kd)` with the
   builder default ke=1e4, kd=10 -- stiff and nearly undamped. MuJoCo's native
   time-constant limit constraint, `solref = (0.02, 1.0)`, is critically damped
   and stable by construction. `AGILE_NEWTON_LIMIT_SOLREF=0.02,1` writes it into
   `jnt_solref` and zeroes the Newton-side `limit_ke` so the update kernel does
   not restore the spring. Same probe: 200/200 steps, peak constraint force
   8.7e3, penetration never past -4 cm. (Substeps, limit damping and the DC
   envelope only delayed the blow-up; `AGILE_NEWTON_LIMIT_KD` never applied with
   the clamp off, as it lives inside the clamp patch.)

Full stack -- flat ground, implicit actuators, `implicitfast`, DC envelope,
native joint limits, **velocity clamp off**: 60/60 iterations, 0 NaN, 1.17 s/iter
at 4096 envs. Launcher: `bench/scripts/run_newton_physfix_train.sh`.

## Where the energy comes from (audit), and the newest stack

**"The body accelerates far beyond what the actuators put in."** True, and
measured: peak pelvis accelerations of 25-30 g on flat ground. The audit
(`bench/scripts/energy_audit.py`, `com_readout_check.py`, `drop_impact_compare.py`)
puts the energy where it belongs:

* Position- and velocity-based COM accelerations agree to 0.1 g; at the peaks the
  contact sensor reads 8.0 kN on a 33 kg robot (8027 N / 33 kg = 24.8 g). The
  impulses are the *ground*, not the motors, and not the solver inventing
  momentum. In flight (no contact) the per-step energy residual is median -0.3 J,
  worst +3.1 J, net dissipative.
* Same 0.5 m drop on both engines: Newton 9.6 kN / 28 g, **PhysX 19.7 kN / 53 g**.
  PhysX hits twice as hard. (MuJoCo-Warp's root-dof `qfrc_constraint` is stale
  most steps and must not be used for this; the contact sensor is reliable.)
* Contact stiffness is derived by Newton from its shape materials
  (`timeconst = 2/kd`, defaults land on 0.02 s = four physics steps);
  `AGILE_NEWTON_CONTACT_SOLREF=0.01,1` gives a PhysX-like two-step contact:
  20.4 kN / 61 g on the same drop. MuJoCo-Warp has no global override.

**"The moment about the COM has the wrong sign."** With every joint locked (one
rigid body) the contact moment is correct: feet 0.25 m behind the COM with +1.3 kN
pitch the body nose-down in 8/8 envs, angular momentum +5.2 to +7.0 kg m^2/s, and
the solver's own contact torques match dL/dt at cosine 0.996-0.998. With the
joints PD-held the *torso* pitches nose-up while total angular momentum barely
moves: the implicit PD snaps the legs back on impact and throws the torso the
other way. Momentum is conserved; it is a task-model (actuator) effect, stronger
on Newton than PhysX because Newton's default contact is softer.

**Arms.** Not physically locked (full torque caps, zero extra damping); the
policy commands them ~0.3 rad while the legs sweep 3-4 rad. The arm-deviation
penalty only applies above 0.4 m, the arms are 25 Nm / 5 Nm against 88-139 Nm
legs, and nothing rewards using them.

**Isaac Lab `develop` + Newton 1.5.1** (branch `newton-port-dev`, a layered venv
under `/workspace/IsaacLab-dev/.venv-lite`): same physics. Stock: joint pushed
into its limit runs 3.15 rad past at 426 rad/s and NaNs; bang-bang NaNs at step
18; reset-stress NaNs at step 236; drop 9.4 kN / 27.7 g. With the patches from
this tree attached (they fit the 1.5.1 internals): limit holds at 0.065 rad,
bang-bang survives 200/200, reset-stress 1500/1500. The one patch upstream now
carries is the warm-start-after-NaN clear (isaaclab_newton 1.6.2). Newton 1.5.1's
MuJoCo solver still lists `joint_velocity_limit` as unsupported and still ships
`limit_ke=1e4, limit_kd=10` as the joint-limit default.

**Training without the harness** (`AGILE_NO_ASSIST=1`, fine-tuned from the
iteration-7000 checkpoint of the fixed run): 0 NaN over 10,000 + 8,500
iterations, height error 0.16 -> 0.036 m, velocity clamp engaged on 0.0014% of
joint samples. Harness-free recordings stand up from the floor twice per episode,
to 0.65-0.71 m. Converged around iteration 25,000: tracking keeps sharpening, the
get-up does not improve, reward oscillates -19..-65 without trend. What the reward
does not yet ask for on flat ground: soft landings (slam penalties are gated
behind the terrain curriculum) and holding the standing height.
