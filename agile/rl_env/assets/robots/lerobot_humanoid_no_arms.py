# SPDX-FileCopyrightText: Adapted from booster_t1.py for LeRobot bipedal (no arms).
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the LeRobot Humanoid (12-DoF, no arms) in Isaac Lab.

Loads the URDF at
    /home/baguette/lerobot-humanoid-design/to_real_robot/bipedal_plateform_no_arms/urdf/robot.urdf
via Isaac Lab's UrdfConverter. Actuator stiffness/damping values are copied
from the validated mjlab velocity-training config.
"""

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from agile.rl_env.mdp.actuators import CoupledAnkleDelayedDCMotorCfg, DelayedDCMotorCfg

# Using the NEW URDF from zoo commit 8a7725d "new urdf" (pulled 2026-04-21),
# with digit-prefixed mesh filenames renamed (m_*) and package:// paths rewritten
# to absolute file://.
URDF_PATH = str(pathlib.Path(__file__).parent / "lerobot_humanoid_no_arms_new.urdf")
assert pathlib.Path(URDF_PATH).exists(), f"URDF not found: {URDF_PATH}"

# Joint-name patterns for use in reward/observation/event term configs.
LEG_JOINT_NAMES = [
    ".*hipz.*",
    ".*hipx.*",
    ".*hipy.*",
    ".*knee.*",
    ".*ankley.*",
    ".*anklex.*",
]

# LeRobot no-arms has no upper body; keep these empty for interface parity.
WAIST_JOINT_NAMES: list[str] = []
ARM_JOINT_NAMES: list[str] = []
HEAD_JOINT_NAMES: list[str] = []
HAND_JOINT_NAMES: list[str] = []

FEET_LINK_NAMES = [
    "foot_subassembly",
    "foot_subassembly_2",
]
# Parent body for lift harness / height reference (Booster T1 uses "H2" the head;
# we use the torso since there is no head).
TRUNK_BODY = "torso_subassembly"

# Approximate standing base height (meters) — matches velocity-policy recordings.
DEFAULT_TRUNK_HEIGHT = 0.72

# Undesired contact links for the illegal_contact penalty. Anything that is NOT
# a foot collision — catches knee-on-ground, pelvis-on-ground, etc.
UNDESIRED_CONTACTS_LINKS = [
    "torso.*",
    ".*hip.*",
    ".*tigh.*",
    ".*shin.*",
    ".*ankle.*",
]

MIN_DELAY_STEPS = 0
# 8 physics steps at 500 Hz = 16 ms, covers the real-robot control-loop latency
# (~20 ms total: sensor + ROS + inference + command). Matches the mjlab config.
MAX_DELAY_STEPS = 8

LEROBOT_NO_ARMS_DELAYED_DC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        merge_fixed_joints=False,
        root_link_name="torso_subassembly",
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    # Straight-legs init at identity orientation. The previous quat
    # ``(0.9946, -0.0523, -0.0872, 0.0)`` (≈-6° roll, -10° pitch) was added to
    # counter-rotate an earlier URDF's observed spawn tilt; with the asymmetric
    # roll component it biased every reset toward a leftward lean (resets are
    # centered on -6° roll rather than 0°), encouraging the policy to learn
    # asymmetric correction strategies. Switching to identity to remove that
    # bias — re-add only if a fresh visualization confirms the URDF still
    # spawns tilted.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "hipz_right": 0.0,
            "hipz_left": 0.0,
            "hipx_right": 0.0,
            "hipx_left": 0.0,
            "hipy_right": 0.0,
            "hipy_left": 0.0,
            "knee_right": 0.0,
            "knee_left": 0.0,
            "ankley_right": 0.0,
            "ankley_left": 0.0,
            "anklex_right": 0.0,
            "anklex_left": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Values ported from the mjlab velocity-training config
        # (lerobot-legged-zoo/training_exemples/lerobot_humanoid_no_arms/
        # lerobot_humanoid_no_arms_constants.py). Kp/Kd, armature, and friction
        # numbers are copied as-is per joint (L/R asymmetries preserved).
        # Caveat: mjlab's `frictionloss` is a Coulomb torque in N·m; Isaac's
        # `friction` is a unitless coefficient that scales with the spatial force
        # through the joint (F_resist <= mu * F_spatial). Numbers match, physics
        # differs — expect load-dependent friction in sim instead of constant.
        #
        # Split into two groups: hips+knees (independent DCMotors) and ankles
        # (coupled-motor diamond clamp). The ankle pair on each side is two
        # physical motors with per-motor firmware tmax=5.5 Nm; the URDF exposes
        # ankley and anklex as independent DOFs, but the real achievable
        # (τ_pitch, τ_roll) is the diamond |τ_roll ± τ_pitch| ≤ 11 Nm, not the
        # 44×44 box mjlab's config implied.
        "hip_knee": DelayedDCMotorCfg(
            max_delay=MAX_DELAY_STEPS,
            min_delay=MIN_DELAY_STEPS,
            saturation_effort=130.0,
            joint_names_expr=[".*hipz.*", ".*hipx.*", ".*hipy.*", ".*knee.*"],
            stiffness={
                ".*hipz.*": 30.0,
                ".*hipx.*": 40.0,
                ".*hipy.*": 60.0,
                ".*knee.*": 60.0,
            },
            damping={
                ".*hipz.*": 3.0,
                ".*hipx.*": 3.0,
                ".*hipy.*": 4.0,
                ".*knee.*": 4.0,
            },
            velocity_limit_sim={".*": 32.0},
            friction={
                "hipz_right": 0.007,
                "hipz_left": 0.007,
                "hipx_right": 0.006,
                "hipx_left": 0.006,
                "hipy_right": 0.007,
                "hipy_left": 0.007,
                "knee_right": 0.005,
                "knee_left": 0.005,
            },
            armature={
                "hipz_right": 0.02266,
                "hipz_left": 0.02266,
                "hipx_right": 0.13327,
                "hipx_left": 0.13388,
                "hipy_right": 0.14077,
                "hipy_left": 0.14077,
                "knee_right": 0.12330,
                "knee_left": 0.12330,
            },
            # Keep effort limits loose in sim so the policy gets a smooth gradient
            # everywhere; rely on reward-side torque penalties + real-robot firmware
            # clipping at deploy rather than hard-clipping in sim (which causes the
            # policy to "rail-ride" the limit).
            effort_limit_sim={
                ".*hipz.*": 88.0,
                ".*hipx.*": 88.0,
                ".*hipy.*": 88.0,
                ".*knee.*": 88.0,
            },
        ),
        "ankles": CoupledAnkleDelayedDCMotorCfg(
            max_delay=MAX_DELAY_STEPS,
            min_delay=MIN_DELAY_STEPS,
            saturation_effort=130.0,
            joint_names_expr=[".*ankley.*", ".*anklex.*"],
            stiffness={
                ".*ankley.*": 20.0,
                ".*anklex.*": 20.0,
            },
            damping={
                ".*ankley.*": 1.5,
                ".*anklex.*": 1.5,
            },
            velocity_limit_sim={".*": 32.0},
            friction={
                "ankley_right": 0.001,
                "ankley_left": 0.001,
                "anklex_right": 0.001,
                "anklex_left": 0.001,
            },
            armature={
                "ankley_right": 0.03007,
                "ankley_left": 0.02970,
                "anklex_right": 0.02994,
                "anklex_left": 0.03006,
            },
            # Loose per-DOF box (smooth gradient). The diamond clamp in
            # compute() still enforces the physical coupling |τ_roll ± τ_pitch| ≤
            # 2·motor_tmax, so the achievable torque set matches real hardware
            # regardless of this box — it just gives the policy room to signal
            # intent before the diamond cuts in.
            effort_limit_sim={
                ".*ankley.*": 44.0,
                ".*anklex.*": 44.0,
            },
            motor_tmax=5.5,
            ankle_pairs=[
                ("ankley_left", "anklex_left"),
                ("ankley_right", "anklex_right"),
            ],
        ),
    },  # type: ignore
)
