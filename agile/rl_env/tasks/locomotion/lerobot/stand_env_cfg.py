# SPDX-FileCopyrightText: LeRobot no-arms "stand in place" task.
# SPDX-License-Identifier: Apache-2.0

"""Stand-in-place variant of the velocity task.

Commands are pinned to zero (so every env's target is "don't move"), terrain
curriculum is disabled (the only learning pressure is balance under
perturbations), and external force/torque + push events are made larger and
more frequent so the policy actively learns to reject disturbances instead of
sitting idle.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from agile.rl_env.tasks.locomotion.lerobot.velocity_env_cfg import LeRobotVelocityEnvCfg


@configclass
class LeRobotStandEnvCfg(LeRobotVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Zero command — always standing, no velocity target.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # Active perturbations — bigger magnitude + more frequent than the
        # velocity task so the policy must learn to reject them.
        # Robot weight ≈ 200 N; torso force ±25 N ≈ 12% of body weight.
        self.events.apply_external_force_torque.interval_range_s = (2.0, 6.0)
        self.events.apply_external_force_torque.params["force_range"] = (-25.0, 25.0)
        self.events.apply_external_force_torque.params["torque_range"] = (-12.0, 12.0)

        self.events.apply_external_force_torque_extremities.interval_range_s = (2.0, 6.0)
        self.events.apply_external_force_torque_extremities.params["force_range"] = (-12.0, 12.0)
        self.events.apply_external_force_torque_extremities.params["torque_range"] = (-1.5, 1.5)

        self.events.push_robot.interval_range_s = (1.5, 4.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }

        # Longer episodes so a late-episode push still leaves recovery time.
        self.episode_length_s = 45.0

        # Disable the velocity-style travel-based curriculum — it never triggers
        # when commanded velocity is zero. The velocity env's __post_init__ then
        # sets `terrain_generator.curriculum = False`, which makes each env land
        # on a uniformly random cell of the 20×16 terrain grid (full-range
        # difficulty randomization across envs, no progression).
        self.curriculum.terrain_levels = None
        # Let envs spawn across the full difficulty range of the grid (default
        # was 1, which would have restricted initial spawns). Does nothing in
        # non-curriculum mode but makes intent explicit and future-proofs.
        self.scene.terrain.max_init_terrain_level = 19
