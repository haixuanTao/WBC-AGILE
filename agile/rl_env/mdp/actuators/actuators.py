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


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.actuators import DCMotor, ImplicitActuator
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from agile.rl_env.mdp.actuators.actuators_cfg import (
        CoupledAnkleDelayedDCMotorCfg,
        DelayedDCMotorCfg,
        DelayedImplicitActuatorCfg,
    )


class DelayedDCMotor(DCMotor):
    """Delayed DC motor actuator model."""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args: Any, **kwargs: Any) -> None:
        if cfg.velocity_limit is None:
            cfg.velocity_limit = cfg.velocity_limit_sim

        super().__init__(cfg, *args, **kwargs)

        # saturation effort as tensor
        self._saturation_effort = self._parse_joint_parameter(self.cfg.saturation_effort, torch.inf)
        # find the velocity on the torque-speed curve that intersects effort_limit in the second and fourth quadrant
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)

        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class CoupledAnkleDelayedDCMotor(DelayedDCMotor):
    """Delayed DC motor that enforces the diamond torque envelope of coupled
    ankle motor pairs.

    After the base class computes joint torques, each configured
    ``(pitch, roll)`` joint pair is mapped back to the two underlying motor
    torques via the inverse of the position coupling
    ``q_pitch = -(m_a - m_b)/2``, ``q_roll = (m_a + m_b)/2`` — i.e.
    ``τ_a = (τ_roll - τ_pitch)/2`` and ``τ_b = (τ_roll + τ_pitch)/2``.
    Each motor is clamped to ``±motor_tmax`` and then forward-mapped back
    to joint torques. Joints outside the configured pairs are left untouched.
    """

    cfg: CoupledAnkleDelayedDCMotorCfg

    def __init__(self, cfg: CoupledAnkleDelayedDCMotorCfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)

        pair_idx: list[tuple[int, int]] = []
        for pitch_name, roll_name in cfg.ankle_pairs:
            try:
                pitch_idx = self.joint_names.index(pitch_name)
                roll_idx = self.joint_names.index(roll_name)
            except ValueError as exc:
                raise ValueError(
                    f"CoupledAnkleDelayedDCMotor: joint(s) not in scope "
                    f"(pair=({pitch_name}, {roll_name}), scope={self.joint_names})"
                ) from exc
            pair_idx.append((pitch_idx, roll_idx))
        self._ankle_pair_idx = pair_idx
        self._motor_tmax = float(cfg.motor_tmax)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        result = super().compute(control_action, joint_pos, joint_vel)
        tau = result.joint_efforts
        if tau is None or not self._ankle_pair_idx:
            return result
        tmax = self._motor_tmax
        for pitch_idx, roll_idx in self._ankle_pair_idx:
            tau_p = tau[..., pitch_idx]
            tau_r = tau[..., roll_idx]
            m_a = torch.clamp((tau_r - tau_p) * 0.5, -tmax, tmax)
            m_b = torch.clamp((tau_r + tau_p) * 0.5, -tmax, tmax)
            tau[..., pitch_idx] = m_b - m_a
            tau[..., roll_idx] = m_a + m_b
        result.joint_efforts = tau
        return result


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit actuator with delayed command application.

    This class extends the :class:`ImplicitActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    """

    cfg: DelayedImplicitActuatorCfg

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)
