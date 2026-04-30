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

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.utils import configclass

from agile.rl_env.mdp.actuators.actuators import (
    CoupledAnkleDelayedDCMotor,
    DelayedDCMotor,
    DelayedImplicitActuator,
)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for delayed direct control (DC) motor actuator model."""

    class_type: type = DelayedDCMotor

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""  # noqa: E501

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""  # noqa: E501


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class CoupledAnkleDelayedDCMotorCfg(DelayedDCMotorCfg):
    """DelayedDCMotor with a post-compute diamond clamp on coupled ankle pairs.

    For a pair of ankle joints (pitch, roll) driven by two linked motors with
    position mapping ``q_pitch = -(m_a - m_b)/2``, ``q_roll = (m_a + m_b)/2``,
    virtual work gives motor torques ``τ_a = (τ_roll - τ_pitch)/2`` and
    ``τ_b = (τ_roll + τ_pitch)/2``. Clamping each motor to ``motor_tmax``
    produces the diamond ``|τ_roll ± τ_pitch| ≤ 2·motor_tmax`` at the joints,
    which is the correct real-hardware envelope.
    """

    class_type: type = CoupledAnkleDelayedDCMotor

    ankle_pairs: list[tuple[str, str]] = []
    """List of ``(pitch_joint_name, roll_joint_name)`` for each coupled motor pair.
    Both names must be resolved inside this actuator's joint scope."""

    motor_tmax: float = 5.5
    """Per-motor torque limit (Nm) used by the diamond clamp."""
