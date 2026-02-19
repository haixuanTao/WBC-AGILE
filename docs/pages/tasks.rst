Tasks
=====

This page describes all available reinforcement learning tasks, organized by behavior type
and robot model. For the overall design philosophy, see :doc:`architecture`.


Folder Structure
----------------

.. code-block:: text

   tasks/
   ├── <task_category>/          # e.g., locomotion, stand_up, debug
   │   ├── __init__.py           # Imports robot-specific modules
   │   └── <robot>/              # e.g., g1, t1
   │       ├── __init__.py       # Registers gym environments (task IDs)
   │       ├── *_env_cfg.py      # Environment configuration(s)
   │       └── agents/           # RL algorithm configurations
   │           ├── __init__.py
   │           └── rsl_rl_ppo_cfg.py

- **Task ID Registration:** Each ``<robot>/__init__.py`` registers gym environments with
  unique task IDs (e.g., ``Velocity-T1-v0``).
- **Environment Config:** Defines the scene, robot, observations, rewards, terminations, and
  curriculum.
- **Agent Config:** Specifies RL algorithm hyperparameters (network architecture, learning
  rates, etc.).


Locomotion
----------

Velocity tracking tasks where robots learn to follow commanded linear and angular velocities.

**Unitree G1 Robot**

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 15 15 15

   * - Task ID
     - Controlled Joints
     - Policy Type
     - Commands
     - Observations
     - Actuator Model
   * - ``Velocity-G1-History-v0``
     - Legs + Waist Roll/Pitch (14 joints)
     - Teacher (Non-Privileged)
     - Velocity (x, y, yaw)
     - History (5 steps)
     - Delayed DC Motor

**Booster T1 Robot**

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 15 15 15

   * - Task ID
     - Controlled Joints
     - Policy Type
     - Commands
     - Observations
     - Actuator Model
   * - ``Velocity-T1-v0``
     - Legs only (12 joints)
     - Teacher (Non-Privileged)
     - Velocity (x, y, yaw)
     - History (5 steps)
     - Delayed DC Motor

.. note::

   The G1 and T1 locomotion environments share nearly identical MDP configurations,
   differing only in robot-specific joints and links. This demonstrates that the same
   training setup can be applied to different robot platforms with minimal modifications.


Locomotion with Height Commands
-------------------------------

Extended velocity tracking tasks that include height tracking. The teacher policy uses
privileged information (terrain height scans), while student policies are distilled for
deployment.

**Unitree G1 Robot**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Task ID
     - Controlled Joints
     - Policy Type
     - Commands
     - Observations
     - Actuator Model
   * - ``Velocity-Height-G1-v0``
     - Legs only (12 joints)
     - Teacher (Privileged)
     - Velocity (x, y, yaw) + Height
     - No history
     - Delayed Implicit Actuator
   * - ``Velocity-Height-G1-Distillation-Recurrent-v0``
     - Legs only (12 joints)
     - Student (Recurrent LSTM)
     - Velocity (x, y, yaw) + Height
     - Recurrent
     - Delayed Implicit Actuator
   * - ``Velocity-Height-G1-Distillation-History-v0``
     - Legs only (12 joints)
     - Student (History stacking)
     - Velocity (x, y, yaw) + Height
     - History (5 steps)
     - Delayed Implicit Actuator

.. note::

   These G1 tasks control only the leg joints while leaving the waist joint uncontrolled
   by the policy. This provides the upper body IK policy more freedom to expand the workspace
   while maintaining accuracy.


Stand Up
--------

Tasks for learning to recover from arbitrary fallen poses and stand up. Uses full-body control
with a lifting assistance mechanism during training.

**Booster T1 Robot**

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 15 15 15

   * - Task ID
     - Controlled Joints
     - Policy Type
     - Commands
     - Observations
     - Actuator Model
   * - ``StandUp-T1-v0``
     - Full body (all joints)
     - Teacher (Non-Privileged)
     - None
     - History (5 steps)
     - Delayed DC Motor

**Pre-collected Fallen States**

The stand-up task uses a pre-collected dataset of fallen robot states for efficient episode
resets. Instead of simulating the robot falling at the start of each episode (~2 seconds),
states are sampled from the dataset for instant resets.

- **Automatic collection:** The ``pre_learn`` hook automatically collects/loads fallen states
  before training starts.
- **Caching:** Collected states are cached to disk and reused across training runs (cache
  invalidates when terrain config changes).
- **Validation:** Use the following to visualize collected poses:

  .. code-block:: bash

     python scripts/play.py --task StandUp-T1-v0 --validate-fallen-states


Pick and Place
--------------

Motion tracking tasks for learning pick-and-place manipulation behaviors. The robot learns
to track reference trajectories for the upper body while maintaining stable lower body control
via a pre-trained locomotion policy.

**Unitree G1 Robot**

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 15 15 15

   * - Task ID
     - Controlled Joints
     - Policy Type
     - Commands
     - Observations
     - Actuator Model
   * - ``G1-PickPlace-Tracking-v0``
     - Right arm + hand + waist (16 joints)
     - Teacher (Non-Privileged)
     - Trajectory tracking
     - Current state
     - Implicit Actuator
   * - ``G1-PickPlace-Tracking-v0-Debug``
     - Right arm + hand + waist (16 joints)
     - Evaluation
     - Trajectory tracking
     - Current state
     - Implicit Actuator (GUI-tunable PD gains)

Key features:

- **Modular Policy Architecture:** Upper body (manipulation) and lower body (locomotion) are
  controlled by separate policies.
- **Reference Trajectory Tracking:** Upper body tracks pre-recorded motion trajectories from
  YAML files.
- **Pre-trained Locomotion:** Lower body uses a frozen pre-trained locomotion policy.
- **Object Interaction:** Scene includes a table and manipulable object.

.. code-block:: bash

   # Training
   python scripts/train.py --task G1-PickPlace-Tracking-v0

   # Evaluate trained policy
   python scripts/eval.py --task G1-PickPlace-Tracking-v0 --checkpoint path/to/checkpoint


Debug Environments
------------------

Special environments for debugging MDP terms, symmetry functions, and robot models with
interactive GUI controls.

**Joint Debug Environment**

Launches two floating robots with a GUI for direct joint control. Actions are mirrored
through symmetry functions between the two robots.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task ID
     - Description
   * - ``Debug-G1-v0``
     - Joint debug environment for Unitree G1
   * - ``Debug-T1-v0``
     - Joint debug environment for Booster T1

**Object Interaction Debug Environment**

Extends the joint debug environment with interactive object pose control.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task ID
     - Description
   * - ``Debug-G1-Object-v0``
     - Object interaction debug for Unitree G1

.. code-block:: bash

   # Joint debug
   python scripts/play.py --task Debug-G1-v0 --num_envs 2

   # Object interaction debug
   python scripts/play.py --task Debug-G1-Object-v0


Adding New Tasks
----------------

1. Create the task directory structure:

   .. code-block:: text

      tasks/<new_task_category>/<robot>/
      ├── __init__.py
      ├── <task_name>_env_cfg.py
      └── agents/
          ├── __init__.py
          └── rsl_rl_ppo_cfg.py

2. Register the environment in ``<robot>/__init__.py``:

   .. code-block:: python

      import gymnasium as gym
      from . import agents

      gym.register(
          id="<TaskName>-<Robot>-v0",
          entry_point="isaaclab.envs:ManagerBasedRLEnv",
          disable_env_checker=True,
          kwargs={
              "env_cfg_entry_point": f"{__name__}.<task_name>_env_cfg:<ConfigClassName>",
              "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:<RunnerConfigClassName>",
          },
      )

3. Update parent ``__init__.py`` files to import the new module.

4. Add the task to the E2E test suite — see :doc:`development` for details.
