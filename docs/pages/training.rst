Training
========

This page covers the training workflow, teacher-student distillation, and hyperparameter
sweeps with Weights & Biases.


Basic Training
--------------

Most training configuration lives in the corresponding ``rsl_rl_ppo_cfg.py`` file for each
task. Many options can be overridden via CLI. Run with ``-h`` for full help:

.. code-block:: bash

   python scripts/train.py -h

For local training with W&B logging:

.. code-block:: bash

   python scripts/train.py \
       --task Velocity-T1-v0 \
       --num_envs 4096 \
       --headless \
       --logger wandb \
       --log_project_name Velocity-T1-v0 \
       --run_name test

.. tip::

   **Experiment Reproducibility:** Training automatically captures and logs lightweight git
   metadata (commit hash, branch, uncommitted changes, and diffs) to your experiment logs.
   When using W&B, this information is uploaded to your run for easy tracking and reproduction.


Environment Validation (Play)
-----------------------------

Before training, validate the task including the scene, actions, and MDP functions using
``scripts/play.py``, which runs the environment with sinusoidal test actions (no policy
required):

.. code-block:: bash

   python scripts/play.py --task Velocity-T1-v0 --num_envs 2


Teacher-Student Distillation
----------------------------

**Teacher Training**

Training a teacher policy with privileged observations is often more effective than directly
training a deployable policy using noisy and partially observable inputs. Train a teacher
following the standard training procedure, adding any useful observations and removing noise.
Once training is complete, export the policy using the eval script.

**Student Distillation**

After obtaining the exported teacher policy (``.pt`` file), distill it into a student policy
that uses realistic (deployable) observations. Configure the distillation process as follows:

.. code-block:: python

   @configclass
   class DistillationRunnerCfg(TeacherPpoRunnerCfg):
       algorithm = RslRlDistillationAlgorithmCfg(
           num_learning_epochs=5,
           gradient_length=15,
           learning_rate=1e-3,
           max_grad_norm=1.0,
           loss_type="mse",
       )
       policy = RslRlStudentTrainedTeacherCfg(
           class_name="StudentTrainedTeacher",  # or "StudentTrainedTeacherRecurrent"
           teacher_path="/path/to/exported/teacher_policy.pt",
           student_hidden_dims=[256, 256, 128],
           activation="elu",
       )

In the environment configuration, define separate observation dictionaries:

- ``policy``: for student observations
- ``teacher``: for teacher observations (corresponds to ``critic`` in RL training and matches
  what was defined as ``policy`` observations during teacher training)

Register the task as a standard ``rsl_rl`` task and start training. During distillation, the
reward is not used for optimization — it is still logged for reference.

.. tip::

   Training the student as a recurrent network is often beneficial as it helps cope with
   noise and partial observability.


Hyperparameter Sweep
--------------------

AGILE integrates with `W&B Sweeps <https://docs.wandb.ai/models/sweeps>`_ for hyperparameter
optimization.

**Configure the sweep** by editing :agile_code_link:`<scripts/wandb_sweep/sweep.yaml>`:

.. code-block:: yaml

   method: bayes
   entity: YOUR_WANDB_ENTITY
   metric:
     name: Metrics/base_velocity/error_vel_xy
     goal: minimize

   command:
     - python
     - scripts/wandb_sweep/train_wrapper.py
     - --num_envs
     - 64
     - --task
     - Velocity-G1-History-v0
     - --logger
     - wandb
     - --headless

   parameters:
     agent.algorithm.learning_rate:
       distribution: log_uniform_values
       min: 1.0e-4
       max: 1.0e-3

**Initialize the sweep:**

.. code-block:: bash

   python scripts/wandb_sweep/init_sweep.py --project_name my_robot_optimization

**Start sweep agents:**

.. code-block:: bash

   # Single agent running 1 experiment
   python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 1

   # Multiple experiments sequentially
   python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 4

The ``--agent_count`` parameter specifies how many experiments a single agent runs
**sequentially**. For parallel optimization, launch multiple instances of the script on
different machines or terminals.


Scaled-Dictionary Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sweep system supports a special pattern for scaling complex dictionary parameters, useful
for robot actuator configurations:

.. code-block:: yaml

   parameters:
     p_gain_leg:
       distribution: uniform
       min: 0.5    # 50% of baseline
       max: 2.0    # 200% of baseline
       p_gain_leg_cli_path:
         value: env.scene.robot.actuators.legs.stiffness
       p_gain_leg_base_dict:
         value: |
           {
             ".*_hip_yaw_joint": 100.0,
             ".*_hip_roll_joint": 100.0,
             ".*_hip_pitch_joint": 100.0,
             ".*_knee_joint": 200.0,
             "waist.*": 200.0
           }

See :agile_code_link:`<scripts/wandb_sweep/README.md>` for full documentation.
