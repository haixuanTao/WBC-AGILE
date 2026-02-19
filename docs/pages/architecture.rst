Architecture
============

AGILE (**A** **G**\ eneric **I**\ saac-**L**\ ab based **E**\ ngine) provides a comprehensive
reinforcement learning framework for training whole-body control policies with validated
sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab, this toolkit enables researchers
and practitioners to develop loco-manipulation behaviors for humanoid robots.


Project Structure
-----------------

.. code-block:: text

   agile/                       # Repository root
   ├── agile/                   # Main package
   │   ├── algorithms/          # Algorithms for policy training
   │   │   ├── rsl_rl/          # Custom rsl_rl library with TensorDict support
   │   │   └── evaluation/      # Evaluation and metrics computation
   │   ├── data/                # Data handling and policy checkpoints
   │   ├── isaaclab_extras/     # Isaac Lab extensions and monkey patches
   │   └── rl_env/              # Reinforcement learning environments
   │       ├── assets/          # Robot assets and configurations
   │       ├── mdp/             # MDP components (rewards, commands, actions, etc.)
   │       ├── tasks/           # Task definitions and configurations
   │       ├── tests/           # Unit tests for MDP components
   │       ├── utils/           # Environment utilities
   │       └── rsl_rl/          # RSL-RL integration and wrappers
   ├── docs/                    # Documentation and media files
   ├── scripts/                 # Utility scripts
   │   ├── train.py             # Training script
   │   ├── eval.py              # Evaluation and policy export script
   │   ├── play.py              # Environment validation script (no policy)
   │   ├── setup/               # Installation and setup scripts
   │   ├── wandb_sweep/         # Hyperparameter optimization with W&B
   │   └── data_recording/      # Data recorder and converter for GR00T
   ├── tests/                   # Test suite
   ├── workflows/               # Docker support
   └── pyproject.toml           # Project configuration


Modular Policy Design
---------------------

AGILE uses a modular approach to enable complex loco-manipulation behaviors. The framework
separates **lower body locomotion** (trained via RL) from **upper body control** (IK/IL/Random),
with optional distillation to deployable student policies.

.. image:: ../figures/separate_upper_lower_body_policy_diagram.png
   :alt: Modular Policy Architecture
   :align: center
   :width: 80%

This architecture enables flexible behavior composition and efficient training strategies.
AGILE policies power `Isaac Lab's official teleoperation examples
<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#teleoperation>`_.

.. note::

   This modular architecture represents the current implementation focus for loco-manipulation
   tasks, particularly enabling **teleoperation** where the upper body responds to external
   commands while maintaining stable locomotion. AGILE is not limited to this approach — the
   framework supports various policy architectures including unified full-body control
   (e.g., the stand-up task) and will expand to support additional architectures in future
   releases.


Training Pipeline
^^^^^^^^^^^^^^^^^

**Step 1: Teacher Policy Training**

- **Lower Body (Locomotion):** Trained end-to-end using RL to track velocity, height,
  and other locomotion commands. The policy receives observations from the robot's state
  and environment and outputs joint position targets for the lower body.
- **Upper Body (Manipulation):** Can use different approaches depending on the task:

  - Inverse Kinematics (IK) for simple position-based control
  - Imitation Learning (IL) for more complex manipulation behaviors
  - Random policies for training robust locomotion agnostic to upper body movements

The teacher policy typically has access to privileged information (e.g., ground truth terrain
height, friction coefficients) that may not be available on the real robot, making it powerful
but not directly deployable.

**Step 2: Student Policy Distillation (Optional)**

After training the teacher, its behavior can be distilled into a deployable student policy
that only uses realistic observations available on hardware (e.g., joint positions, velocities,
IMU readings). Two student architectures are supported:

- **Recurrent networks (LSTM/GRU):** Better at handling noise and partial observability
  through temporal memory.
- **MLP with history stacking:** Simpler architecture that concatenates recent observation
  history.

The distillation step can be skipped if the teacher policy already uses only deployable
observations.


Self-Contained Task Design
--------------------------

Each task configuration is **intentionally self-contained** with all MDP components defined
in a single file. This deviates from Isaac Lab's inheritance-based approach based on practical
experience:

- **Transparency and Maintainability:** Complete configuration visible in one file — no
  inheritance tracing needed.
- **Seamless with Isaac Lab:** Works naturally with Isaac Lab's manager-based environment
  architecture.
- **Efficient Collaboration:** Multiple developers can work on different tasks independently
  without conflicts.
- **Faster Iteration:** Changes are localized with immediate, visible impact — no hidden
  side effects.

See :doc:`tasks` for the full list of available tasks and their configurations.


Supported Embodiments
---------------------

The framework has been validated on two humanoid robots:

- **Booster T1**
- **Unitree G1**

Both robot USDs are available in Isaac Sim 5.1. For G1, two actuator configurations are
provided: a delayed DC motor model and an implicit actuator setup adapted from
`BeyondMimic <https://github.com/Beyond-Mimic/BeyondMimic>`_, both verified in sim-to-sim
and sim-to-real transfers.
