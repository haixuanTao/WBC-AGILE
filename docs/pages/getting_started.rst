Getting Started
===============

This guide walks you through installing AGILE and running your first training experiment.


Prerequisites
-------------

**Isaac Lab 2.3.1** is required. Follow the official
`installation guide <https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/index.html>`_.
Isaac Sim 5.1 is needed to use the verified USD assets provided in this project.
The conda-based installation is recommended.

.. code-block:: bash

   # Ensure you're using version 2.3.1
   git checkout v2.3.1


Local Development Setup
-----------------------

Clone the repository and install dependencies:

:agile_git_clone_code_block:

.. code-block:: bash

   # Ensure ISAACLAB_PATH is set
   export ISAACLAB_PATH=/path/to/isaac_lab

   # Install all dependencies and packages
   ./scripts/setup/install_deps_local.sh

   # Verify the custom rsl_rl is correctly installed
   ${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py

The :agile_code_link:`<scripts/setup/install_deps_local.sh>` script will:

- Install runtime dependencies (tensordict, wandb, datasets, etc.)
- Remove any conflicting rsl_rl packages from Isaac Lab
- Install the custom rsl_rl with TensorDict support
- Install the agile package


Quick Start
-----------

Get started with AGILE in two simple steps locally.

**1. Train a velocity tracking policy:**

.. code-block:: bash

   python scripts/train.py \
       --task Velocity-T1-v0 \
       --num_envs 2048 \
       --headless

**2. Visualize the trained policy:**

.. code-block:: bash

   python scripts/eval.py \
       --task Velocity-T1-v0 \
       --num_envs 32 \
       --checkpoint <path_to_checkpoint>

.. tip::

   **Try a pre-trained policy:** We provide a variety of pre-trained policies for
   different robots and tasks. For a quick start, try the G1 recurrent student policy.
   It has better tracking performance compared to the velocity-tracking-only policy,
   does not require linear velocity observations, and is ready for direct deployment
   on real hardware. See :doc:`pretrained_policies` for details.


Next Steps
----------

- Explore :doc:`tasks` for the full list of available tasks and robot embodiments.
- Learn about :doc:`training` including teacher-student distillation for robust deployment.
- See :doc:`evaluation` for performance analysis and metrics.
- Check out :doc:`pretrained_policies` for ready-to-use policies.
