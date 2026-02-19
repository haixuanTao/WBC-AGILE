Sim-to-MuJoCo Transfer
======================

AGILE provides a **generic Sim2MuJoCo framework** that enables seamless policy transfer from
Isaac Lab to MuJoCo simulation. The framework is task-agnostic and automatically handles
observation/action mapping by parsing the exported I/O descriptor YAML file — no code changes
are needed for different tasks.


Quick Start
-----------

1. Export the policy and I/O descriptor from a trained checkpoint.
2. Obtain the robot MJCF from
   `Unitree's official repository <https://github.com/unitreerobotics/unitree_mujoco>`_
   or bring your own.
3. Run evaluation in MuJoCo.

.. code-block:: bash

   python scripts/sim2mujoco_eval.py \
       --checkpoint path/to/policy.pt \
       --config path/to/config.yaml \
       --mjcf unitree_mujoco/unitree_robots/g1/scene_29dof.xml


Step-by-Step Tutorial
---------------------

**1. Export Policy to TorchScript**

.. code-block:: bash

   python scripts/eval.py \
       --task Velocity-G1-History-v0 \
       --checkpoint path/to/checkpoint.pt

This automatically exports ``policy.pt`` in the checkpoint directory's ``exported/`` folder.

**2. Export I/O Descriptors**

.. code-block:: bash

   python scripts/export_IODescriptors.py \
       --task Velocity-G1-History-v0 \
       --output_dir path/to/output

Generates a YAML file describing the observation and action spaces.

**3. Get Robot MJCF**

Official models from `Unitree's MuJoCo repository
<https://github.com/unitreerobotics/unitree_mujoco>`_ are recommended:

.. code-block:: bash

   git clone https://github.com/unitreerobotics/unitree_mujoco.git
   # G1 robot: unitree_mujoco/unitree_robots/g1/g1_29dof.xml

**4. Run Sim2MuJoCo Evaluation**

.. code-block:: bash

   python scripts/sim2mujoco_eval.py \
       --checkpoint path/to/policy.pt \
       --config path/to/config.yaml \
       --mjcf unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
       --duration 10.0

.. tip::

   **Interactive Control:** The sim2mujoco module supports keyboard teleoperation. Use arrow
   keys or I/J/K/L for movement, U/O for turning, and Page Up/Down (or 9/0) for height
   control. Press SPACE to stop. Remove the ``--no-viewer`` flag to enable the interactive
   viewer.
