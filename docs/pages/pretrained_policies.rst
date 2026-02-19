Pre-trained Policies
====================

AGILE includes several verified pre-trained checkpoints for evaluation and deployment.


Directory Structure
-------------------

.. code-block:: text

   agile/data/policy/
   ├── velocity_g1/              # G1 – Velocity tracking (TorchScript)
   ├── velocity_height_g1/       # G1 – Velocity + height (TorchScript + Checkpoint)
   │   ├── exported/             # Exported student policy (TorchScript + ONNX)
   │   ├── *_teacher.pt          # Teacher policy (TorchScript)
   │   ├── *_student.pt          # Student policy (TorchScript)
   │   └── *_student_checkpoint.pt  # Student training checkpoint (State dict)
   └── velocity_t1/              # T1 – Velocity tracking (TorchScript)


Available Policies
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 10 25

   * - Policy
     - Task
     - Commands
     - Format
     - Description
   * - ``velocity_g1/unitree_g1_velocity_history.pt``
     - ``Velocity-G1-History-v0``
     - v_x, v_y, w_z
     - TorchScript
     - History-based
   * - ``velocity_height_g1/unitree_g1_velocity_height_teacher.pt``
     - ``Velocity-Height-G1-v0``
     - v_x, v_y, w_z
     - TorchScript
     - Privileged teacher
   * - ``velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt``
     - ``Velocity-Height-G1-Distillation-Recurrent-v0``
     - v_x, v_y, w_z, h_root
     - TorchScript
     - Recurrent LSTM student
   * - ``velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt``
     - ``Velocity-Height-G1-Distillation-Recurrent-v0``
     - v_x, v_y, w_z, h_root
     - State dict
     - Training checkpoint (batched eval)
   * - ``velocity_t1/booster_t1_velocity_v0.pt``
     - ``Velocity-T1-v0``
     - v_x, v_y, w_z, h_root
     - TorchScript
     - History-based

.. note::

   Root linear velocity is considered privileged information, as accurate estimation usually
   requires additional hardware during deployment. Only the velocity-height teacher policy
   accesses this information; all other policies do not rely on it and are suitable for
   direct deployment on real robots.


Policy Formats
--------------

**TorchScript** (``.pt`` + ``.yaml``):
Self-contained exported policies ready for deployment, with normalizer included.
Load with ``torch.jit.load()``.

**State dict / Checkpoint** (``.pt`` only):
Training checkpoints containing ``model_state_dict``, ``optimizer_state_dict``, and training
iteration. Supports recurrent policies and batched evaluation. Load with ``torch.load()``.
Required for resuming training.


Usage
-----

.. code-block:: bash

   # TorchScript policies (auto-detected)
   python scripts/eval.py --task Velocity-G1-History-v0 \
       --checkpoint agile/data/policy/velocity_g1/unitree_g1_velocity_history.pt

   # State dict checkpoint (for batched evaluation / resuming training)
   python scripts/eval.py --task Velocity-Height-G1-Distillation-Recurrent-v0 \
       --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt

The evaluation script automatically:

- Loads TorchScript models directly (fast inference, self-contained).
- Falls back to state dict loading for checkpoint files (supports batched evaluation,
  resuming training).
- Exports policies to the ``exported/`` folder (TorchScript + ONNX).
