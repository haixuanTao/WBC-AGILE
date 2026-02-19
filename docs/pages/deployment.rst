Deployment
==========

This page covers Docker builds for CI and production, as well as sim-to-real deployment.


Docker Build Process
--------------------

The :agile_code_link:`<workflows/Dockerfile>` performs the following:

1. Starts from the ``nvcr.io/nvidia/isaac-lab:2.3.1`` base image.
2. Installs Python dependencies into Isaac Lab's environment.
3. Removes conflicting rsl_rl packages.
4. Installs the custom rsl_rl with TensorDict support.
5. Verifies correct installation.


Sim-to-Real Transfer
--------------------

Policy deployment for sim-to-real transfer currently utilizes NVIDIA's internal deployment
framework, which is planned for public release in the near future.

**Pre-trained policies** are included in the repository for evaluation and deployment. See
:doc:`pretrained_policies` for available policies and usage instructions.


Sim-to-Sim Validation
----------------------

Before deploying to real hardware, validate your policy in an alternative simulator:

1. **Sim-to-MuJoCo:** Transfer and evaluate in MuJoCo to verify cross-simulator consistency.
   See :doc:`sim2mujoco` for details.
2. **Joint-level inspection:** Check for aggressive commands, torque/position limit violations,
   and noisy actions.

.. tip::

   Policies that look smooth in Isaac Sim may be relying on high friction/damping to mask
   noisy actions. Use policy-level regularization (action rate penalties, l2c2 regularization)
   to ensure the policy *itself* is smooth. See :doc:`lessons_learned` for more practical
   advice.
