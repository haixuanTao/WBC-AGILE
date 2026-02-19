Lessons Learned
===============

A living collection of practical lessons and rules of thumb from the AGILE team's experience
developing humanoid RL policies. These notes are specific to this setup and may not generalize
to very different applications.


Robot Modeling First
--------------------

If the model is wrong, the policy learns the wrong physics — and those "skills" won't survive
on hardware.

**USD model non-negotiables:**

- Mass and inertia are realistic and consistent; verify COM per link and whole body.
- Joint limits match real mechanical limits (position, velocity); include soft limits if
  applicable.
- Geometry is clean and accurate for kinematics/self-collision; avoid interpenetration.
- Contact shapes approximate real surfaces with plausible friction/restitution; prefer simple,
  watertight collision shapes.
- Sensor frames are correct (IMU orientation, camera extrinsics, sign conventions).
- Units are consistent (m, kg, s, N) across the entire model.

**Robot configuration (beyond USD):**

- Set realistic torque/current and velocity limits; clamp in sim exactly as on the robot.
- Task-aware limits: lowering joint velocity limits can stabilize early training.
- Choose plausible friction, damping, and armature baselines.
- Match PD gains and effort saturations to the deployment stack.

**Validate before training:**

- Gravity settle: rests without jitter/drift.
- Drop/impulse tests on key links to check inertials.
- Joint limit sweep: drive to limits; verify clamping/signs/units.
- Zero-command stand: robot should not "creep."
- Contact sanity: visualize contact points/normals.


General Philosophy
------------------

- **Start Simple, Then Iterate:** Start with the simplest possible environment, reward, and
  network (e.g., a 3-layer MLP). Give the agent all observations it might need (even
  privileged ones). Get a basic setup working before thinking about deployment.
- **Be Robust to Seeds:** Do not rely on a few lucky seeds. Always test final hyperparameters
  on ~5 different seeds.


Environment and Reward Design
-----------------------------

**Reward Recipe:** A good starting point is ``Task + Style + Regularization``:

- **Task:** The main goal (e.g., velocity tracking). Start with this; highest weight.
- **Style:** How to do it (e.g., feet parallel, torso orientation). Add if you see ugly
  behavior.
- **Regularization:** What to avoid (e.g., joint limit penalty, torque penalty, action rate).
  Add once the task can be learned. Crucial for sim-to-real transfer.

**Curricula:** Exploration is one of the hardest problems in RL. Good curricula guide the
agent from an easy task to the final complex one:

- **Fading Guidance:** Start with helper aids (external forces, simplified physics, strong
  reward shaping) and gradually fade them out.
- **Increasing Difficulty:** Start with minimal penalties and a simple task. Gradually increase
  difficulty.

**Terminations are Critical:**

- If the agent is in a bad state it can't recover from (e.g., fallen), terminate. This
  massively speeds up learning.
- If ``episode_length`` plots crash to zero, the agent is "suicidal" — termination penalties
  are too high or positive rewards are too low.

**Symmetry:** Only use symmetry augmentation if the skill should be symmetric (e.g., walking).

**Observations:**

- Start with privileged info for both actor and critic.
- Use observation history (stacking) for partial observability.
- The critic can remain privileged, but the actor must eventually use only real-world sensors
  (via distillation).
- Clamp and scale observations — raycasting can return infinite values and some observations
  have high magnitude.


Training and Monitoring
-----------------------

- **Watch Videos!** Plots can be misleading. Record videos at different training stages.

- **PPO's Entropy Coefficient (``ent_coef``):**

  - Too high: the policy maximizes its own noise and never exploits.
  - Too low: the policy converges too quickly to a bad local minimum.
  - Start with the default (e.g., 0.005) and decay to 0.0 over training.

- **Key Plots to Watch:**

  - **Metrics vs. Reward:** If reward goes up but the task metric does not, the agent is
    **reward hacking**.
  - **Losses:** ``value_loss`` should converge to a small value (<1.0). If higher, scale down
    all reward weights.
  - **Curriculum:** Check if the agent reaches the final, hardest stage.
  - **Policy noise:** The action distribution should converge to small variance.


Sim-to-Real and Deployment
---------------------------

- **The "Fake Smoothness" Trap:** A policy may look smooth in simulation because high
  friction/damping is masking noisy, aggressive actions. Real hardware hates this. Use
  policy-level regularization (action rate, velocity, acceleration penalties, l2c2
  regularization).

- **Sim-to-Sim Deploy First:** Before going to hardware, deploy in a test-only simulation.
  Check joint-level commands for aggressiveness and limit violations.

- **Hardware specifics matter:**

  - **PD Gains:** Lower gains can lead to smoother, more compliant actions.
  - **Parallel Joints:** Joints like ankles may require extra regularization and low PD gains.
  - **Timing:** Account for delays in the deployment stack; use delayed actuator models.

- **The #1 Rule of Sim-to-Real:** If hardware behavior looks very different from simulation,
  **fix the simulation to match reality**. Do not hack the simulation to make policy learning
  simpler.


A Final Word
------------

RL can feel like a dark art. There is no single general recipe. A single hyperparameter, a
small reward change, or a minor physics tweak can be the difference between a deployable
policy and a total failure. When stuck, go talk to another "RL wizard" — discussing the
problem is often the fastest way to find a solution.
