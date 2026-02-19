FAQ (Office Hours)
==================

Frequently asked questions from the AGILE robotics livestream office hour session, covering
training workflows, sim-to-real transfer, and extending the framework for custom robots
and tasks.

**Resources:**

- `YouTube Recording <https://www.youtube.com/live/ANvkdrESIuc?si=KPd8PvXFipt8FsG9>`_


Quantitative Evaluation and Sim-to-Real
-----------------------------------------

**Q: How do you define success beyond reward curves?**

AGILE uses both qualitative and quantitative signals:

- **Qualitative:** How natural the motion looks (video review).
- **Quantitative:** Tracking errors (velocity/height), episode survival time, robustness
  under repeated disturbances (pushes), and checks for joint-limit violations. Policies that
  violate joint limits tend to transfer poorly across simulators and to real hardware. AGILE
  also validates sim-to-sim transfer by switching to MuJoCo after Isaac Lab checks.

**Q: What metrics predict sim-to-real transfer best?**

Tracking accuracy alone is not sufficient. Key indicators include:

- Survival time under disturbances
- Joint limit compliance (position/velocity)
- Sim-to-sim validation (including MuJoCo)
- Qualitative review to catch unnatural gaits


Physics, Contact, and Realism
-------------------------------

**Q: How do you ensure realistic friction and contact in simulation?**

- **Robot parameters:** Start from open-source system ID parameters and adjust using a
  train-deploy-tune loop (e.g., PD gains).
- **Object parameters:** Mass/inertia and friction are often rough estimates.
- **Robustness strategy:** Randomize parameters (e.g., friction coefficients) so the policy
  is less sensitive to modeling inaccuracies.

**Q: How do you handle complex contact dynamics for sim-to-real?**

In addition to the physics solver and USD collision setup:

- Randomize friction/contact-related parameters.
- Use contact modeling tricks (e.g., more point-like foot contact rather than idealized flat
  cuboid contact) to better match real-world behavior.


Training Workflow and Fine-Tuning
----------------------------------

**Q: What does "fine-tuning" mean in the AGILE context?**

Here, "fine-tuning" refers to starting from a baseline RL policy and tuning a smaller set of
parameters (often via sweeps) to improve performance (tracking, robustness). This is different
from foundation model fine-tuning (LLM/VLM/VLA).

**Q: How do you run hyperparameter sweeps and keep experiments organized?**

AGILE uses W&B to log runs, track configurations, and compare results. The workflow supports
sweeps over RL hyperparameters and task/environment parameters (MDP terms like rewards,
curriculum, events). See :doc:`training` for details.


Robustness and Error Recovery
-------------------------------

**Q: If the robot does something wrong, can it self-correct?**

During training, disturbances are injected (push torso/hand) and randomization is applied
so the policy experiences varied conditions. For locomotion, upper-body motion is randomized
during training so the lower-body controller learns to cope. However, disturbances beyond
the policy/robot limits can still cause failure.


Embodiments and Adding New Robots
-----------------------------------

**Q: How hard is it to add a new embodiment (e.g., H1)?**

AGILE reduces effort by:

- Keeping MDP terms as robot-agnostic as possible.
- Putting robot-specific details (joint/link names, limits, actuator configs) in a single
  robot asset/config file.
- Emphasizing model verification early (contact geometry, joint limits, PD gains, actuator
  model).

Similar tasks across robots (e.g., G1 vs T1 velocity tracking) can share almost identical
configurations aside from minimal robot-specific parameters.

**Q: What are the biggest causes of poor policies when adding a robot?**

Incorrect model setup: wrong joint scales/ranges, incorrect contact geometry/sensors,
mismatched PD gains. These issues can waste days/weeks if discovered late.


Teleoperation and Deployment
------------------------------

**Q: Is MuJoCo transfer supported?**

Yes. AGILE exports the policy and an I/O descriptor so the same trained policy can be tested
in other simulators (including MuJoCo). See :doc:`sim2mujoco`.

**Q: Beyond keyboard teleop, what devices can be used?**

AGILE is mainly designed for policy training. Trained policies can be copied to Isaac Lab for
teleoperation, which supports keyboard, spacemouse, and Cloud XR devices. See the
`Isaac Lab teleoperation examples
<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.


Joint Control and Actuator Modeling
-------------------------------------

**Q: What control/actuator models are used for joints during training?**

AGILE provides two common actuator setups, both verified with real-world testing:

- A delayed motor configuration to model more realistic actuator behavior (torque-speed
  envelope near limits).
- An implicit actuator setup where the simulator handles integration.


Modular Control and Distillation
----------------------------------

**Q: If lower and upper body are trained separately, how do you prevent "policy exploitation" between controllers?**

The team trains locomotion first, freezes it, then uses it as an API while training the
upper-body policy. This reduces non-stationarity between controllers, similar to the approach
in `VIRAL <https://arxiv.org/pdf/2511.15200>`_.


Domain Randomization
----------------------

**Q: What domain randomization strategy works best for sim-to-sim adaptation?**

See the :doc:`lessons_learned` page for guidance on randomization terms and practical
recommendations. Empirical experience shows that randomization within proper armature and PD
gain ranges matters significantly.
