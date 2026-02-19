AGILE
=====

**A Generic Isaac-Lab based Engine for Humanoid Loco-Manipulation Learning**

AGILE provides a comprehensive reinforcement learning framework for training whole-body
control policies with validated sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab,
this toolkit enables researchers and practitioners to develop loco-manipulation behaviors for
humanoid robots.

----

Start Here
----------

.. grid:: 3

   .. grid-item-card:: Getting Started
      :link: pages/getting_started
      :link-type: doc

      Install dependencies, set up Isaac Lab, and train your first policy.

   .. grid-item-card:: Deployment
      :link: pages/deployment
      :link-type: doc

      Docker builds, sim-to-real transfer, and production deployment.

   .. grid-item-card:: FAQ (Office Hours)
      :link: pages/faq
      :link-type: doc

      Answers from the AGILE robotics livestream covering training, sim-to-real, and more.


System Design
-------------

.. grid:: 2

   .. grid-item-card:: Architecture
      :link: pages/architecture
      :link-type: doc

      Project structure, modular policy design, and self-contained task philosophy.

   .. grid-item-card:: Tasks
      :link: pages/tasks
      :link-type: doc

      Available tasks for locomotion, height tracking, stand-up, pick-and-place, and debug.

   .. grid-item-card:: Training
      :link: pages/training
      :link-type: doc

      RL training, teacher-student distillation, and W&B hyperparameter sweeps.

   .. grid-item-card:: Evaluation
      :link: pages/evaluation
      :link-type: doc

      Metrics, trajectory logging, deterministic scenarios, and HTML reports.


Reference
---------

.. grid:: 3

   .. grid-item-card:: Pre-trained Policies
      :link: pages/pretrained_policies
      :link-type: doc

      Ready-to-use checkpoints for G1 and T1 robots with usage instructions.

   .. grid-item-card:: Sim-to-MuJoCo
      :link: pages/sim2mujoco
      :link-type: doc

      Generic framework for policy transfer from Isaac Lab to MuJoCo simulation.

   .. grid-item-card:: Data Recording
      :link: pages/data_recording
      :link-type: doc

      Record demonstrations and fine-tune GR00T vision-language-action models.


Tools & Operations
------------------

.. grid:: 2

   .. grid-item-card:: Lessons Learned
      :link: pages/lessons_learned
      :link-type: doc

      Practical RL tips from robot modeling to reward design to sim-to-real deployment.

   .. grid-item-card:: Development
      :link: pages/development
      :link-type: doc

      Code style, pre-commit hooks, testing, CI pipeline, and contributing.


.. toctree::
   :maxdepth: 1
   :caption: Start Here
   :hidden:

   pages/getting_started
   pages/deployment
   pages/faq

.. toctree::
   :maxdepth: 1
   :caption: System Design
   :hidden:

   pages/architecture
   pages/tasks
   pages/training
   pages/evaluation

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   pages/pretrained_policies
   pages/sim2mujoco
   pages/data_recording

.. toctree::
   :maxdepth: 1
   :caption: Tools & Operations
   :hidden:

   pages/lessons_learned
   pages/development
