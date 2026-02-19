Evaluation
==========

AGILE provides comprehensive evaluation tools for policy analysis, including motion smoothness
metrics, success rate computation, trajectory logging, deterministic scenarios, and interactive
HTML reports.


Basic Evaluation
----------------

A trained checkpoint is required. Run evaluations using :agile_code_link:`<scripts/eval.py>`:

.. code-block:: bash

   python scripts/eval.py \
       --task <task_name> \
       --checkpoint path/to/model.pt \
       --num_envs 16 \
       --run_evaluation


Trajectory Logging
------------------

Save complete episode data for offline analysis:

.. code-block:: bash

   python scripts/eval.py \
       --task <task_name> \
       --checkpoint path/to/model.pt \
       --num_envs 16 \
       --run_evaluation \
       --save_trajectories

**Output structure:**

.. code-block:: text

   logs/rsl_rl/<experiment_name>/
   ├── trajectories/
   │   ├── episode_000.parquet
   │   ├── episode_001.parquet
   │   └── ...
   └── metrics.json


Deterministic Scenario Evaluation
----------------------------------

Run controlled tests with specific commands instead of random ones:

.. code-block:: bash

   python scripts/eval.py \
       --task <task_name> \
       --checkpoint path/to/model.pt \
       --eval_config agile/algorithms/evaluation/configs/examples/x_velocity_sweep.yaml \
       --run_evaluation \
       --save_trajectories \
       --generate_report

Benefits:

- Reproducible testing
- Systematic capability evaluation
- Per-environment control
- Time-based command changes
- Automatic HTML report generation

**Example scenarios** (in ``configs/examples/``):

- ``x_velocity_sweep.yaml`` — Test forward/backward walking
- ``y_velocity_sweep.yaml`` — Test lateral movement
- ``yaw_rate_sweep.yaml`` — Test turning
- ``height_sweep.yaml`` — Test height control
- ``multi_env_capability_test.yaml`` — Test all capabilities in parallel


HTML Reports
------------

Interactive HTML reports with per-joint and tracking analysis can be generated automatically
during evaluation or manually after the fact.

**Automatic (during evaluation):**

.. code-block:: bash

   python scripts/eval.py \
       --task <task_name> \
       --checkpoint path/to/model.pt \
       --run_evaluation \
       --save_trajectories \
       --generate_report

**Manual (after evaluation):**

.. code-block:: bash

   # All episodes
   python agile/algorithms/evaluation/generate_report.py \
       --log_dir logs/evaluation/task_datetime

   # Specific episodes
   python agile/algorithms/evaluation/generate_report.py \
       --log_dir logs/evaluation/task_datetime \
       --episodes 0,3,5

   # Failed episodes only
   python agile/algorithms/evaluation/generate_report.py \
       --log_dir logs/evaluation/task_datetime \
       --episodes failed

**Report features:**

- **Summary Dashboard** (``index.html``): Success rate, sortable episode table with
  search/filter, tracking error summary plots.
- **Detailed Episode Pages** (``episodes/episode_XXX.html``): Tracking performance
  (lin_vel_x, lin_vel_y, ang_vel_z, height), all joints organized by body part, collapsible
  sections, joint position/velocity limits, interactive Plotly plots.

Output location: ``logs/evaluation/task_datetime/reports/``


Key CLI Options
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--task``
     - Task name (required)
   * - ``--checkpoint``
     - Path to checkpoint (or auto-detects latest from logs)
   * - ``--run_evaluation``
     - Enable PolicyEvaluator
   * - ``--save_trajectories``
     - Save trajectory data to parquet files
   * - ``--trajectory_fields``
     - Specific fields to save (default: all)
   * - ``--num_envs``
     - Number of parallel environments (default: 16)
   * - ``--eval_config``
     - Path to YAML scenario config (optional)
   * - ``--generate_report``
     - Generate HTML report after evaluation (requires ``--save_trajectories``)


Analyzing Trajectories in Python
---------------------------------

.. code-block:: python

   import sys
   sys.path.insert(0, "agile/algorithms/evaluation")
   from plotting import load_episode, load_metadata, plot_joint_trajectories
   import matplotlib.pyplot as plt

   metadata = load_metadata("logs/rsl_rl/experiment")
   df = load_episode("logs/rsl_rl/experiment", episode_id=0)

   fig, axes = plot_joint_trajectories(
       df,
       joint_names=["left_hip_yaw_joint", "right_knee_joint"],
       metadata=metadata,
       show_limits=True,
   )
   plt.show()
