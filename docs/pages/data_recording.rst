Data Recording and GR00T Post-Training
=======================================

Record demonstration data from an RL specialist policy and use it to fine-tune a GR00T
vision-language-action model.


Pipeline Overview
-----------------

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  1. Record       │───>│  2. Convert     │───>│  3. Fine-tune   │───>│  4. Evaluate    │
   │  (RL Policy)     │    │  (HDF5→LeRobot) │    │  (GR00T)        │    │  (Closed-loop)  │
   └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        Isaac Lab              Python              Isaac-GR00T            Isaac Lab


Prerequisites
-------------

- Trained RL policy checkpoint (from ``scripts/train.py``)
- Isaac Lab environment with the ``G1-PickPlace-Tracking-v0-Record`` task
- `Isaac-GR00T-N1.5 <https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release>`_ repository
  for fine-tuning


Step 1: Record Demonstration Data
----------------------------------

Set up the RL environment with the appropriate observation group for data collection. The
``G1-PickPlace-Tracking-v0-Record`` environment captures RGB images, proprioceptive states,
and actions from an RL policy.

.. code-block:: bash

   python scripts/record.py \
       --task G1-PickPlace-Tracking-v0-Record \
       --checkpoint <path/to/rl/checkpoint.pt> \
       --record \
       --record_output data/recording \
       --num_envs 100 \
       --num_steps 300 \
       --enable_cameras

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Description
   * - ``--task``
     - Environment with camera sensor
   * - ``--checkpoint``
     - Path to trained RL policy
   * - ``--record``
     - Enable HDF5 recording
   * - ``--record_output``
     - Output directory (creates ``data.h5`` inside)
   * - ``--num_envs``
     - Parallel environments (more = faster collection)
   * - ``--num_steps``
     - Total simulation steps to run

**Output:** ``data/recording/data.h5`` containing episodes with observations and actions.


Step 2: Convert to GR00T Format
---------------------------------

Convert the HDF5 dataset to
`LeRobot-compatible format <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md>`_
for GR00T training:

.. code-block:: bash

   python scripts/data_recording/convert_to_gr00t.py \
       -i data/recording/data.h5 \
       -o data/gr00t \
       --task "pick up the object"

This generates:

- ``meta/`` — Dataset metadata (episodes.jsonl, info.json, tasks.jsonl)
- ``videos/`` — MP4 videos from RGB observations
- ``data/`` — Parquet files with states and actions


Step 3: Configure Data Pipeline
---------------------------------

Configure the data modalities for GR00T model training. The modality configuration specifies
which observation channels (state, video, actions) are used and how they map to the training
pipeline. Edit the ``modality.json`` file in the GR00T dataset directory:

.. code-block:: json

   {
     "state": {
       "base_ang_vel": { "start": 0, "end": 3 },
       "joint_pos": { "start": 6, "end": 23 },
       "joint_vel": { "start": 23, "end": 38 }
     },
     "action": {
       "action": { "start": 0, "end": 21 }
     },
     "video": {
       "image": { "original_key": "observation.images.image" }
     },
     "annotation": {
       "human.action.task_description": {}
     }
   }

Next, add a corresponding data config class to ``gr00t/experiment/data_config.py`` in the
Isaac-GR00T repository and register it in ``DATA_CONFIG_MAP``:

.. code-block:: python

   DATA_CONFIG_MAP["g1_pp_tracking_sim"] = G1PPTrackingSimDataConfig()


Step 4: Fine-tune GR00T
------------------------

Run fine-tuning in the
`Isaac-GR00T-N1.5 <https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release>`_ repository:

.. code-block:: bash

   python scripts/gr00t_finetune.py \
       --dataset-path <path/to/gr00t/data> \
       --data-config g1_pp_tracking_sim \
       --video-backend torchvision_av \
       --num-gpus 1 \
       --max-steps 10000 \
       --output-dir outputs/gr00t-pp-tracking

.. tip::

   Increase ``--max-steps`` and add more data for better performance.


Step 5: Closed-Loop Evaluation
-------------------------------

**Launch GR00T N1.5 Inference Server** (in the Isaac-GR00T repository):

.. code-block:: bash

   python scripts/inference_service.py \
       --server \
       --model_path <path/to/finetuned/checkpoint> \
       --embodiment-tag new_embodiment \
       --data-config g1_pp_tracking_sim \
       --denoising-steps 4 \
       --port 6666

**Run Simulation Client** (in this repository):

.. code-block:: bash

   python scripts/record.py \
       --task G1-PickPlace-Tracking-v0-GR00T-Inference \
       --gr00t \
       --gr00t_host localhost \
       --gr00t_port 6666 \
       --gr00t_task_description "Pick up object" \
       --gr00t_action_horizon 4 \
       --num_envs 1 \
       --enable_cameras

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--gr00t``
     - Use GR00T policy instead of RL checkpoint
   * - ``--gr00t_host``
     - Inference server hostname
   * - ``--gr00t_port``
     - Inference server port
   * - ``--gr00t_task_description``
     - Language instruction for the task
   * - ``--gr00t_action_horizon``
     - Steps to execute per action chunk
