"""Export any AGILE checkpoint to ONNX (and accompanying TorchScript .pt).

Usage:
    $ISAACLAB_PATH/isaaclab.sh -p scripts/export_onnx.py \
        --task Velocity-LeRobot-NoArms-v0 \
        --checkpoint logs/rsl_rl/.../model_N.pt \
        [--output /path/to/policy.onnx]
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from pathlib import Path

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry

import agile.rl_env.tasks  # noqa: F401 — register tasks
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx

ckpt_path = Path(args.checkpoint).resolve()
if not ckpt_path.exists():
    sys.exit(f"checkpoint not found: {ckpt_path}")

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
assert isinstance(env_cfg, ManagerBasedRLEnvCfg)

env_cfg.scene.num_envs = 1
env_cfg.observations.policy.enable_corruption = False
# AGILE velocity envs default to concatenate_terms=False, which returns obs as
# a nested TensorDict. Force-flatten for the export path so OnPolicyRunner's
# dimensionality resolver sees a plain tensor.
env_cfg.observations.policy.concatenate_terms = True
env_cfg.observations.policy.flatten_history_dim = True
if hasattr(env_cfg.observations, "critic"):
    env_cfg.observations.critic.concatenate_terms = True
    if hasattr(env_cfg.observations.critic, "flatten_history_dim"):
        env_cfg.observations.critic.flatten_history_dim = True

env = gym.make(args.task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
runner.load(str(ckpt_path))

if args.output:
    out_path = Path(args.output).resolve()
else:
    out_path = ckpt_path.parent / "exported" / f"policy_{ckpt_path.stem}.onnx"

out_path.parent.mkdir(parents=True, exist_ok=True)
export_policy_as_onnx(
    runner.alg.policy,
    normalizer=runner.obs_normalizer,
    path=str(out_path.parent),
    filename=out_path.name,
)

size = out_path.stat().st_size
print(f"[OK] exported ONNX: {out_path} ({size / 1024:.1f} KiB)")

env.close()
app.close()
