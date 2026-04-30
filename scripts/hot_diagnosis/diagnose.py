"""End-to-end hot diagnosis.

0. fetch shadow CSVs + policy.onnx + config/env from the Pi
1. (optional) `--go-to-zero` — ramp the real robot to zero and pause for visual OK
2. sim_rollout — run the pulled policy through a MuJoCo rollout; must not fall
3. onnx_replay — feed Pi-logged observations through the pulled onnx, diff the logged actions
4. summary — all-green => OK; else print which step(s) failed

Run from repo root:
    python -m scripts.hot_diagnosis.diagnose \\
        --s2m-config /home/baguette/tmp_eval/config.yaml \\
        --mjcf /home/baguette/tmp_eval/mjcf/robot.xml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

DEFAULT_PI = "lerobot@172.18.129.122"
DEFAULT_REMOTE = "/home/lerobot/devel/lerobot-humanoid-runtime/control/policy/velocity_v16_iter124750"


def _run(cmd: list[str], step: str) -> int:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n=== [{step}] {printable}")
    rc = subprocess.call(cmd)
    print(f"=== [{step}] rc={rc}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=DEFAULT_PI)
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    ap.add_argument("--local-dir", type=Path, default=Path("hot_data"))
    ap.add_argument("--s2m-config", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--go-to-zero", action="store_true",
                    help="Trigger go_to_zero_and_hold on the Pi between fetch and sim check.")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--sim-init-from-state-log", action="store_true",
                    help="Seed sim_rollout from the pulled bipedal_state_log.csv last row.")
    args = ap.parse_args()

    results: dict[str, int] = {}

    if not args.skip_fetch:
        results["fetch"] = _run(
            [PY, str(HERE / "fetch.py"),
             "--pi", args.pi,
             "--remote-dir", args.remote_dir,
             "--local-dir", str(args.local_dir)],
            "fetch",
        )
        if results["fetch"] != 0:
            print("[diagnose] fetch failed — aborting")
            return 2

    if args.go_to_zero:
        results["go_to_zero"] = _run(
            [PY, str(HERE / "go_to_zero.py"),
             "--pi", args.pi,
             "--remote-dir", args.remote_dir],
            "go_to_zero",
        )
        if results["go_to_zero"] != 0:
            print("[diagnose] go_to_zero failed — aborting")
            return 2

    onnx = args.local_dir / "policy.onnx"
    shadow_csv = args.local_dir / "v16_shadow.csv"

    if not args.skip_sim:
        sim_cmd = [PY, "-m", "scripts.hot_diagnosis.sim_rollout",
                   "--policy", str(onnx),
                   "--s2m-config", str(args.s2m_config),
                   "--mjcf", str(args.mjcf),
                   "--out-dir", str(args.local_dir / "sim_rollout")]
        if args.sim_init_from_state_log:
            sim_cmd += ["--init-from-state-log",
                        str(args.local_dir / "bipedal_state_log.csv")]
        results["sim_rollout"] = _run(sim_cmd, "sim_rollout")

    if not args.skip_replay:
        results["onnx_replay"] = _run(
            [PY, str(HERE / "onnx_replay.py"),
             "--shadow-csv", str(shadow_csv),
             "--onnx", str(onnx),
             "--tol", str(args.tol)],
            "onnx_replay",
        )

    print("\n=== SUMMARY")
    for name, rc in results.items():
        verdict = "OK" if rc == 0 else "FAIL"
        print(f"  {name:12s}  rc={rc}  {verdict}")
    if all(rc == 0 for rc in results.values()):
        print("[diagnose] ALL OK — pulled policy is sane in sim and matches the Pi's logs")
        return 0
    print("[diagnose] FAIL — see individual step output above for why")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
