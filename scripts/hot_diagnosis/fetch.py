"""Pull the latest shadow-mode logs + policy artifacts off the Pi."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_PI = "lerobot@172.18.129.122"
DEFAULT_REMOTE = "/home/lerobot/devel/lerobot-humanoid-runtime/control/policy/velocity_v16_iter25000"
DEFAULT_FILES = (
    "v16_shadow.csv",
    "bipedal_state_log.csv",
    "policy.onnx",
    "config.yaml",
    "env.yaml",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=DEFAULT_PI)
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    ap.add_argument("--local-dir", type=Path, default=Path("hot_data"))
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = ap.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        src = f"{args.pi}:{args.remote_dir}/{name}"
        dst = args.local_dir / name
        print(f"[fetch] {src} -> {dst}")
        subprocess.check_call(["rsync", "-av", "--partial", src, str(dst)])
    print(f"[fetch] wrote -> {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
