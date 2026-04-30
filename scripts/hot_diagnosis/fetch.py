"""Pull the latest shadow-mode logs + policy artifacts off the Pi.

Handles both the legacy fixed-name files (e.g. ``bipedal_state_log.csv``) and
the new timestamped ones (``bipedal_state_log_<ts>.csv``) -- for stems that
match multiple remote files it picks the most-recently modified one.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_PI = "lerobot@172.18.129.122"
DEFAULT_REMOTE = "/home/lerobot/devel/lerobot-humanoid-runtime/control/policy/velocity_v16_iter124750"
# Fixed names (taken literally) and latest-of-stem patterns (resolved via ssh).
DEFAULT_FIXED = (
    "policy.onnx",
    "config.yaml",
    "env.yaml",
    "v16_shadow.csv",
)
DEFAULT_STEMS = (
    # (local_name, remote_pattern)
    ("bipedal_state_log.csv", "bipedal_state_log*.csv"),
    ("runtime_debug_ctrl.csv", "runtime_debug_ctrl*.csv"),
    ("action_scale_log.csv", "action_scale_log*.csv"),
)


def _latest_remote(pi: str, remote_dir: str, pattern: str) -> str | None:
    """Return the newest file in ``remote_dir`` matching ``pattern`` (by mtime),
    or None if nothing matches."""
    cmd = ["ssh", pi, f"ls -1t {remote_dir}/{pattern} 2>/dev/null | head -n 1"]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError:
        return None
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=DEFAULT_PI)
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    ap.add_argument("--local-dir", type=Path, default=Path("hot_data"))
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Override -- list of exact remote filenames to fetch (no timestamp resolution).",
    )
    ap.add_argument(
        "--preserve-remote-name",
        action="store_true",
        help="For timestamped stems, save locally using the remote filename (keeps timestamp).",
    )
    args = ap.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    if args.files is not None:
        # Legacy path: literal fetch by name.
        for name in args.files:
            src = f"{args.pi}:{args.remote_dir}/{name}"
            dst = args.local_dir / name
            print(f"[fetch] {src} -> {dst}")
            rc = subprocess.call(["rsync", "-av", "--partial", src, str(dst)])
            if rc != 0:
                print(f"[fetch][WARN] rsync rc={rc} for {name} (skipping)")
                missing.append(name)
    else:
        # Default: fetch the fixed-name files plus the latest-per-stem files.
        for name in DEFAULT_FIXED:
            src = f"{args.pi}:{args.remote_dir}/{name}"
            dst = args.local_dir / name
            print(f"[fetch] {src} -> {dst}")
            rc = subprocess.call(["rsync", "-av", "--partial", src, str(dst)])
            if rc != 0:
                print(f"[fetch][WARN] rsync rc={rc} for {name} (skipping)")
                missing.append(name)

        for local_name, pattern in DEFAULT_STEMS:
            remote_path = _latest_remote(args.pi, args.remote_dir, pattern)
            if remote_path is None:
                print(f"[fetch][WARN] no match for {pattern!r} on remote (skipping)")
                missing.append(pattern)
                continue
            remote_base = Path(remote_path).name
            dst_name = remote_base if args.preserve_remote_name else local_name
            src = f"{args.pi}:{remote_path}"
            dst = args.local_dir / dst_name
            print(f"[fetch] latest match {pattern!r} -> {remote_base} -> {dst}")
            rc = subprocess.call(["rsync", "-av", "--partial", src, str(dst)])
            if rc != 0:
                print(f"[fetch][WARN] rsync rc={rc} for {remote_base} (skipping)")
                missing.append(remote_base)

    print(f"[fetch] wrote -> {args.local_dir}")
    if missing:
        print(f"[fetch] missing on remote: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
