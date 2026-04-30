"""Trigger go_to_zero_and_hold on the Pi, then pause for visual validation.

The Pi script ramps motors with low gains (set via `apply_base_gain_profile`)
so the robot settles toward zero gently instead of snapping. After the ramp
and hold the Pi script stops motors and exits; we then wait for the user to
confirm the pose looks correct before continuing the diagnosis.
"""
from __future__ import annotations

import argparse
import subprocess

DEFAULT_PI = "lerobot@172.18.129.122"
DEFAULT_REMOTE = "/home/lerobot/devel/lerobot-humanoid-runtime/control/policy/velocity_v16_iter124750"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=DEFAULT_PI)
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    ap.add_argument("--ramp-s", type=float, default=4.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--no-stop-after", action="store_true",
                    help="Keep the Pi holding zero indefinitely (Ctrl-C on Pi to stop).")
    args = ap.parse_args()

    remote_cmd = (
        f"cd {args.remote_dir} && "
        f"python3 go_to_zero_and_hold.py --ramp-s {args.ramp_s} --hold-s {args.hold_s}"
    )
    if args.no_stop_after:
        remote_cmd += " --no-stop-after"

    print(f"[go_to_zero] ssh {args.pi} :: {remote_cmd}")
    rc = subprocess.call(["ssh", "-t", args.pi, remote_cmd])
    if rc != 0:
        print(f"[go_to_zero] remote exited rc={rc}")
        return rc

    input("[go_to_zero] Robot should be at zero pose. Verify visually, "
          "then press Enter to continue. ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
