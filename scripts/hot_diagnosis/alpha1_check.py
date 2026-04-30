"""Hypothetical α=1.0 clamp check on a rollout.

Asks: "If the run had used action_scale=1.0, how often and on which motors
would the policy's raw output have hit the motor-space limits?"

Lets you decide, independent of the α you actually ran at, whether the
policy is asking for impossible positions (policy OOD) or is just being
throttled (policy reasonable, α is just held low for safety).

Two paths:
  - New runs (runtime_debug_ctrl.csv has ``alpha1_*`` columns): just summarises.
  - Old runs (no alpha1_* cols): recomputes from ``action_pre_scale`` + env.yaml.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Robot constants (copied from the Pi's root_constant.py so the script is
# self-contained; matches the OLD tight JOINT_LIMITS_DEG -- if these change on
# the Pi, update here too).
# -----------------------------------------------------------------------------
JOINT_LIMITS_DEG = {
    1: (-209.123, -47.685),  # L_hipz
    2: (-50.887, +50.934),  # L_hipx
    3: (-168.065, 0),  # L_hipy
    4: (0.978, +112.172),  # L_knee
    5: (-89.753, +20.042),  # L_ankle1 (coupled)
    6: (-21.14, +285.830),  # L_ankle2 (coupled)
    7: (+64.894, +215.255),  # R_hipz
    8: (-50.0, +50.0),  # R_hipx (widened to match sim soft limit on 2026-04-24)
    9: (-0.0, +162.438),  # R_hipy
    10: (-98.105, -0.693),  # R_knee
    11: (-21.50, +87.972),  # R_ankle1 (coupled)
    12: (-78.191, +16.9),  # R_ankle2 (coupled)
}
COMMAND_MARGIN_DEG = 1.0

# calibrated_deg = sign * raw_motor_deg + offset_deg   (so raw = (cal - off) / sign)
MOTOR_SIGN = {
    1: +1.0,
    2: +1.0,
    3: +1.0,
    4: -1.0,
    5: -1.0,
    6: -1.0,
    7: +1.0,
    8: +1.0,
    9: +1.0,
    10: +1.0,
    11: -1.0,
    12: -1.0,
}
MOTOR_OFFSET_DEG = {
    1: +132.68,
    2: +19.394,
    3: +88.096,
    4: +57.352,
    5: 0.0,
    6: 0.0,
    7: -132.68,
    8: -19.394,
    9: -88.096,
    10: +57.352,
    11: 0.0,
    12: 0.0,
}

# Policy action index → joint name (matches POLICY_ACTION_KEYS in RL_agent_isolated.py).
POLICY_JOINT_NAMES = [
    "right_hipz",
    "right_hipx",
    "right_hipy",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hipz",
    "left_hipx",
    "left_hipy",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
]

# Direct (non-coupled) joints: idx → motor_id
DIRECT_IDX_TO_MOTOR = {
    0: 7,  # right_hipz
    1: 8,  # right_hipx
    2: 9,  # right_hipy
    3: 10,  # right_knee
    6: 1,  # left_hipz
    7: 2,  # left_hipx
    8: 3,  # left_hipy
    9: 4,  # left_knee
}

# Coupled ankle pairs: (pitch_idx, roll_idx) → (motor_pitch, motor_roll)
# coupling (with sign_pitch=-1, sign_roll=+1, offset=0):
#   pitch_deg = -(m_p - m_r)/2  →  m_p = roll - pitch,  m_r = roll + pitch
ANKLE_COUPLES = [
    (10, 11, 5, 6),  # left:  idx 10=pitch, 11=roll -> m5, m6
    (4, 5, 11, 12),  # right: idx 4=pitch,  5=roll  -> m11, m12
]


def joint_rad_to_motor_raw_deg(joint_rad: float, motor_id: int) -> float:
    joint_deg = math.degrees(joint_rad)
    return (joint_deg - MOTOR_OFFSET_DEG[motor_id]) / MOTOR_SIGN[motor_id]


def motor_outside_limits(motor_id: int, raw_deg: float) -> tuple[bool, float, float, float]:
    lo, hi = JOINT_LIMITS_DEG[motor_id]
    lo_safe = lo + COMMAND_MARGIN_DEG
    hi_safe = hi - COMMAND_MARGIN_DEG
    if raw_deg < lo_safe:
        return True, lo_safe - raw_deg, lo_safe, hi_safe
    if raw_deg > hi_safe:
        return True, raw_deg - hi_safe, lo_safe, hi_safe
    return False, 0.0, lo_safe, hi_safe


def check_alpha1_for_action(action_pre_scale_rad: np.ndarray) -> dict:
    """For one policy tick's 12-dim action vector, return per-motor overshoot
    at α=1.0 with use_default_offset=True, scale=1.0, all defaults=0 rad."""
    per_motor = {}

    # Direct joints.
    for idx, mid in DIRECT_IDX_TO_MOTOR.items():
        joint_cmd_rad = float(action_pre_scale_rad[idx])  # default=0, α=1, scale=1
        raw = joint_rad_to_motor_raw_deg(joint_cmd_rad, mid)
        outside, overshoot, lo, hi = motor_outside_limits(mid, raw)
        per_motor[mid] = {
            "raw_deg": raw,
            "overshoot_deg": overshoot,
            "outside": outside,
            "lo": lo,
            "hi": hi,
        }

    # Ankle couplings.
    for pitch_idx, roll_idx, m_pitch, m_roll in ANKLE_COUPLES:
        pitch_deg = math.degrees(float(action_pre_scale_rad[pitch_idx]))
        roll_deg = math.degrees(float(action_pre_scale_rad[roll_idx]))
        raw_p = roll_deg - pitch_deg  # from pitch = -(m_p - m_r)/2, roll = (m_p + m_r)/2
        raw_r = roll_deg + pitch_deg
        for mid, raw in ((m_pitch, raw_p), (m_roll, raw_r)):
            outside, overshoot, lo, hi = motor_outside_limits(mid, raw)
            per_motor[mid] = {
                "raw_deg": raw,
                "overshoot_deg": overshoot,
                "outside": outside,
                "lo": lo,
                "hi": hi,
            }

    would_clamp = any(v["outside"] for v in per_motor.values())
    worst = max(per_motor.items(), key=lambda kv: kv[1]["overshoot_deg"])
    return {
        "would_clamp": would_clamp,
        "n_events": sum(1 for v in per_motor.values() if v["outside"]),
        "worst_mid": int(worst[0]) if worst[1]["outside"] else 0,
        "worst_overshoot_deg": float(worst[1]["overshoot_deg"]) if worst[1]["outside"] else 0.0,
        "per_motor": per_motor,
    }


def _latest_match(dir_path: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(glob.glob(str(dir_path / pat)), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if matches:
            return Path(matches[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=5, help="how many worst-overshoot ticks to print")
    args = ap.parse_args()

    ctrl_path = _latest_match(
        args.rollout_dir,
        [
            "runtime_debug_ctrl_*.csv",
            "runtime_debug_ctrl.csv",
        ],
    )
    if ctrl_path is None:
        print(f"[alpha1] no runtime_debug_ctrl CSV found in {args.rollout_dir}")
        return 2
    print(f"[alpha1] using {ctrl_path}")

    df = pd.read_csv(ctrl_path)
    n = len(df)
    print(f"[alpha1] {n} policy ticks")

    if "alpha1_would_clamp" in df.columns:
        print("[alpha1] NEW format detected: summarising agent-side logged alpha1_* columns")
        _summarise_native(df)
    else:
        print("[alpha1] OLD format (no alpha1_* cols): recomputing from action_pre_scale")
        _summarise_recomputed(df, args.topk)

    return 0


def _summarise_native(df: pd.DataFrame) -> None:
    triggered = int((df["alpha1_would_clamp"].fillna(0) > 0).sum())
    print(f"\nTicks where α=1 would clamp: {triggered} / {len(df)} ({100.0 * triggered / max(1, len(df)):.1f}%)")
    if triggered == 0:
        return
    by_mid = df[df["alpha1_would_clamp"] > 0]["alpha1_worst_mid"].astype(int).value_counts()
    by_reason = df[df["alpha1_would_clamp"] > 0]["alpha1_worst_reason"].value_counts()
    print("\nBy worst_mid (how often each motor was the worst offender):")
    for mid, cnt in by_mid.head(12).items():
        print(f"  m{mid:<2}: {int(cnt)}")
    print("\nBy worst_reason:")
    for reason, cnt in by_reason.items():
        print(f"  {reason!s:>10}: {int(cnt)}")
    print(f"\nmax |overshoot| seen: {df['alpha1_worst_overshoot_deg'].abs().max():.2f}°")
    print(
        f"mean |overshoot| on triggered ticks: {df[df['alpha1_would_clamp'] > 0]['alpha1_worst_overshoot_deg'].abs().mean():.2f}°"
    )


def _summarise_recomputed(df: pd.DataFrame, topk: int) -> None:
    if "action_pre_scale" not in df.columns:
        print("[alpha1] no action_pre_scale column -- aborting")
        return
    rows = []
    per_motor_overshoot = dict.fromkeys(range(1, 13), 0.0)
    per_motor_trigger_count = dict.fromkeys(range(1, 13), 0)
    would_clamp_count = 0
    for i, raw_str in enumerate(df["action_pre_scale"]):
        if not isinstance(raw_str, str) or not raw_str.startswith("["):
            continue
        act = np.asarray(json.loads(raw_str), dtype=np.float32)
        if act.size < 12:
            continue
        res = check_alpha1_for_action(act[:12])
        if res["would_clamp"]:
            would_clamp_count += 1
            for mid, info in res["per_motor"].items():
                if info["outside"]:
                    per_motor_trigger_count[mid] += 1
                    per_motor_overshoot[mid] = max(per_motor_overshoot[mid], info["overshoot_deg"])
        rows.append((i, res["would_clamp"], res["worst_mid"], res["worst_overshoot_deg"]))

    n = len(rows)
    if n == 0:
        print("[alpha1] no rows with parseable actions")
        return
    print(f"\nTicks where α=1 would clamp: {would_clamp_count} / {n} ({100.0 * would_clamp_count / max(1, n):.1f}%)")

    print("\nPer-motor trigger count (how many ticks α=1 would push that motor out):")
    names = {
        1: "L_hipz",
        2: "L_hipx",
        3: "L_hipy",
        4: "L_knee",
        5: "L_apitch",
        6: "L_aroll",
        7: "R_hipz",
        8: "R_hipx",
        9: "R_hipy",
        10: "R_knee",
        11: "R_apitch",
        12: "R_aroll",
    }
    for mid in range(1, 13):
        c = per_motor_trigger_count[mid]
        m = per_motor_overshoot[mid]
        if c > 0:
            print(f"  m{mid:<2} {names[mid]:>9}:  ticks={c:6d}  ({100.0 * c / n:5.1f}%)   max_overshoot={m:6.2f}°")

    # Top-k worst ticks
    rows.sort(key=lambda r: -r[3])
    print(f"\nTop {topk} worst overshoots (retrospective α=1 hypothetical):")
    for i, clamp, mid, over in rows[:topk]:
        if clamp:
            print(f"  tick {i}: m{mid} overshoot={over:.2f}°")


if __name__ == "__main__":
    raise SystemExit(main())
