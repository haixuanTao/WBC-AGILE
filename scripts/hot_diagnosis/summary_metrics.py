"""Behavioural summary metrics for a hot-diagnosis run.

Reads ``bipedal_state_log.csv`` (+ optional ``action_scale_log.csv``) and
prints a compact report on base stability, gait cadence, command energy, and
tracking error -- answers questions like "was it standing? stomping?
stable?" without having to plot.

Also optionally prints a tail-window block (default last 15 s) so you can
see behaviour at the final (post-ramp) scale separately from the ramp-up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MOTOR_LABELS = {
    1: "L_hipz",
    2: "L_hipx",
    3: "L_hipy",
    4: "L_knee",
    5: "L_ank1",
    6: "L_ank2",
    7: "R_hipz",
    8: "R_hipx",
    9: "R_hipy",
    10: "R_knee",
    11: "R_ank1",
    12: "R_ank2",
}
UPPER_MOTORS = (1, 2, 3, 7, 8, 9)  # hips only (no knee/ankle)
KNEE_MOTORS = (4, 10)  # L_knee, R_knee -- cadence probes
ANKLE_MOTORS = (5, 6, 11, 12)


@dataclass
class WindowMetrics:
    label: str
    duration_s: float
    n_samples: int
    rate_hz: float


def _zero_crossings_hz(x: np.ndarray, dt_s: float, min_std_deg: float = 2.0) -> float:
    """Cadence estimate: half-cycles/sec of centered signal. For a clean
    sinusoid of frequency f, returns f. Returns 0 when amplitude is below
    ``min_std_deg`` (prevents micro-jitter from reading as "stepping" --
    real stepping produces several degrees of knee-target swing)."""
    if x.size < 4:
        return 0.0
    x = x - np.median(x)
    if np.std(x) < min_std_deg:
        return 0.0
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    duration = (x.size - 1) * dt_s
    if duration <= 0:
        return 0.0
    return 0.5 * crossings / duration  # Hz


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _report(df: pd.DataFrame, dt_s: float, label: str) -> None:
    n = len(df)
    if n < 2:
        print(f"\n--- {label}: too few samples ({n}), skipping ---")
        return
    duration = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    rate = 1.0 / dt_s

    # --- base pose / tilt ---
    pgz = df["projected_gravity_z"].to_numpy(dtype=float)
    tilt_deg = np.degrees(np.arccos(np.clip(-pgz, -1.0, 1.0)))  # 0 = upright
    tilt_mean = float(np.mean(tilt_deg))
    tilt_std = float(np.std(tilt_deg))
    tilt_max = float(np.max(tilt_deg))

    # --- base angular velocity (body-frame gyro) ---
    gx = df["imu_gyro_x_rad_s"].to_numpy(dtype=float)
    gy = df["imu_gyro_y_rad_s"].to_numpy(dtype=float)
    gz = df["imu_gyro_z_rad_s"].to_numpy(dtype=float)
    g_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    gyro_rms = _rms(g_mag)
    gyro_peak = float(np.max(g_mag))

    # --- linear accel magnitude (convention-agnostic: we report mean + std
    # + peak of |a|). mean(|a|) ~ 9.8 means gravity is still in the signal;
    # mean(|a|) ~ 0 means firmware already removed it. Either way std tracks
    # motion/impacts regardless of convention.
    ax = df["imu_lin_acc_x_m_s2"].to_numpy(dtype=float)
    ay = df["imu_lin_acc_y_m_s2"].to_numpy(dtype=float)
    az = df["imu_lin_acc_z_m_s2"].to_numpy(dtype=float)
    a_mag = np.sqrt(ax * ax + ay * ay + az * az)
    acc_mean = float(np.mean(a_mag))
    acc_std = float(np.std(a_mag))
    acc_peak = float(np.max(a_mag))

    # --- gait cadence (knee target zero-crossings) ---
    knee_cadences = {}
    for mid in KNEE_MOTORS:
        col = f"m{mid}_target_pos_deg"
        if col not in df.columns:
            continue
        knee_cadences[mid] = _zero_crossings_hz(df[col].to_numpy(dtype=float), dt_s)
    mean_cadence = float(np.mean(list(knee_cadences.values()))) if knee_cadences else 0.0

    # --- command energy: target-position derivative RMS per motor group ---
    def _group_cmd_rms(motor_ids) -> float:
        vals = []
        for mid in motor_ids:
            col = f"m{mid}_target_pos_deg"
            if col not in df.columns:
                continue
            dtgt = np.diff(df[col].to_numpy(dtype=float)) / dt_s  # deg/s
            vals.append(_rms(dtgt))
        return float(np.mean(vals)) if vals else 0.0

    cmd_upper = _group_cmd_rms(UPPER_MOTORS)
    cmd_knee = _group_cmd_rms(KNEE_MOTORS)
    cmd_ankle = _group_cmd_rms(ANKLE_MOTORS)

    # --- tracking error: target vs measured (raw motor deg) ---
    track_rms_per = {}
    for mid in range(1, 13):
        pcol = f"m{mid}_pos_deg"
        tcol = f"m{mid}_target_pos_deg"
        if pcol not in df.columns or tcol not in df.columns:
            continue
        err = df[tcol].to_numpy(dtype=float) - df[pcol].to_numpy(dtype=float)
        # Ignore rows with NaN/invalid target (happens before target is set).
        err = err[np.isfinite(err)]
        if err.size:
            track_rms_per[mid] = _rms(err)
    worst_track = max(track_rms_per.items(), key=lambda kv: kv[1]) if track_rms_per else (0, 0.0)

    # --- thermal / torque peaks ---
    max_temp = 0.0
    max_tau = 0.0
    for mid in range(1, 13):
        tcol = f"m{mid}_temp_c"
        taucol = f"m{mid}_tau_nm"
        if tcol in df.columns:
            v = df[tcol].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                max_temp = max(max_temp, float(np.max(v)))
        if taucol in df.columns:
            v = df[taucol].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                max_tau = max(max_tau, float(np.max(np.abs(v))))

    # --- verdict ---
    estop_ever = bool(df["estop"].astype(bool).any()) if "estop" in df.columns else False
    if estop_ever:
        verdict = "ESTOP TRIGGERED DURING WINDOW"
    elif tilt_max > 25.0 or tilt_std > 6.0:
        verdict = "UNSTABLE (large tilt)"
    elif 0.5 <= mean_cadence <= 2.5 and tilt_std < 3.0 and gyro_rms < 0.8:
        verdict = f"STABLE STEPPING (~{mean_cadence:.2f} Hz cadence)"
    elif mean_cadence < 0.3 and tilt_std < 1.5 and gyro_rms < 0.3:
        verdict = "STABLE STAND"
    elif gyro_rms > 1.5 or acc_std > 5.0:
        verdict = "HIGH MOTION (bouncing / rough)"
    else:
        verdict = "MIXED -- see metrics"

    # --- print ---
    print(f"\n=== {label} ===")
    print(f"  duration: {duration:.1f}s  samples: {n}  rate: {rate:.1f} Hz  estop: {estop_ever}")
    print(f"  base tilt deg   : mean={tilt_mean:5.2f}  std={tilt_std:5.2f}  max={tilt_max:5.2f}")
    print(f"  gyro |w| rad/s  : rms={gyro_rms:5.3f}   peak={gyro_peak:5.3f}")
    print(f"  |acc| m/s^2     : mean={acc_mean:5.2f}  std={acc_std:5.2f}  peak={acc_peak:5.2f}")
    if knee_cadences:
        parts = "  ".join(f"m{mid}({MOTOR_LABELS[mid]})={c:4.2f}" for mid, c in knee_cadences.items())
        print(f"  knee cadence Hz : {parts}   mean={mean_cadence:.2f}")
    print(f"  cmd deg/s rms   : upper_hip={cmd_upper:6.1f}  knee={cmd_knee:6.1f}  ankle={cmd_ankle:6.1f}")
    if worst_track[0]:
        wid, werr = worst_track
        print(
            f"  tracking err deg: worst={MOTOR_LABELS.get(wid, f'm{wid}')} rms={werr:6.2f}  "
            f"(all-motor median rms={np.median(list(track_rms_per.values())):.2f})"
        )
    print(f"  max |tau| Nm    : {max_tau:5.2f}   max temp C: {max_temp:5.1f}")
    print(f"  verdict         : {verdict}")


def _print_scale_context(scale_path: Path) -> None:
    if not scale_path.exists():
        return
    try:
        sdf = pd.read_csv(scale_path)
    except Exception:
        return
    if sdf.empty or "upper_scale" not in sdf.columns or "ankle_scale" not in sdf.columns:
        return
    has_knee = "knee_scale" in sdf.columns
    last = sdf.iloc[-1]
    first = sdf.iloc[0]
    print("\n--- action scale history ---")
    if has_knee:
        print(
            f"  init : upper={first['upper_scale']:.2f}  knee={first['knee_scale']:.2f}  "
            f"ankle={first['ankle_scale']:.2f}"
        )
        print(
            f"  final: upper={last['upper_scale']:.2f}  knee={last['knee_scale']:.2f}  ankle={last['ankle_scale']:.2f}"
        )
    else:
        print(f"  init : upper={first['upper_scale']:.2f}  ankle={first['ankle_scale']:.2f}")
        print(f"  final: upper={last['upper_scale']:.2f}  ankle={last['ankle_scale']:.2f}")
    print(f"  changes: {len(sdf) - 1}")


def _bucket_metrics(df: pd.DataFrame, dt_s: float) -> dict:
    """Compact metric set used for the per-scale table. Same definitions as
    _report so the numbers line up."""
    if len(df) < 2:
        return {}
    pgz = df["projected_gravity_z"].to_numpy(dtype=float)
    tilt = np.degrees(np.arccos(np.clip(-pgz, -1.0, 1.0)))
    gx = df["imu_gyro_x_rad_s"].to_numpy(dtype=float)
    gy = df["imu_gyro_y_rad_s"].to_numpy(dtype=float)
    gz = df["imu_gyro_z_rad_s"].to_numpy(dtype=float)
    g_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    cadences = {}
    for mid in KNEE_MOTORS:
        col = f"m{mid}_target_pos_deg"
        if col in df.columns:
            cadences[mid] = _zero_crossings_hz(df[col].to_numpy(dtype=float), dt_s)

    worst_track = 0.0
    for mid in range(1, 13):
        pcol = f"m{mid}_pos_deg"
        tcol = f"m{mid}_target_pos_deg"
        if pcol in df.columns and tcol in df.columns:
            err = df[tcol].to_numpy(dtype=float) - df[pcol].to_numpy(dtype=float)
            err = err[np.isfinite(err)]
            if err.size:
                worst_track = max(worst_track, _rms(err))

    return {
        "tilt_mean": float(np.mean(tilt)),
        "tilt_std": float(np.std(tilt)),
        "tilt_max": float(np.max(tilt)),
        "gyro_rms": _rms(g_mag),
        "cad_L": cadences.get(4, 0.0),
        "cad_R": cadences.get(10, 0.0),
        "worst_track_deg": worst_track,
    }


def _report_by_scale(state_df: pd.DataFrame, scale_path: Path, dt_s: float, min_bucket_s: float = 1.0) -> None:
    """Slice the state log per (upper_scale, ankle_scale) segment from the
    action_scale_log and print one metrics row per segment."""
    if not scale_path.exists():
        return
    try:
        sdf = pd.read_csv(scale_path)
    except Exception:
        return
    if sdf.empty or "upper_scale" not in sdf.columns or "ankle_scale" not in sdf.columns:
        return

    t_all = state_df["time_s"].to_numpy(dtype=float)
    scale_t = sdf["time_s"].to_numpy(dtype=float)
    # Each row in the scale log sets a scale at that time; the segment runs
    # until the next row (or the end of the state log).
    bounds = list(scale_t) + [float(t_all[-1]) + 1e-6]

    has_knee = "knee_scale" in sdf.columns

    print("\n=== Per-scale breakdown ===")
    if has_knee:
        header = (
            f"  {'upper':>5} {'knee':>5} {'ankle':>5} {'dur_s':>6}  "
            f"{'tilt_mean':>9} {'tilt_std':>8} {'tilt_max':>8}  "
            f"{'gyro_rms':>8}  {'cad_L':>5} {'cad_R':>5}  {'trk_worst':>9}"
        )
    else:
        header = (
            f"  {'upper':>5} {'ankle':>5} {'dur_s':>6}  "
            f"{'tilt_mean':>9} {'tilt_std':>8} {'tilt_max':>8}  "
            f"{'gyro_rms':>8}  {'cad_L':>5} {'cad_R':>5}  {'trk_worst':>9}"
        )
    print(header)
    for i, row in sdf.iterrows():
        t_lo = float(bounds[i])
        t_hi = float(bounds[i + 1])
        seg = state_df[(t_all >= t_lo) & (t_all < t_hi)]
        if len(seg) < 2:
            continue
        dur = float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0])
        if dur < min_bucket_s:
            continue
        m = _bucket_metrics(seg, dt_s)
        if not m:
            continue
        if has_knee:
            print(
                f"  {row['upper_scale']:5.2f} {row['knee_scale']:5.2f} {row['ankle_scale']:5.2f} {dur:6.1f}  "
                f"{m['tilt_mean']:9.2f} {m['tilt_std']:8.2f} {m['tilt_max']:8.2f}  "
                f"{m['gyro_rms']:8.3f}  {m['cad_L']:5.2f} {m['cad_R']:5.2f}  {m['worst_track_deg']:9.2f}"
            )
        else:
            print(
                f"  {row['upper_scale']:5.2f} {row['ankle_scale']:5.2f} {dur:6.1f}  "
                f"{m['tilt_mean']:9.2f} {m['tilt_std']:8.2f} {m['tilt_max']:8.2f}  "
                f"{m['gyro_rms']:8.3f}  {m['cad_L']:5.2f} {m['cad_R']:5.2f}  {m['worst_track_deg']:9.2f}"
            )


def _report_by_tilt(
    state_df: pd.DataFrame,
    dt_s: float,
    edges: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0, 25.0, 35.0),
    min_bucket_s: float = 0.5,
) -> None:
    """Bucket samples by instantaneous tilt and print activity per bucket.

    Answers: when the robot is leaning more, is it also pushing harder /
    moving more / tracking worse? Time-based bucket (not chronological) --
    the same bucket can accumulate samples from anywhere in the run.
    """
    if len(state_df) < 10:
        return
    pgz = state_df["projected_gravity_z"].to_numpy(dtype=float)
    tilt = np.degrees(np.arccos(np.clip(-pgz, -1.0, 1.0)))

    gx = state_df["imu_gyro_x_rad_s"].to_numpy(dtype=float)
    gy = state_df["imu_gyro_y_rad_s"].to_numpy(dtype=float)
    gz = state_df["imu_gyro_z_rad_s"].to_numpy(dtype=float)
    g_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Aggregate per-motor abs torque across all 12 motors (max per sample).
    tau_cols = [f"m{mid}_tau_nm" for mid in range(1, 13) if f"m{mid}_tau_nm" in state_df.columns]
    if tau_cols:
        tau_abs = np.nanmax(np.abs(state_df[tau_cols].to_numpy(dtype=float)), axis=1)
    else:
        tau_abs = np.zeros(len(state_df))

    # Per-sample aggregate tracking err: max |target - pos| across motors.
    err_samples = np.zeros(len(state_df))
    for mid in range(1, 13):
        pcol = f"m{mid}_pos_deg"
        tcol = f"m{mid}_target_pos_deg"
        if pcol in state_df.columns and tcol in state_df.columns:
            e = np.abs(state_df[tcol].to_numpy(dtype=float) - state_df[pcol].to_numpy(dtype=float))
            err_samples = np.maximum(err_samples, np.where(np.isfinite(e), e, 0.0))

    # Knee target velocity (cmd speed) averaged across L+R.
    knee_cmd_vel = []
    for mid in KNEE_MOTORS:
        col = f"m{mid}_target_pos_deg"
        if col in state_df.columns:
            d = np.diff(state_df[col].to_numpy(dtype=float)) / dt_s
            knee_cmd_vel.append(np.abs(np.concatenate([[0.0], d])))
    knee_cmd_mean = np.mean(np.stack(knee_cmd_vel, axis=0), axis=0) if knee_cmd_vel else np.zeros(len(state_df))

    print("\n=== Per-tilt-bucket activity ===")
    print(
        f"  {'tilt_lo':>7} {'tilt_hi':>7} {'n':>6} {'dur_s':>6} {'%_time':>7}  "
        f"{'gyro_rms':>8} {'gyro_p95':>8}  {'tau_peak':>8} {'tau_rms':>7}  "
        f"{'knee_cmd':>8}  {'trk_peak':>8}"
    )
    total_n = len(state_df)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (tilt >= lo) & (tilt < hi)
        n = int(mask.sum())
        dur = n * dt_s
        if dur < min_bucket_s:
            continue
        print(
            f"  {lo:7.1f} {hi:7.1f} {n:6d} {dur:6.1f} {100.0 * n / total_n:7.1f}  "
            f"{_rms(g_mag[mask]):8.3f} {np.percentile(g_mag[mask], 95):8.3f}  "
            f"{np.max(tau_abs[mask]):8.2f} {_rms(tau_abs[mask]):7.2f}  "
            f"{_rms(knee_cmd_mean[mask]):8.1f}  {np.percentile(err_samples[mask], 95):8.2f}"
        )
    # catch-all: tilt >= last edge
    mask = tilt >= edges[-1]
    n = int(mask.sum())
    dur = n * dt_s
    if dur >= min_bucket_s:
        print(
            f"  {edges[-1]:7.1f} {'+inf':>7} {n:6d} {dur:6.1f} {100.0 * n / total_n:7.1f}  "
            f"{_rms(g_mag[mask]):8.3f} {np.percentile(g_mag[mask], 95):8.3f}  "
            f"{np.max(tau_abs[mask]):8.2f} {_rms(tau_abs[mask]):7.2f}  "
            f"{_rms(knee_cmd_mean[mask]):8.1f}  {np.percentile(err_samples[mask], 95):8.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument(
        "--tail-seconds",
        type=float,
        default=15.0,
        help="Also print a block restricted to the final N seconds (0 to disable).",
    )
    ap.add_argument(
        "--min-bucket-seconds",
        type=float,
        default=1.0,
        help="Per-scale rows shorter than this are skipped in the breakdown.",
    )
    args = ap.parse_args()

    state_path = args.rollout_dir / "bipedal_state_log.csv"
    if not state_path.exists():
        print(f"[summary] {state_path} not found")
        return 2

    df = pd.read_csv(state_path, low_memory=False)
    if "mode" in df.columns:
        df = df[df["mode"] == "control"].reset_index(drop=True)
    if len(df) < 2:
        print("[summary] no control-mode samples in state log")
        return 2

    dt_all = np.diff(df["time_s"].to_numpy(dtype=float))
    dt_all = dt_all[(dt_all > 0) & (dt_all < 1.0)]
    dt_s = float(np.median(dt_all)) if dt_all.size else 0.01

    _report(df, dt_s, label="Whole run (control mode)")

    if args.tail_seconds > 0:
        t = df["time_s"].to_numpy(dtype=float)
        cutoff = t[-1] - float(args.tail_seconds)
        tail = df[t >= cutoff].reset_index(drop=True)
        if len(tail) >= 2 and (t[-1] - cutoff) >= 1.0:
            _report(tail, dt_s, label=f"Last {args.tail_seconds:g}s")

    scale_path = args.rollout_dir / "action_scale_log.csv"
    _print_scale_context(scale_path)
    _report_by_scale(df, scale_path, dt_s, min_bucket_s=args.min_bucket_seconds)
    _report_by_tilt(df, dt_s, min_bucket_s=args.min_bucket_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
