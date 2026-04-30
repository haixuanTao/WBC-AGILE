"""Replay Pi-logged observations through the ONNX policy and diff the logged actions.

v16_shadow.csv stores per-step `[time, step, observation, action_pre_scale]`
produced by the RLAgent running on the Pi. If the pulled `policy.onnx` is
literally the one that produced the CSV, we expect exact (or near-exact)
numerical agreement when we feed the same obs back through it. Large diffs
flag either a mismatched onnx file, a CPU/GPU rounding gap, or a bug in how
the Pi serializes obs vs how our local replay deserializes them.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import numpy as np


def _parse_list(cell: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(cell), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-csv", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-4, help="max|Δ| threshold for a step to count as a match")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    out_name = sess.get_outputs()[0].name
    print(f"[replay] onnx in={in_name}{in_shape} out={out_name}")

    total_rows = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    n_elems = 0
    bad: list[tuple[int, float]] = []

    with args.shadow_csv.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.max_rows and i >= args.max_rows:
                break
            obs = _parse_list(row["observation"])
            logged = _parse_list(row["action_pre_scale"])
            out = sess.run([out_name], {in_name: obs[None, :]})[0].reshape(-1)
            if out.shape != logged.shape:
                print(f"[replay][ERR] shape mismatch at row {i}: onnx_out={out.shape} logged={logged.shape}")
                return 2
            diff = np.abs(out - logged)
            m = float(diff.max())
            max_abs_diff = max(max_abs_diff, m)
            sum_abs_diff += float(diff.sum())
            n_elems += diff.size
            total_rows = i + 1
            if m > args.tol:
                bad.append((i, m))

    mean_abs_diff = sum_abs_diff / max(n_elems, 1)
    print(f"[replay] rows={total_rows}  max|Δ|={max_abs_diff:.6g}  mean|Δ|={mean_abs_diff:.6g}  tol={args.tol:.6g}")
    if bad:
        print(f"[replay] {len(bad)} rows exceed tol. first 10:")
        for idx, m in bad[:10]:
            print(f"    row {idx}: max|Δ|={m:.6g}")
        return 1
    print("[replay] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
