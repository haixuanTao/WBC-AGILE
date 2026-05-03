# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Build a single-file HTML report from one or more sim2mujoco_sweep.py JSONs.

Each sweep contributes a section with a per-axis line chart and a colored
fall-rate table. A wandb run id may be supplied to add a leading
learning-curve section overlaying soft-eval history with optional
hard-eval checkpoint metrics.

Examples:
    python scripts/sim2mujoco_report.py \\
        --sweep /tmp/dr_sweep.json \\
        --output /tmp/report.html

    python scripts/sim2mujoco_report.py \\
        --sweep /tmp/dr_sweep.json \\
        --sweep /tmp/init_sweep.json \\
        --wandb-run 1ms-ai/Velocity-LeRobot-NoArms/3w14ovgr \\
        --hard-history /tmp/hardeval_multi.json \\
        --output /tmp/report.html
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
from pathlib import Path

import plotly.graph_objects as go

SCHED_COLORS = {"stand": "#1f77b4", "fwd": "#ff7f0e", "strafe": "#2ca02c", "yaw": "#d62728"}

# Categories must match scripts/sim2mujoco_sweep.py CATEGORY_ORDER / CATEGORY_LABELS.
CATEGORY_ORDER = ("dr", "actuator", "init_state", "disturbance", "other")
CATEGORY_LABELS = {
    "dr": "Domain randomization",
    "actuator": "Actuator (gains & limits)",
    "init_state": "Initial state",
    "disturbance": "Disturbance",
    "other": "Other",
}


def _fr_color(rate: float) -> str:
    """RdYlGn_r-like gradient: green at 0, yellow at 0.5, red at 1."""
    rate = max(0.0, min(1.0, rate))
    if rate < 0.5:
        r, g, b = int(120 + 135 * rate * 2), 200, 120
    else:
        r, g, b = 220, int(200 - 120 * (rate - 0.5) * 2), 90
    return f"rgb({r},{g},{b})"


def _axis_chart(axis_name: str, axis_block: dict, *, max_survival: float) -> str:
    """Single plotly figure -> HTML div."""
    res = axis_block["results"]
    xs = sorted(float(k) for k in res)
    rbx = {float(k): v for k, v in res.items()}
    fig = go.Figure()

    train = axis_block.get("training_range")
    if train and train[1] > train[0] and (train[0] > xs[0] or train[1] < xs[-1]):
        fig.add_vrect(
            x0=train[0],
            x1=train[1],
            fillcolor="#2ca02c",
            opacity=0.12,
            line_width=0,
            annotation_text="trained",
            annotation_position="top left",
        )

    for sn, color in SCHED_COLORS.items():
        ys = [rbx[x]["per_schedule"].get(sn, {}).get("mean_survival_s") for x in xs]
        if any(y is not None for y in ys):
            fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=sn, line={"color": color, "width": 1.5})
    fig.add_scatter(
        x=xs,
        y=[rbx[x]["mean_survival_s"] for x in xs],
        mode="lines+markers",
        name="avg",
        line={"color": "black", "width": 2.5},
        marker={"symbol": "square", "size": 8},
    )

    label = axis_block.get("description", axis_name)
    xlabel = axis_name if label == axis_name else f"{axis_name} — {label}"
    fig.update_layout(
        title={"text": f"<b>{html.escape(axis_name)}</b>", "x": 0.02, "xanchor": "left", "font": {"size": 14}},
        xaxis_title=xlabel,
        yaxis_title="survival (s)",
        yaxis_range=[0, max_survival + 0.5],
        height=320,
        margin={"l": 60, "r": 30, "t": 40, "b": 50},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
        plot_bgcolor="#fafafa",
    )
    fig.update_xaxes(gridcolor="#e8e8e8")
    fig.update_yaxes(gridcolor="#e8e8e8")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _grouped_axes(sweep: dict) -> list[tuple[str, list[tuple[str, dict]]]]:
    """Return [(category, [(axis_name, block), ...]), ...] in CATEGORY_ORDER."""
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for name, block in sweep["axes"].items():
        by_cat.setdefault(block.get("category", "other"), []).append((name, block))
    return [(c, by_cat[c]) for c in CATEGORY_ORDER if c in by_cat]


def _sweep_table(sweep: dict) -> str:
    """Render the colored fall-rate table as an HTML <table>."""
    rows_html = []
    for axis_name, block in sweep["axes"].items():
        first = True
        for k_str in sorted(block["results"], key=float):
            r = block["results"][k_str]
            ps = r.get("per_schedule", {})
            cells = [
                f'<td class="ax">{html.escape(axis_name)}</td>',
                f"<td>{float(k_str):g}</td>",
            ]
            for sn in ("stand", "fwd", "strafe", "yaw"):
                fr = ps.get(sn, {}).get("fall_rate", 0.0)
                cells.append(f'<td style="background:{_fr_color(fr)}">{fr:.2f}</td>')
            cells.append(f"<td>{r['mean_survival_s']:.1f}</td>")
            cls = ' class="group"' if first else ""
            rows_html.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
            first = False

    return (
        '<table class="fr">'
        "<thead><tr>"
        "<th>axis</th><th>value</th><th>stand</th><th>fwd</th><th>strafe</th><th>yaw</th><th>surv (s)</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )


def _learning_curve_section(wandb_run: str, hard_history: list[dict] | None) -> str:
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run)
    keys = ["sim_to_sim/iter", "sim_to_sim/mean_survival_time_s", "sim_to_sim/fall_rate"]
    soft_hist = sorted(run.history(keys=keys, pandas=False), key=lambda r: r.get("sim_to_sim/iter", 0))
    soft = [
        (r["sim_to_sim/iter"], r["sim_to_sim/mean_survival_time_s"], r["sim_to_sim/fall_rate"])
        for r in soft_hist
        if r.get("sim_to_sim/iter") is not None
    ]
    sx = [r[0] for r in soft]

    figs = []
    for ylabel, y_idx, hard_key, yrange in [
        ("survival (s)", 1, "mean_survival_time_s", None),
        ("fall rate", 2, "fall_rate", [0, 1.05]),
    ]:
        fig = go.Figure()
        fig.add_scatter(
            x=sx,
            y=[r[y_idx] for r in soft],
            mode="lines+markers",
            name="soft eval",
            line={"color": "#2ca02c", "width": 1.8},
        )
        if hard_history:
            fig.add_scatter(
                x=[r["step"] for r in hard_history],
                y=[r[hard_key] for r in hard_history],
                mode="lines+markers",
                name="hard eval",
                line={"color": "#d62728", "width": 2.5},
                marker={"symbol": "square", "size": 9},
            )
        fig.update_layout(
            title={"text": f"<b>{ylabel}</b>", "x": 0.02, "xanchor": "left"},
            xaxis_title="checkpoint iter",
            yaxis_title=ylabel,
            height=320,
            margin={"l": 60, "r": 30, "t": 40, "b": 50},
            legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
            plot_bgcolor="#fafafa",
        )
        if yrange:
            fig.update_yaxes(range=yrange)
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))
    return "\n".join(figs)


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1080px; margin: 2em auto; padding: 0 2em; color: #222; }
h1 { font-size: 1.8em; margin-bottom: 0.1em; }
.meta { color: #777; margin-bottom: 2em; font-size: 0.95em; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; margin-top: 3em; font-size: 1.3em; }
h3 { color: #333; margin-top: 2em; margin-bottom: 0.4em; font-size: 1.05em; }
ol.toc { line-height: 1.8; }
ol.toc a { color: #225; text-decoration: none; }
ol.toc a:hover { text-decoration: underline; }
table.fr { border-collapse: collapse; width: 100%; margin: 1.5em 0; font-size: 0.88em; }
table.fr th { background: #333; color: white; padding: 0.5em 0.6em; text-align: center; font-weight: 600; }
table.fr td { padding: 0.35em 0.6em; text-align: center; border-bottom: 1px solid #eee; }
table.fr td.ax { text-align: left; color: #555; }
table.fr tr.group td { font-weight: 700; }
table.fr tr.group td.ax { color: #000; }
.chart-stack > div { margin: 0.5em 0; }
"""


def build_html(
    output: Path,
    sweeps: list[Path],
    *,
    wandb_run: str | None = None,
    hard_history_path: Path | None = None,
    title: str = "Sim-to-Sim Robustness Report",
) -> None:
    sweep_dicts = [(p, json.loads(p.read_text())) for p in sweeps]
    hard_history = (
        sorted(json.loads(hard_history_path.read_text()), key=lambda r: r["step"]) if hard_history_path else None
    )

    sections: list[str] = []
    toc: list[str] = []

    if wandb_run:
        anchor = "learning-curve"
        toc.append(f'<li><a href="#{anchor}">Learning curve{" (soft vs hard)" if hard_history else ""}</a></li>')
        sections.append(
            f'<h2 id="{anchor}">Learning curve</h2><div class="chart-stack">'
            + _learning_curve_section(wandb_run, hard_history)
            + "</div>"
        )

    for i, (path, sweep) in enumerate(sweep_dicts, start=1):
        anchor = f"sweep-{i}"
        ckpt = Path(sweep.get("checkpoint", "?")).name
        n_trials = sweep.get("n_trials", "?")
        duration = sweep.get("duration_s", "?")
        n_axes = len(sweep.get("axes", {}))
        max_survival = float(sweep.get("duration_s", 12.0))

        groups = _grouped_axes(sweep)
        cat_summary = ", ".join(f"{CATEGORY_LABELS[c]} ({len(g)})" for c, g in groups)
        toc.append(
            f'<li><a href="#{anchor}">{html.escape(path.stem)} '
            f"<small>[{n_axes} axes — {html.escape(cat_summary)}]</small></a></li>"
        )

        body_parts = [
            f'<h2 id="{anchor}">{html.escape(path.stem)}'
            f' <small style="font-weight:400;color:#777;font-size:0.7em">'
            f"{html.escape(ckpt)} · N={n_trials} × {duration}s</small></h2>"
        ]
        for cat, axes_in_cat in groups:
            body_parts.append(f'<h3 id="{anchor}-{cat}">{CATEGORY_LABELS[cat]}</h3>')
            charts = "".join(_axis_chart(name, block, max_survival=max_survival) for name, block in axes_in_cat)
            body_parts.append(f'<div class="chart-stack">{charts}</div>')
        body_parts.append(_sweep_table(sweep))
        sections.append("".join(body_parts))

    meta = datetime.date.today().isoformat()
    if wandb_run:
        meta = f"{html.escape(wandb_run)} · {meta}"

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='meta'>{meta}</div>"
        "<h2>Contents</h2><ol class='toc'>" + "".join(toc) + "</ol>" + "".join(sections) + "</body></html>"
    )
    output.write_text(doc, encoding="utf-8")
    print(f"saved: {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, action="append", required=True, help="Sweep JSON (repeat for multiple).")
    p.add_argument("--output", type=Path, required=True, help="Output .html path.")
    p.add_argument("--wandb-run", type=str, default=None, help="entity/project/run_id for the learning-curve overlay.")
    p.add_argument(
        "--hard-history",
        type=Path,
        default=None,
        help="Optional JSON: list of {step, mean_survival_time_s, fall_rate}.",
    )
    p.add_argument("--title", type=str, default="Sim-to-Sim Robustness Report")
    args = p.parse_args()

    build_html(
        output=args.output,
        sweeps=args.sweep,
        wandb_run=args.wandb_run,
        hard_history_path=args.hard_history,
        title=args.title,
    )


if __name__ == "__main__":
    main()
