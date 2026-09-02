# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the first non-finite quantity, with one step of history.

Enable with ``AGILE_NAN_WATCHDOG=1``. Hooks rsl-rl's entry point,
``RslRlVecEnvWrapper.step`` -- its output is exactly what rsl_rl's ``check_nan``
inspects -- and after every env step checks the articulation state, the contact
sensor and the returned observations.

Two kinds of report:

* a non-finite *state* quantity (joint or root state, torque, contact force):
  which one went first, which env, and that env's state on the previous step;
* a non-finite *observation* on finite state: which observation term produced
  it, in which env, plus a sweep of every tensor on ``robot.data`` and the
  sensors for the source.

[newton] The earlier version wrapped ``ManagerBasedRLEnv.step``; that method is
replaced wholesale by ``manager_based_rl_env_patch`` after this module runs, so
the wrapper was discarded and nothing was ever reported.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_SENTINEL = "_agile_nan_watchdog_applied"

_FIELDS = (
    "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
    "joint_pos", "joint_vel", "joint_acc", "applied_torque",
)


def _as_torch(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if hasattr(value, "torch"):
        value = value.torch
    return value if isinstance(value, torch.Tensor) else None


def _fmt_absmax(v: torch.Tensor) -> str:
    fin = torch.isfinite(v)
    return f"{v[fin].abs().max().item():.4g}" if fin.any() else "nan"


def apply_nan_watchdog_patch() -> bool:
    if os.environ.get("AGILE_NAN_WATCHDOG", "0") != "1":
        return False
    from isaaclab.envs import ManagerBasedRLEnv

    if getattr(ManagerBasedRLEnv, _SENTINEL, False):
        return False

    state = {"n": 0, "prev": None, "hist": []}

    def _report_state(env, first_bad, snap):
        bad = ~torch.isfinite(snap[first_bad])
        envs = bad.any(dim=tuple(range(1, bad.ndim))) if bad.ndim > 1 else bad
        e = int(envs.nonzero()[0])
        print("\n" + "=" * 78, flush=True)
        print(f"[watchdog] FIRST NON-FINITE at env-step {state['n']}", flush=True)
        print(f"[watchdog] leading quantity: {first_bad}", flush=True)
        print(f"[watchdog] envs affected: {int(envs.sum())} / {envs.numel()}   first env: {e}", flush=True)
        print(f"[watchdog] episode_length_buf[{e}] = {int(env.episode_length_buf[e])} (0 means it reset this step)", flush=True)
        print(f"\n[watchdog] all quantities THIS step (env {e}):", flush=True)
        for k, v in snap.items():
            fin = "finite" if torch.isfinite(v[e]).all() else "NON-FINITE"
            print(f"    {k:24s} {fin:11s} absmax={_fmt_absmax(v[e])}", flush=True)
        prev = state["prev"]
        if prev is not None:
            print(f"\n[watchdog] PREVIOUS step (env {e}) -- the last good state:", flush=True)
            for k, v in prev.items():
                if e < v.shape[0]:
                    print(f"    {k:24s} absmax={_fmt_absmax(v[e])}", flush=True)
        # [newton] which joint, and what MuJoCo actually applied to it
        try:
            robot = env.scene["robot"]
            names = list(robot.joint_names)
            hist = state["hist"]
            if hist:
                print(f"\n[watchdog] max|joint_vel| of env {e} over the last {len(hist)} steps: "
                      + " -> ".join(f"{h[e]:.3g}" for h in hist), flush=True)
            pv = prev.get("joint_vel") if prev is not None else None
            if pv is not None:
                worst = torch.argsort(pv[e].abs(), descending=True)[:3].tolist()
                print(f"[watchdog] fastest joints on the last good step: "
                      + ", ".join(f"{names[j]}={pv[e][j].item():.1f} rad/s (pos {prev['joint_pos'][e][j].item():.2f})" for j in worst), flush=True)
                from isaaclab_newton.physics import NewtonManager
                s = NewtonManager._solver; mjw = s.mjw_model; mjd = s.mjw_data
                jm = s.mjc_jnt_to_newton_dof.numpy()
                labels = [l.rsplit("/", 1)[-1] for l in NewtonManager.get_model().joint_label]
                qd_start = NewtonManager.get_model().joint_qd_start.numpy()
                arm = mjw.dof_armature.numpy(); rng = mjw.jnt_range.numpy(); lim = mjw.jnt_limited.numpy()
                qfa = mjd.qfrc_actuator.numpy(); qfc = mjd.qfrc_constraint.numpy(); qfp = mjd.qfrc_passive.numpy()
                qfs = mjd.qfrc_smooth.numpy() if hasattr(mjd, "qfrc_smooth") else None
                afr = mjw.jnt_actfrcrange.numpy()
                for j in worst:
                    jn = names[j]
                    # newton joint index with this name in world e -> mjc joint index
                    cand = [k for k, l in enumerate(labels) if l == jn]
                    nj = cand[e] if e < len(cand) else (cand[0] if cand else -1)
                    if nj < 0: continue
                    d = int(qd_start[nj])
                    mj = [k for k in range(jm.shape[1]) if int(jm[e, k]) == d]
                    mj = mj[0] if mj else -1
                    ld = d - int(qd_start[[k for k,l in enumerate(labels)][0]]) if False else None
                    print(f"[watchdog]   {jn}: mjc_jnt={mj} newton_dof={d} armature={arm[e, mj] if mj>=0 and arm.ndim==2 else 'n/a'} "
                          f"limited={bool(lim[mj]) if mj>=0 else 'n/a'} range={rng[e, mj].tolist() if mj>=0 else 'n/a'} actfrcrange={afr[e, mj].tolist() if mj>=0 else 'n/a'}", flush=True)
                    # mujoco dof index for this joint = position of d within world e's dof list
                    # MuJoCo dof index of this joint: authoritative via jnt_dofadr (1 dof per hinge)
                    dofadr = mjw.jnt_dofadr.numpy()
                    md = int(dofadr[mj]) if mj >= 0 else -1
                    if md >= 0:
                        print(f"[watchdog]     qfrc: actuator={qfa[e, md]:.1f} constraint={qfc[e, md]:.1f} passive={qfp[e, md]:.1f}"
                              + (f" smooth={qfs[e, md]:.1f}" if qfs is not None else ""), flush=True)
        except Exception as exc:
            print(f"[watchdog]   (joint breakdown failed: {exc})", flush=True)
        print("=" * 78 + "\n", flush=True)

    def _report_obs(env, robot, obs_bad):
        om = env.observation_manager
        print("\n" + "=" * 78, flush=True)
        print(f"[watchdog] NON-FINITE OBSERVATION at env-step {state['n']} in groups {obs_bad}; robot state is finite", flush=True)
        for g in obs_bad:
            names = om._group_obs_term_names[g]
            cfgs = om._group_obs_term_cfgs[g]
            for name, cfg in zip(names, cfgs):
                try:
                    raw = _as_torch(cfg.func(env, **cfg.params))
                    if raw is None:
                        continue
                    fin = torch.isfinite(raw)
                    if not fin.all():
                        envs = (~fin).reshape(raw.shape[0], -1).any(dim=1)
                        e = int(envs.nonzero()[0])
                        print(f"[watchdog]   term {g}/{name}: NON-FINITE in {int(envs.sum())} envs (first env {e}, "
                              f"episode_length_buf={int(env.episode_length_buf[e])}); finite absmax={_fmt_absmax(raw)}", flush=True)
                except Exception as exc:
                    print(f"[watchdog]   term {g}/{name}: re-eval failed: {exc}", flush=True)
        for k in dir(robot.data):
            if k.startswith("_"):
                continue
            try:
                v = _as_torch(getattr(robot.data, k))
            except Exception:
                continue
            if v is not None and v.is_floating_point() and not torch.isfinite(v).all():
                print(f"[watchdog]   robot.data.{k} is NON-FINITE", flush=True)
        for sname, sensor in getattr(env.scene, "sensors", {}).items():
            for k in ("net_forces_w", "net_forces_w_history", "ray_hits_w", "pos_w", "quat_w"):
                v = _as_torch(getattr(sensor.data, k, None))
                if v is not None and v.is_floating_point() and not torch.isfinite(v).all():
                    print(f"[watchdog]   sensor {sname}.data.{k} is NON-FINITE", flush=True)
        print("=" * 78 + "\n", flush=True)

    def _inspect(env, out):
        state["n"] += 1
        robot = env.scene["robot"]
        snap, first_bad = {}, None
        for name in _FIELDS:
            t = _as_torch(getattr(robot.data, name, None))
            if t is None:
                continue
            snap[name] = t.detach().clone()
            if first_bad is None and not torch.isfinite(t).all():
                first_bad = name
        sensor = env.scene.sensors.get("contact_forces") if hasattr(env.scene, "sensors") else None
        if sensor is not None:
            cf = _as_torch(getattr(sensor.data, "net_forces_w", None))
            if cf is not None:
                snap["contact_net_forces_w"] = cf.detach().clone()
                if first_bad is None and not torch.isfinite(cf).all():
                    first_bad = "contact_net_forces_w"
        jv = snap.get("joint_vel")
        if jv is not None:
            state["hist"].append(jv.abs().amax(dim=1).detach().clone())
            if len(state["hist"]) > 10:
                state["hist"].pop(0)
        if first_bad is not None:
            _report_state(env, first_bad, snap)
        else:
            obs = out[0] if isinstance(out, tuple) else None
            obs_bad = []
            try:
                for g in (obs.keys() if hasattr(obs, "keys") else []):
                    v = _as_torch(obs[g])
                    if v is not None and not torch.isfinite(v).all():
                        obs_bad.append(g)
            except Exception:
                pass
            if obs_bad:
                _report_obs(env, robot, obs_bad)
        state["prev"] = snap

    try:
        from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper

        _orig_vec_step = RslRlVecEnvWrapper.step

        def vec_step_with_watchdog(self, actions):
            out = _orig_vec_step(self, actions)
            _inspect(self.unwrapped, out)
            return out

        RslRlVecEnvWrapper.step = vec_step_with_watchdog
        print("[watchdog] armed on RslRlVecEnvWrapper.step", flush=True)
    except Exception as exc:
        original_step = ManagerBasedRLEnv.step

        def step_with_watchdog(self, action):
            out = original_step(self, action)
            _inspect(self, out)
            return out

        ManagerBasedRLEnv.step = step_with_watchdog
        print(f"[watchdog] could not hook RslRlVecEnvWrapper.step ({exc}); wrapped ManagerBasedRLEnv.step", flush=True)
    setattr(ManagerBasedRLEnv, _SENTINEL, True)
    return True
