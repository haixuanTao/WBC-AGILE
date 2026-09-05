# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[newton] Collide with the generated terrain as a heightfield, not a trimesh.

The foot collision spheres catch on the internal triangle edges of the terrain
trimesh: an edge contact yields a horizontal normal, and friction on a horizontal
normal acts vertically and drags the foot down (measured: 95% of the contacts
reporting a downward force had a horizontal normal, 84% of that force was
friction). A heightfield has one well-defined up-normal per cell and no
internal edges, and MuJoCo-Warp collides every robot geom type against it.

How: the trimesh is captured when :class:`TerrainImporter` imports it; the USD
terrain prim is excluded from Newton's physics import (its exact path goes to
``excludePaths``, so the USD mesh stays for the height-scan ray caster); after
the global ``add_usd`` pass the mesh is rasterized on the GPU (downward Warp ray
query per cell) and added with ``ModelBuilder.add_shape_heightfield``.

The raster covers the sub-terrain grid plus ``AGILE_NEWTON_HEIGHTFIELD_MARGIN``
metres of the flat border (default 12; the generator's border is 100 m, more
than a 15 s episode can cross) at ``AGILE_NEWTON_HEIGHTFIELD_RES`` metres per
cell (default 0.1 = the generator's horizontal_scale).

Enable with ``AGILE_NEWTON_HEIGHTFIELD=1``.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

_SENTINEL = "_agile_newton_heightfield_patch_applied"
CAPTURED: dict = {}   # name -> {"mesh": trimesh, "prim_path": str, "border": float, "mu": float}


@wp.kernel(enable_backward=False)
def _raster_kernel(
    mesh_id: wp.uint64,
    x0: float, y0: float, dx: float, dy: float,
    ncol: int, z_top: float, max_t: float, fallback: float,
    heights: wp.array2d(dtype=wp.float32),
):
    r, c = wp.tid()
    start = wp.vec3(x0 + float(c) * dx, y0 + float(r) * dy, z_top)
    q = wp.mesh_query_ray(mesh_id, start, wp.vec3(0.0, 0.0, -1.0), max_t)
    if q.result:
        heights[r, c] = z_top - q.t
    else:
        heights[r, c] = fallback


def rasterize(mesh, x_min, x_max, y_min, y_max, res, device):
    """Heights on a regular grid, rows along y, columns along x."""
    ncol = int(np.ceil((x_max - x_min) / res)) + 1
    nrow = int(np.ceil((y_max - y_min) / res)) + 1
    v = np.asarray(mesh.vertices, dtype=np.float32); f = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    wmesh = wp.Mesh(points=wp.array(v, dtype=wp.vec3, device=device), indices=wp.array(f, dtype=wp.int32, device=device))
    z_top = float(v[:, 2].max()) + 1.0
    max_t = float(v[:, 2].max() - v[:, 2].min()) + 2.0
    heights = wp.zeros((nrow, ncol), dtype=wp.float32, device=device)
    wp.launch(_raster_kernel, dim=(nrow, ncol),
              inputs=[wmesh.id, float(x_min), float(y_min), float(res), float(res), ncol, z_top, max_t, float(v[:, 2].min())],
              outputs=[heights], device=device)
    h = heights.numpy()
    return h, nrow, ncol


_GEOM_POS_SENTINEL = "_agile_hfield_geom_pos_fix_applied"


def apply_newton_hfield_geom_pos_fix() -> bool:
    """Newton 1.5.x: per-world ``mjw_model.geom_pos`` of a heightfield drops ``min_z``.

    ``SolverMuJoCo`` builds the MuJoCo spec with the heightfield geom at
    ``xform.p.z + min_z`` (so the surface spans ``[min_z, max_z]``), but
    ``_update_geom_properties`` -- run on every ``SHAPE_PROPERTIES`` notification,
    including model init -- rewrites the per-world ``geom_pos`` from the Newton shape
    transform alone. Measured on Isaac Lab develop with AGILE's terrain
    (``min_z = -0.14``): the MuJoCo model had the geom at z=-0.14, the Warp model at
    z=0, and every terrain contact sat 0.14 m above the mesh (median +0.1398, p90
    0.1412). Every reset then starts 14 cm inside the ground.

    This wraps ``_update_geom_properties`` and copies the MuJoCo model's heightfield
    geom positions back into all worlds afterwards. No-op where the two already agree
    (Newton 1.2 measured -0.4 mm without it).
    """
    try:
        import mujoco
        from newton.solvers import SolverMuJoCo
    except Exception:
        return False
    if getattr(SolverMuJoCo, _GEOM_POS_SENTINEL, False):
        return False
    original = SolverMuJoCo._update_geom_properties

    def _update_geom_properties_with_hfield_fix(self):
        original(self)
        try:
            mj, mjw = self.mj_model, self.mjw_model
            hg = [g for g in range(mj.ngeom) if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
            if not hg:
                return
            gp = wp.to_torch(mjw.geom_pos)  # (nworld, ngeom) vec3 or (ngeom,) vec3
            fixed = 0
            for g in hg:
                want = mj.geom_pos[g]
                cur = gp[..., g, :]
                if (cur - cur.new_tensor(want)).abs().max().item() > 1e-6:
                    cur[...] = cur.new_tensor(want)
                    fixed += 1
            if fixed and not getattr(self, "_agile_hfield_geom_pos_reported", False):
                self._agile_hfield_geom_pos_reported = True
                print(f"[newton] heightfield: restored min_z offset on {fixed} hfield geom(s) in all worlds "
                      f"(z={mj.geom_pos[hg[0]][2]:+.4f}; Newton's geom sync had dropped it)", flush=True)
        except Exception as exc:  # never break the sim over the fix
            print(f"[newton] heightfield geom_pos fix skipped: {exc}", flush=True)

    SolverMuJoCo._update_geom_properties = _update_geom_properties_with_hfield_fix
    setattr(SolverMuJoCo, _GEOM_POS_SENTINEL, True)
    return True


def apply_newton_heightfield_terrain_patch() -> bool:
    if os.environ.get("AGILE_NEWTON_HEIGHTFIELD", "0") != "1":
        return False
    apply_newton_hfield_geom_pos_fix()
    try:
        from isaaclab.terrains.terrain_importer import TerrainImporter
        from newton import Heightfield, ModelBuilder
    except Exception:
        return False
    if getattr(ModelBuilder, _SENTINEL, False):
        return False
    native = False
    try:  # Isaac Lab develop converts tagged terrains itself (SubTerrainBaseCfg.convert_to_heightfield)
        from isaaclab_newton.physics import NewtonManager
        native = hasattr(NewtonManager, "_inject_terrain_heightfields")
    except Exception:
        pass
    # AGILE_NEWTON_HEIGHTFIELD_NATIVE=1 uses Isaac Lab develop's own conversion instead of this
    # wrapper. Off by default: the native path is one heightfield (no tiling, see
    # AGILE_NEWTON_HEIGHTFIELD_TILE) and, on Newton 1.5.x, needs the geom_pos fix below;
    # measured on develop, wrapper + 8 m tiles: 0.0% of terrain contacts deeper than 5 cm,
    # native: 0.2%, with the deep ones at -1.5..-2.8 m.
    native = native and os.environ.get("AGILE_NEWTON_HEIGHTFIELD_NATIVE", "0") != "0"
    if native:
        print("[newton] heightfield: Isaac Lab has native terrain heightfields -> using convert_to_heightfield; "
              "wrapper only keeps the terrain mesh for diagnostics")

    # 1) capture the generated trimesh when it is imported
    original_import_mesh = TerrainImporter.import_mesh

    def import_mesh_and_capture(self, name, mesh):
        original_import_mesh(self, name, mesh)
        gen = getattr(self.cfg, "terrain_generator", None)
        mat = getattr(self.cfg, "physics_material", None)
        CAPTURED[name] = {
            "mesh": mesh,
            "prim_path": self.cfg.prim_path + f"/{name}",
            "border": float(getattr(gen, "border_width", 0.0)) if gen is not None else 0.0,
            "mu": float(getattr(mat, "static_friction", 1.0)) if mat is not None else 1.0,
        }
        print(f"[newton] heightfield: captured terrain mesh '{name}' ({len(mesh.vertices)} verts, {len(mesh.faces)} faces) at {CAPTURED[name]['prim_path']}", flush=True)

    TerrainImporter.import_mesh = import_mesh_and_capture

    if native:
        setattr(ModelBuilder, _SENTINEL, True)
        return False
    # 2) swap the collider at model-build time
    original_add_usd = ModelBuilder.add_usd

    def add_usd_with_heightfield(self, source, *args, **kwargs):
        # Isaac Lab 3.0.0b2 imports the whole stage once (root "/"); develop imports it one
        # top-level prim at a time ("/World/ground", "/World/envs/env_0", ...). Act on any call
        # whose root contains a captured terrain prim, once per terrain.
        rp = str(kwargs.get("root_path", "/") or "/").rstrip("/") or "/"
        hits = {name: info for name, info in CAPTURED.items()
                if not info.get("added") and (rp == "/" or info["prim_path"] == rp or info["prim_path"].startswith(rp + "/"))}
        if not hits:
            return original_add_usd(self, source, *args, **kwargs)
        ignore = list(kwargs.get("ignore_paths") or [])
        for info in hits.values():
            ignore.append(info["prim_path"])           # exact path -> excludePaths of the physics parser
            info["added"] = True
        kwargs["ignore_paths"] = ignore
        out = original_add_usd(self, source, *args, **kwargs)
        res = float(os.environ.get("AGILE_NEWTON_HEIGHTFIELD_RES", "0.1"))
        margin = float(os.environ.get("AGILE_NEWTON_HEIGHTFIELD_MARGIN", "12.0"))
        for name, info in hits.items():
            mesh = info["mesh"]; b = mesh.bounds  # (2, 3)
            shrink = max(info["border"] - margin, 0.0)
            x_min, x_max = b[0, 0] + shrink, b[1, 0] - shrink
            y_min, y_max = b[0, 1] + shrink, b[1, 1] - shrink
            h, nrow, ncol = rasterize(mesh, x_min, x_max, y_min, y_max, res, wp.get_preferred_device())
            cfg = ModelBuilder.ShapeConfig(mu=info["mu"], restitution=0.0, density=0.0)
            # AGILE_NEWTON_HEIGHTFIELD_TILE (m): split the raster into tiles, each its own
            # heightfield shape centred on itself. MuJoCo-Warp's heightfield collision
            # reports spurious deep contacts whose depth grows with the distance from the
            # heightfield origin (measured: -0.1 m at the centre column, -1.5..-2.8 m at
            # +-32 m), so keeping every point within a few metres of its shape origin
            # sidesteps it. Tiles overlap by one sample row/column, so the surface is
            # identical at the seams. 0 (default) = one heightfield.
            tile_m = float(os.environ.get("AGILE_NEWTON_HEIGHTFIELD_TILE", "0"))
            tile_n = int(round(tile_m / res)) if tile_m > 0 else 0
            sids = []
            r_edges = list(range(0, nrow - 1, tile_n)) + [nrow - 1] if tile_n else [0, nrow - 1]
            c_edges = list(range(0, ncol - 1, tile_n)) + [ncol - 1] if tile_n else [0, ncol - 1]
            for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
                for c0, c1 in zip(c_edges[:-1], c_edges[1:]):
                    ht = np.ascontiguousarray(h[r0:r1 + 1, c0:c1 + 1])
                    nr, nc = ht.shape
                    if nr < 2 or nc < 2:
                        continue
                    hx = (nc - 1) * res / 2.0; hy = (nr - 1) * res / 2.0
                    cx = x_min + c0 * res + hx; cy = y_min + r0 * res + hy
                    zmin, zmax = float(ht.min()), float(ht.max())
                    if zmax - zmin < 1e-4:  # flat tile: keep a non-degenerate z range
                        zmax = zmin + 1e-4
                    hf = Heightfield(data=ht, nrow=nr, ncol=nc, hx=hx, hy=hy, min_z=zmin, max_z=zmax)
                    sids.append(self.add_shape_heightfield(xform=wp.transform(wp.vec3(cx, cy, 0.0), wp.quat_identity()), heightfield=hf, cfg=cfg,
                                                           label=f"{info['prim_path']}/heightfield_{len(sids)}"))
            info["raster"] = dict(h=h, x_min=x_min, y_min=y_min, res=res, nrow=nrow, ncol=ncol)
            print(f"[newton] heightfield: terrain '{name}' -> {nrow}x{ncol} cells @ {res} m over x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}], "
                  f"z[{h.min():.3f},{h.max():.3f}], {len(sids)} shape(s)" + (f" (tiles of {tile_m:g} m)" if tile_n else "") + "; trimesh collider excluded", flush=True)
        return out

    ModelBuilder.add_usd = add_usd_with_heightfield
    setattr(ModelBuilder, _SENTINEL, True)
    return True


apply_newton_heightfield_terrain_patch()
