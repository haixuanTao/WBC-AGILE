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


def apply_newton_heightfield_terrain_patch() -> bool:
    if os.environ.get("AGILE_NEWTON_HEIGHTFIELD", "0") != "1":
        return False
    try:
        from isaaclab.terrains.terrain_importer import TerrainImporter
        from newton import Heightfield, ModelBuilder
    except Exception:
        return False
    if getattr(ModelBuilder, _SENTINEL, False):
        return False

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

    # 2) swap the collider at model-build time
    original_add_usd = ModelBuilder.add_usd

    def add_usd_with_heightfield(self, source, *args, **kwargs):
        if not CAPTURED or kwargs.get("root_path", "/") != "/":
            return original_add_usd(self, source, *args, **kwargs)
        ignore = list(kwargs.get("ignore_paths") or [])
        for info in CAPTURED.values():
            ignore.append(info["prim_path"])           # exact path -> excludePaths of the physics parser
        kwargs["ignore_paths"] = ignore
        out = original_add_usd(self, source, *args, **kwargs)
        res = float(os.environ.get("AGILE_NEWTON_HEIGHTFIELD_RES", "0.1"))
        margin = float(os.environ.get("AGILE_NEWTON_HEIGHTFIELD_MARGIN", "12.0"))
        for name, info in CAPTURED.items():
            mesh = info["mesh"]; b = mesh.bounds  # (2, 3)
            shrink = max(info["border"] - margin, 0.0)
            x_min, x_max = b[0, 0] + shrink, b[1, 0] - shrink
            y_min, y_max = b[0, 1] + shrink, b[1, 1] - shrink
            h, nrow, ncol = rasterize(mesh, x_min, x_max, y_min, y_max, res, wp.get_preferred_device())
            hx = (ncol - 1) * res / 2.0; hy = (nrow - 1) * res / 2.0
            cx = x_min + hx; cy = y_min + hy
            hf = Heightfield(data=h, nrow=nrow, ncol=ncol, hx=hx, hy=hy, min_z=float(h.min()), max_z=float(h.max()))
            cfg = ModelBuilder.ShapeConfig(mu=info["mu"], restitution=0.0, density=0.0)
            sid = self.add_shape_heightfield(xform=wp.transform(wp.vec3(cx, cy, 0.0), wp.quat_identity()), heightfield=hf, cfg=cfg, label=f"{info['prim_path']}/heightfield")
            info["raster"] = dict(h=h, x_min=x_min, y_min=y_min, res=res, nrow=nrow, ncol=ncol)
            print(f"[newton] heightfield: terrain '{name}' -> {nrow}x{ncol} cells @ {res} m over x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}], z[{h.min():.3f},{h.max():.3f}], shape {sid}; trimesh collider excluded", flush=True)
        return out

    ModelBuilder.add_usd = add_usd_with_heightfield
    setattr(ModelBuilder, _SENTINEL, True)
    return True


apply_newton_heightfield_terrain_patch()
