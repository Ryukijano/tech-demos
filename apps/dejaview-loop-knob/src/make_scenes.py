# SPDX-License-Identifier: Apache-2.0
"""Generate small synthetic multi-view scenes with exact ground-truth cameras and depth.

Kubric-style tabletop scenes (textured ground + textured primitives), rendered with a
pure-numpy perspective-correct z-buffer rasterizer. No GPU, no OpenGL, no external
renderer, no third-party assets — everything is procedural, so the data is
licence-clean. DVLT's training mix includes synthetic data (e.g. Kubric), which makes
this a reasonable, if easy, domain.

Conventions (match DVLT / OpenCV):
  - camera-to-world extrinsics c2w = [R | t], camera +z forward, +x right, +y down
  - depth = camera-space z
  - world is z-up

Outputs per scene under  <out>/scene<i>/ :
  view_<j>.png            RGB render (render_size, downsampled to image_size)
  depth.npy               (V, H, W) float32 GT depth, 0 where sky/invalid
  cameras.npz             c2w (V, 3, 4) float64, K (3, 3) float64
  preview.png             grid of all candidate views
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# ------------------------------- geometry builders -------------------------------


def box(center, size, yaw):
    """Axis box of half-extents `size`, rotated by yaw around z, at `center` (base on z=0 handled by caller)."""
    sx, sy, sz = size
    c = np.array(
        [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ]
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    v = c @ rot.T + np.asarray(center)
    f = [
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    return v, np.array(f)


def icosphere(center, radius, subdiv=2):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )
    for _ in range(subdiv):
        vlist = list(verts)
        cache = {}
        new_faces = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = vlist[a] + vlist[b]
                m = m / np.linalg.norm(m)
                cache[key] = len(vlist)
                vlist.append(m)
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        verts = np.array(vlist)
        faces = np.array(new_faces)
    return verts * radius + np.asarray(center), faces


def pyramid(center, half_base, height, yaw):
    c = np.array(
        [
            [-half_base, -half_base, 0], [half_base, -half_base, 0],
            [half_base, half_base, 0], [-half_base, half_base, 0],
            [0, 0, height],
        ]
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    v = c @ rot.T + np.asarray(center)
    f = [(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    return v, np.array(f)


def ground_grid(half, n):
    """Ground plane z=0 subdivided into an n x n grid (so near-plane clipping is per-cell)."""
    xs = np.linspace(-half, half, n + 1)
    verts, faces = [], []
    for i in range(n + 1):
        for j in range(n + 1):
            verts.append([xs[j], xs[i], 0.0])
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces += [(a, b, d), (a, d, c)]
    return np.array(verts), np.array(faces)


# ------------------------------- camera helpers -------------------------------


def look_at_c2w(pos, target, world_up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=float)
    z = np.asarray(target, dtype=float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, world_up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # points "down" in a z-up world -> OpenCV convention
    rot = np.stack([x, y, z], axis=1)
    return np.concatenate([rot, pos[:, None]], axis=1)  # (3, 4)


# ------------------------------- rasterizer -------------------------------


def render(verts, faces, face_colors, checker_freq, c2w, K, H, W, near=0.05):
    """Perspective-correct z-buffer rasterizer. Returns (rgb uint8 HxWx3, depth float32 HxW; 0 = sky)."""
    rot, t = c2w[:, :3], c2w[:, 3]
    cam = (verts - t) @ rot  # world -> camera (R^T (X - t))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    # sky: vertical gradient
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    g = np.linspace(0.65, 0.95, H)[:, None]
    rgb[..., 0] = 0.55 * g
    rgb[..., 1] = 0.70 * g
    rgb[..., 2] = 0.95 * g

    light = np.array([0.4, 0.25, 0.88])
    light /= np.linalg.norm(light)

    tri_cam = cam[faces]  # (F, 3, 3)
    tri_world = verts[faces]
    keep = (tri_cam[:, :, 2] > near).all(axis=1)
    tri_cam, tri_world, cols = tri_cam[keep], tri_world[keep], face_colors[keep]

    z = tri_cam[:, :, 2]
    u = fx * tri_cam[:, :, 0] / z + cx
    v = fy * tri_cam[:, :, 1] / z + cy

    # normals + lambert shading per face
    n = np.cross(tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(norm, 1e-12, None)
    lam = np.abs(n @ light)
    shade = 0.45 + 0.55 * lam

    order = np.argsort(-z.mean(axis=1))  # rough far-to-near, irrelevant with z-buffer but cheap
    for fi in order:
        u0, u1, u2 = u[fi]
        v0, v1, v2 = v[fi]
        xmin = max(int(np.floor(min(u0, u1, u2))), 0)
        xmax = min(int(np.ceil(max(u0, u1, u2))) + 1, W)
        ymin = max(int(np.floor(min(v0, v1, v2))), 0)
        ymax = min(int(np.ceil(max(v0, v1, v2))) + 1, H)
        if xmin >= xmax or ymin >= ymax:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax) + 0.5, np.arange(ymin, ymax) + 0.5)
        d = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
        if abs(d) < 1e-12:
            continue
        w1 = ((gx - u0) * (v2 - v0) - (gy - v0) * (u2 - u0)) / d
        w2 = ((gy - v0) * (u1 - u0) - (gx - u0) * (v1 - v0)) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z0, z1, z2 = z[fi]
        inv_z = w0 / z0 + w1 / z1 + w2 / z2
        zpix = 1.0 / inv_z
        sub = zbuf[ymin:ymax, xmin:xmax]
        upd = inside & (zpix < sub)
        if not upd.any():
            continue
        # perspective-correct world position for the 3D checker texture
        pw = (
            w0[..., None] * tri_world[fi, 0] / z0
            + w1[..., None] * tri_world[fi, 1] / z1
            + w2[..., None] * tri_world[fi, 2] / z2
        ) * zpix[..., None]
        f = checker_freq[0] if cols[fi, 3] > 0.5 else checker_freq[1]
        check = (np.floor(pw[..., 0] * f) + np.floor(pw[..., 1] * f) + np.floor(pw[..., 2] * f)) % 2
        mod = 0.55 + 0.45 * check
        col = cols[fi, :3][None, None, :] * (mod * shade[fi])[..., None]
        sub[upd] = zpix[upd]
        tgt = rgb[ymin:ymax, xmin:xmax]
        tgt[upd] = col[upd]

    depth = np.where(np.isinf(zbuf), 0.0, zbuf).astype(np.float32)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth


# ------------------------------- scene assembly -------------------------------

PALETTE = np.array(
    [
        [0.85, 0.25, 0.20], [0.20, 0.55, 0.85], [0.95, 0.75, 0.15],
        [0.30, 0.75, 0.35], [0.75, 0.35, 0.80], [0.90, 0.50, 0.20],
        [0.25, 0.80, 0.75], [0.80, 0.80, 0.80],
    ]
)


def build_scene(rng):
    all_v, all_f, all_c = [], [], []

    def add(v, f, color, obj_checker):
        base = len(np.concatenate(all_v)) if all_v else 0
        all_v.append(v)
        all_f.append(f + base)
        c = np.zeros((len(f), 4))
        c[:, :3] = color
        c[:, 3] = obj_checker
        all_c.append(c)

    gv, gf = ground_grid(half=8.0, n=16)
    add(gv, gf, np.array([0.55, 0.52, 0.48]), 1.0)

    n_obj = rng.integers(5, 8)
    placed = []
    for _ in range(n_obj):
        for _attempt in range(30):
            pos = rng.uniform(-2.0, 2.0, size=2)
            if all(np.linalg.norm(pos - p) > 1.1 for p in placed):
                break
        placed.append(pos)
        kind = rng.integers(0, 3)
        color = PALETTE[rng.integers(0, len(PALETTE))] * rng.uniform(0.85, 1.1)
        if kind == 0:
            size = rng.uniform(0.3, 0.65, size=3)
            v, f = box([pos[0], pos[1], size[2]], size, rng.uniform(0, np.pi))
        elif kind == 1:
            r = rng.uniform(0.35, 0.6)
            v, f = icosphere([pos[0], pos[1], r], r, subdiv=2)
        else:
            hb = rng.uniform(0.35, 0.6)
            v, f = pyramid([pos[0], pos[1], 0.0], hb, rng.uniform(0.8, 1.5), rng.uniform(0, np.pi))
        add(v, f, color, 0.0)

    verts = np.concatenate(all_v)
    faces = np.concatenate(all_f)
    cols = np.concatenate(all_c)
    freqs = (rng.uniform(0.8, 1.4), rng.uniform(2.0, 3.5))  # (ground, objects)
    return verts, faces, cols, freqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--n-views", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=252)
    ap.add_argument("--supersample", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out)
    size = args.image_size
    rs = size * args.supersample
    fx = 1.1 * size

    meta = {"image_size": size, "n_views": args.n_views, "fx": fx, "scenes": []}
    for si in range(args.n_scenes):
        rng = np.random.default_rng(1000 + si)
        verts, faces, cols, freqs = build_scene(rng)
        sdir = out_root / f"scene{si}"
        sdir.mkdir(parents=True, exist_ok=True)

        K = np.array([[fx, 0, size / 2.0], [0, fx, size / 2.0], [0, 0, 1.0]])
        Krs = K.copy()
        Krs[:2] *= args.supersample

        c2ws, depths = [], []
        for vi in range(args.n_views):
            az = 2 * np.pi * vi / args.n_views + rng.uniform(-0.08, 0.08)
            el = np.deg2rad(rng.uniform(25, 50))
            rad = rng.uniform(4.5, 6.5)
            pos = np.array([rad * np.cos(az) * np.cos(el), rad * np.sin(az) * np.cos(el), rad * np.sin(el)])
            target = rng.uniform(-0.3, 0.3, size=3) * np.array([1, 1, 0.5]) + np.array([0, 0, 0.5])
            c2w = look_at_c2w(pos, target)
            rgb_hi, depth_hi = render(verts, faces, cols, freqs, c2w, Krs, rs, rs)
            img = Image.fromarray(rgb_hi).resize((size, size), Image.LANCZOS)
            img.save(sdir / f"view_{vi:02d}.png")
            depths.append(depth_hi[:: args.supersample, :: args.supersample])
            c2ws.append(c2w)
            print(f"scene{si} view{vi} done", flush=True)

        np.save(sdir / "depth.npy", np.stack(depths))
        np.savez(sdir / "cameras.npz", c2w=np.stack(c2ws), K=K)
        meta["scenes"].append({"name": f"scene{si}", "n_faces": int(len(faces))})

        tiles = [np.array(Image.open(sdir / f"view_{vi:02d}.png")) for vi in range(args.n_views)]
        rows = []
        per_row = 5
        for r in range(0, args.n_views, per_row):
            rows.append(np.concatenate(tiles[r : r + per_row], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(sdir / "preview.png")

    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print("all scenes done")


if __name__ == "__main__":
    main()
