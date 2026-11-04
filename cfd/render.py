"""Turn solver snapshots into images and video.

Deliberately depends only on numpy and Pillow for rendering. matplotlib is not
used: every render worker would pay its import cost, and all we need is a
colour ramp. Encoding uses imageio's bundled ffmpeg, so there is no system
ffmpeg requirement.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re

import numpy as np
from PIL import Image, ImageDraw

# Anchors for a perceptually-ordered dark->bright ramp, close to inferno.
_INFERNO = np.array(
    [
        [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
        [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
        [252, 165, 10], [246, 215, 70], [252, 255, 164],
    ],
    dtype=np.float64,
)

# Diverging blue->white->red, natural for temperature about a mid-point.
_COOLWARM = np.array(
    [
        [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249],
        [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77],
        [180, 4, 38],
    ],
    dtype=np.float64,
)

# Reversed: coldest fluid is light blue, hot fades to black. For renders that
# cut away the hot side, so the warm bulk sinks into the background.
_ICE = np.array(
    [
        [214, 243, 255], [165, 221, 250], [116, 191, 240], [74, 155, 220],
        [44, 112, 182], [26, 74, 134], [14, 42, 85], [6, 18, 40], [0, 0, 0],
    ],
    dtype=np.float64,
)

RAMPS = {"inferno": _INFERNO, "coolwarm": _COOLWARM, "ice": _ICE}



def _build_lut(anchors: np.ndarray, size: int = 256) -> np.ndarray:
    """Linearly interpolate anchor colours into a 256-entry lookup table."""
    src = np.linspace(0.0, 1.0, len(anchors))
    dst = np.linspace(0.0, 1.0, size)
    return np.stack([np.interp(dst, src, anchors[:, c]) for c in range(3)], axis=1).astype(np.uint8)


_LUTS = {name: _build_lut(a) for name, a in RAMPS.items()}


def colorize(field: np.ndarray, vmin: float, vmax: float, ramp: str = "coolwarm") -> np.ndarray:
    """Map a 2-D field to an RGB uint8 image."""
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    return _LUTS[ramp][idx]


def draw_vectors(
    img: Image.Image, snap: dict, stride: int = 16, ref_frac: float = 0.25, gain: float = 0.9, color=(255, 255, 255)
):
    """Overlay a sparse velocity field as arrows, drawn in place.

    Lengths scale against a fixed fraction of u_ref rather than each frame's own
    maximum, so an arrow means the same speed in every frame and every tile.
    Measured peak speeds run 14-24% of u_ref across Ra 1e4-1e7, so a quarter of
    u_ref is close to full-scale without clipping the fast tiles.
    """
    ux, uy = snap.get("ux"), snap.get("uy")
    if ux is None or uy is None:
        return img

    n = ux.shape[0]
    px = img.size[0] / n
    ref = float(snap.get("u_ref") or 0.05) * ref_frac
    max_len = stride * px * gain
    draw = ImageDraw.Draw(img)

    for i in range(stride // 2, n, stride):
        for j in range(stride // 2, n, stride):
            vx, vy = float(ux[i, j]), float(uy[i, j])
            mag = math.hypot(vx, vy)
            if mag < 1e-9:
                continue
            length = min(mag / ref, 1.0) * max_len
            if length < 1.5:  # below this an arrow is indistinguishable from a dot
                continue
            # Image rows run opposite to y, matching the flip in render_frame.
            cx, cy = (i + 0.5) * px, (n - 1 - j + 0.5) * px
            dx, dy = vx / mag * length, -vy / mag * length
            x0, y0 = cx - dx / 2, cy - dy / 2
            x1, y1 = cx + dx / 2, cy + dy / 2
            draw.line([(x0, y0), (x1, y1)], fill=color, width=1)
            head = max(2.0, length * 0.35)
            ang = math.atan2(dy, dx)
            for side in (2.6, -2.6):
                draw.line(
                    [(x1, y1), (x1 + head * math.cos(ang + side), y1 + head * math.sin(ang + side))],
                    fill=color,
                    width=1,
                )
    return img


def render_frame(snap: dict, size: int = 512, ramp: str = "coolwarm", label: bool = True, vectors: int = 0) -> Image.Image:
    """Render one snapshot as a square image, oriented physically.

    Arrays are indexed [x, y]; transposing and flipping puts x to the right and
    y upward so buoyant plumes rise on screen instead of travelling sideways.
    vectors, if non-zero, is the cell stride between velocity arrows.
    """
    temp = snap["temp"].T[::-1]
    rgb = colorize(temp, -0.5, 0.5, ramp)
    img = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.BILINEAR)

    if vectors:
        draw_vectors(img, snap, stride=vectors)

    if label:
        draw = ImageDraw.Draw(img)
        nu = snap.get("nu", float("nan"))
        text = f"Ra={snap['ra']:.0e}"
        if np.isfinite(nu):
            text += f"   Nu={nu:.2f}"
        draw.text((8, 6), text, fill=(255, 255, 255))
    return img


def tile(images: list[Image.Image], cols: int, pad: int = 4, bg=(12, 12, 16)) -> Image.Image:
    """Compose frames into a grid montage."""
    if not images:
        raise ValueError("no images to tile")
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def encode(frames, out_path: str, fps: int = 30):
    """Write an mp4 from an iterable of PIL images."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for im in frames:
            writer.append_data(np.asarray(im))
    finally:
        writer.close()


def _montage_frames(paths: list[pathlib.Path], cols: int | None):
    """Group sweep PNGs by timestep and yield one montage image per step.

    sweep.py names frames tile{id}_{step}.png, so a directory holds every tile
    interleaved. Encoding them in filename order would cut between unrelated
    simulations; grouping by step gives one grid per instant instead.
    """
    by_step: dict[int, dict[int, pathlib.Path]] = {}
    for p in paths:
        m = re.fullmatch(r"tile(\d+)_(\d+)", p.stem)
        if not m:
            continue
        tile_id, step = int(m.group(1)), int(m.group(2))
        by_step.setdefault(step, {})[tile_id] = p

    if not by_step:
        raise SystemExit("PNGs found but none match the tile{id}_{step}.png naming")

    n_tiles = max(len(v) for v in by_step.values())
    # ffmpeg needs a constant frame size, so a step missing a tile has to go.
    steps = [s for s in sorted(by_step) if len(by_step[s]) == n_tiles]
    dropped = len(by_step) - len(steps)
    if cols is None:
        cols = math.ceil(math.sqrt(n_tiles))

    def gen():
        for s in steps:
            row = [Image.open(by_step[s][t]).convert("RGB") for t in sorted(by_step[s])]
            yield row[0] if n_tiles == 1 else tile(row, cols)

    return gen, len(steps), n_tiles, cols, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir", help="directory of .npz snapshots (lbm.py) or .png tiles (sweep.py)")
    ap.add_argument("--out", default="movie.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cols", type=int, default=None, help="montage columns; default is square-ish")
    ap.add_argument("--ramp", default="coolwarm", choices=sorted(RAMPS))
    ap.add_argument("--vectors", type=int, default=0, help="cell stride between velocity arrows; 0 disables")
    args = ap.parse_args()

    root = pathlib.Path(args.frames_dir)
    npz_paths = sorted(root.glob("frame_*.npz"))

    if npz_paths:
        def gen():
            for p in npz_paths:
                with np.load(p) as snap:
                    yield render_frame(dict(snap), size=args.size, ramp=args.ramp, vectors=args.vectors)

        encode(gen(), args.out, args.fps)
        print(f"wrote {args.out} from {len(npz_paths)} snapshots")
        return

    png_paths = sorted(root.glob("*.png"))
    if not png_paths:
        raise SystemExit(f"no frame_*.npz or *.png found in {root}")

    gen, n_steps, n_tiles, cols, dropped = _montage_frames(png_paths, args.cols)
    if not n_steps:
        raise SystemExit("no timestep has a complete set of tiles")
    encode(gen(), args.out, args.fps)
    layout = "single tile" if n_tiles == 1 else f"{n_tiles} tiles in {cols} columns"
    note = f", dropped {dropped} incomplete step(s)" if dropped else ""
    print(f"wrote {args.out} from {n_steps} steps, {layout}{note}")


if __name__ == "__main__":
    main()
