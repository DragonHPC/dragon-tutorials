"""Oblique volume rendering for the 3D cavity.

Projecting straight down an axis shows a flat rectangle, so the camera here is
always oblique: the volume is rotated by azimuth/elevation and then composited
front to back along the view axis.

Opacity is keyed to |grad T| rather than to temperature. Temperature fills the
whole box smoothly, so compositing it directly gives uniform fog; its gradient
is near zero in well-mixed fluid and large at plume edges and boundary layers,
which leaves the thermal structure hanging in clear space.

Rotating the volume once per frame and then compositing along an axis costs
~2M trilinear samples, where casting rays through the original grid at image
resolution would cost ~30M for the same picture.

    python vol3d.py frames/ --out room.mp4 --spin 1.5
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from render import colorize, encode


def rotation(az_deg: float, el_deg: float) -> np.ndarray:
    """Camera rotation: azimuth about the vertical axis, then elevation.

    Fields arrive as (x, y, z) with z vertical, and compositing runs down axis
    0, so axis 0 is depth and axis 1 is the image horizontal. Azimuth must
    therefore turn about axis 2 to orbit the room with the floor staying down;
    turning about axis 1 instead tumbles it end over end, and a quarter of the
    way round the camera is looking straight at the ceiling.
    """
    az, el = np.radians(az_deg), np.radians(el_deg)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    r_az = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    r_el = np.array([[ce, 0.0, se], [0.0, 1.0, 0.0], [-se, 0.0, ce]])
    return r_el @ r_az


def resample(vol: np.ndarray, rot: np.ndarray, out_shape: tuple[int, int, int], order: int = 1):
    """Sample vol onto a rotated grid, with anything outside the box set to NaN."""
    grids = np.indices(out_shape, dtype=np.float32)
    out_c = (np.array(out_shape, dtype=np.float32) - 1.0) / 2.0
    in_c = (np.array(vol.shape, dtype=np.float32) - 1.0) / 2.0
    offs = grids.reshape(3, -1) - out_c[:, None]
    coords = rot @ offs + in_c[:, None]
    flat = map_coordinates(vol, coords, order=order, mode="constant", cval=np.nan)
    return flat.reshape(out_shape)


def _zero_shell(vol: np.ndarray, trim: int = 1) -> np.ndarray:
    """Blank the near-wall region.

    Walls hold bounce-back values, and the thermal boundary layers just inside
    them are the strongest feature in the box - left visible they wrap the
    interior in an opaque shell and there is nothing to see into.
    """
    vol[:trim, :, :] = vol[-trim:, :, :] = 0.0
    vol[:, :trim, :] = vol[:, -trim:, :] = 0.0
    vol[:, :, :trim] = vol[:, :, -trim:] = 0.0
    return vol


def opacity_from_gradient(
    temp: np.ndarray, percentile: float = 97.0, gamma: float = 0.7, trim: int = 1
) -> np.ndarray:
    """Opacity proportional to local thermal structure, normalised robustly."""
    gx, gy, gz = np.gradient(temp.astype(np.float32))
    mag = _zero_shell(np.sqrt(gx * gx + gy * gy + gz * gz), trim)
    ref = np.percentile(mag, percentile)
    return np.zeros_like(mag) if ref <= 0 else np.clip(mag / ref, 0.0, 1.0) ** gamma


def opacity_from_highpass(
    temp: np.ndarray, sigma: float = 2.5, percentile: float = 99.0, gamma: float = 3.0, trim: int = 1
) -> np.ndarray:
    """Opacity from what is left after removing the smooth background.

    A differentially heated cavity develops a stratified core whose gradient
    spans the whole box, so |grad T| marks almost everything and the render
    fogs over. Subtracting a blurred copy keeps only the plumes and the wall
    boundary layers. gamma is high because the view integrates ~60 slabs: an
    alpha of 0.07 per slab is already opaque by the back of the box.
    """
    t = temp.astype(np.float32)
    hp = _zero_shell(np.abs(t - gaussian_filter(t, sigma)), trim)
    ref = np.percentile(hp, percentile)
    return np.zeros_like(hp) if ref <= 0 else np.clip(hp / ref, 0.0, 1.0) ** gamma


def opacity_from_threshold(
    temp: np.ndarray, threshold: float = 0.1, softness: float = 0.05,
    trim: int = 1, invert: bool = False,
) -> np.ndarray:
    """Opacity from a temperature cut: only fluid past threshold shows.

    The high-pass modes mark every thermal edge, including the stratified core
    and the cold descending sheets, which is a lot of structure to read at once.
    A cut keeps just the buoyant fluid, so the plume above the heated patch
    stands alone in clear space. softness ramps the cut over a temperature band
    rather than switching it hard, which would alias into stair-stepped shells.
    invert keeps the cold side instead, which hides the plume and leaves the
    sinking return flow.
    """
    t = temp.astype(np.float32)
    d = (threshold - t) if invert else (t - threshold)
    a = d / softness if softness > 0 else (d > 0).astype(np.float32)
    return _zero_shell(np.clip(a, 0.0, 1.0), trim)


def opacity_two_sided(
    temp: np.ndarray, threshold: float = 0.1, cold_threshold: float = -0.4,
    softness: float = 0.05, trim: int = 1, cold_alpha: float = 0.35,
) -> np.ndarray:
    """Opacity from both tails at once, with the cold side held back.

    Showing the plume and its return flow together needs two cuts rather than
    one, and they cannot carry equal weight: the cold tail of this room is an
    order of magnitude fatter than the hot tail, so matching their opacities
    buries the plume inside a blue shell. cold_alpha keeps the cold fluid as a
    translucent context for the opaque hot fluid.
    """
    t = temp.astype(np.float32)
    a_hot = np.clip((t - threshold) / softness, 0.0, 1.0)
    a_cold = np.clip((cold_threshold - t) / softness, 0.0, 1.0) * cold_alpha
    return _zero_shell(np.maximum(a_hot, a_cold), trim)


def slice_frame(
    temp: np.ndarray, axis: str = "y", at: float = 0.5, size: int = 600,
    vmin: float = -0.3, vmax: float = 0.3, ramp: str = "inferno",
    overlay: bool = False, threshold: float = 0.3, cold_threshold: float = -0.3,
    softness: float = 0.02, cold_alpha: float = 1.0, bg=(8, 8, 12),
) -> Image.Image:
    """A single plane through the room, drawn without depth integration.

    A volume render has to accumulate opacity along the view ray, which hides
    anything thin: the stem of a plume can be perfectly sharp in the field and
    still barely register. A plane shows the field as it is.

    With overlay set, everything between the two cuts is dropped to background
    and only the hot and cold extremes are drawn, which isolates the wall
    boundary layers and the plumes they shed.
    """
    i = int(np.clip(at, 0.0, 1.0) * (temp.shape[{"x": 0, "y": 1, "z": 2}[axis]] - 1))
    if axis == "x":
        plane = temp[i, :, :]
    elif axis == "y":
        plane = temp[:, i, :]
    else:
        plane = temp[:, :, i]
    # x and y slices hold the vertical axis last; stand it up and flip so the
    # floor is at the bottom. A z slice is already a horizontal plan view.
    plane = plane.T[::-1] if axis in ("x", "y") else plane.T

    if not overlay:
        return Image.fromarray(colorize(plane, vmin, vmax, ramp)).resize(
            (size, size), Image.NEAREST
        )

    p = plane.astype(np.float32)
    hot_span = max(float(p.max()) - threshold, 1e-6)
    cold_span = max(cold_threshold - float(p.min()), 1e-6)
    field = np.full(p.shape, 0.5, dtype=np.float32)
    hot_m, cold_m = p >= threshold, p <= cold_threshold
    field[hot_m] = 0.65 + 0.35 * np.clip((p[hot_m] - threshold) / hot_span, 0.0, 1.0) ** 0.4
    field[cold_m] = 0.35 - 0.35 * np.clip((cold_threshold - p[cold_m]) / cold_span, 0.0, 1.0) ** 0.4

    a = np.zeros(p.shape, dtype=np.float32)
    a[hot_m] = np.clip((p[hot_m] - threshold) / softness, 0.0, 1.0)
    a[cold_m] = np.clip((cold_threshold - p[cold_m]) / softness, 0.0, 1.0) * cold_alpha

    rgb = colorize(field, 0.0, 1.0, ramp).astype(np.float32)
    img = rgb * a[..., None] + np.array(bg, dtype=np.float32) * (1.0 - a[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )


def render_volume(
    temp: np.ndarray,
    az: float = 35.0,
    el: float = 22.0,
    size: int = 640,
    pad: float = 1.3,
    density: float = 0.35,
    depth_fade: float = 0.45,
    mode: str = "highpass",
    sigma: float = 2.5,
    gamma: float = 3.0,
    threshold: float = 0.1,
    softness: float = 0.05,
    invert: bool = False,
    cold_threshold: float = -0.4,
    cold_alpha: float = 0.35,
    trim: int = 3,
    ramp: str = "coolwarm",
    bg=(8, 8, 12),
) -> Image.Image:
    if mode == "gradient":
        alpha_vol = opacity_from_gradient(temp, gamma=gamma, trim=trim)
    elif mode == "twosided":
        alpha_vol = opacity_two_sided(
            temp, threshold=threshold, cold_threshold=cold_threshold,
            softness=softness, trim=trim, cold_alpha=cold_alpha,
        )
    elif mode == "threshold":
        alpha_vol = opacity_from_threshold(
            temp, threshold=threshold, softness=softness, trim=trim, invert=invert
        )
    else:
        alpha_vol = opacity_from_highpass(temp, sigma=sigma, gamma=gamma, trim=trim)
    rot = rotation(az, el)

    # Colour normally follows absolute temperature, but the two-sided cuts move
    # with the room's drifting mean: by the end of a run the hot cut sits near
    # zero, which is the pale middle of a diverging ramp, and the plume washes
    # out to white. Keying colour to distance past each cut keeps hot red and
    # cold blue whatever the cuts are doing.
    if mode == "twosided":
        t = temp.astype(np.float32)
        hot_span = max(float(np.nanmax(t)) - threshold, 1e-6)
        cold_span = max(cold_threshold - float(np.nanmin(t)), 1e-6)
        field = np.full(t.shape, 0.5, dtype=np.float32)
        hot_m, cold_m = t >= threshold, t <= cold_threshold
        # Sublinear: most visible fluid sits just past its cut, so a linear
        # ramp would leave the whole plume pale.
        hot_f = np.clip((t[hot_m] - threshold) / hot_span, 0.0, 1.0) ** 0.4
        cold_f = np.clip((cold_threshold - t[cold_m]) / cold_span, 0.0, 1.0) ** 0.4
        field[hot_m] = 0.65 + 0.35 * hot_f
        field[cold_m] = 0.35 - 0.35 * cold_f
        lo, hi = 0.0, 1.0
    else:
        field, lo, hi = temp.astype(np.float32), -0.5, 0.5

    side = int(round(max(temp.shape) * pad))
    shape = (side, side, side)
    t_r = resample(field, rot, shape)
    a_r = resample(alpha_vol, rot, shape)

    outside = ~np.isfinite(a_r)
    a_r[outside] = 0.0
    t_r[~np.isfinite(t_r)] = (lo + hi) / 2.0

    rgb_vol = colorize(t_r, lo, hi, ramp).astype(np.float32)

    # Front-to-back over the view axis; each slab occludes what is behind it.
    acc_rgb = np.zeros((side, side, 3), dtype=np.float32)
    acc_a = np.zeros((side, side), dtype=np.float32)
    for k in range(side):
        a = a_r[k] * density
        if not a.any():
            continue
        shade = 1.0 - depth_fade * (k / max(side - 1, 1))
        w = (1.0 - acc_a) * a
        acc_rgb += w[..., None] * rgb_vol[k] * shade
        acc_a += w

    img = acc_rgb + (1.0 - acc_a)[..., None] * np.array(bg, dtype=np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Depth runs down the first axis; transpose so the vertical axis is upright.
    return Image.fromarray(img.transpose(1, 0, 2)[::-1], mode="RGB").resize(
        (size, size), Image.BILINEAR
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir", help="directory of frame_*.npz from lbm3d.py")
    ap.add_argument("--out", default="volume.mp4")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=22.0)
    ap.add_argument("--spin", type=float, default=0.0, help="degrees of azimuth added per frame")
    ap.add_argument("--density", type=float, default=0.35)
    ap.add_argument("--mode", default="highpass", choices=("highpass", "gradient", "threshold", "twosided"))
    ap.add_argument("--sigma", type=float, default=2.5, help="blur width removed by the high-pass")
    ap.add_argument("--gamma", type=float, default=3.0, help="higher suppresses the bulk more")
    ap.add_argument("--threshold", type=float, default=0.1, help="threshold mode: show fluid hotter than this (-0.5 to 0.5)")
    ap.add_argument("--threshold-pct", type=float, default=None, help="threshold mode: set the cut per frame at this percentile instead of a fixed value")
    ap.add_argument("--softness", type=float, default=0.05, help="threshold mode: temperature width of the cut")
    ap.add_argument("--invert", action="store_true", help="threshold mode: keep the cold side of the cut instead of the hot")
    ap.add_argument("--cold-pct", type=float, default=2.0, help="twosided mode: percentile for the cold cut")
    ap.add_argument("--cold-alpha", type=float, default=0.35, help="twosided mode: opacity of the cold fluid relative to the hot")
    ap.add_argument("--still-index", type=int, default=-1, help="which frame --still renders")
    ap.add_argument("--trim", type=int, default=3, help="cells of near-wall region to blank")
    ap.add_argument("--ramp", default="coolwarm")
    ap.add_argument("--still", action="store_true", help="render only the last frame, as a png")
    ap.add_argument("--slice", dest="slice_axis", choices=("x", "y", "z"), default=None, help="draw a plane through the room instead of rendering the volume")
    ap.add_argument("--slice-at", type=float, default=0.5, help="where the slice plane sits, as a fraction of the axis")
    ap.add_argument("--vmin", type=float, default=-0.3, help="slice: temperature at the bottom of the ramp")
    ap.add_argument("--vmax", type=float, default=0.3, help="slice: temperature at the top of the ramp")
    ap.add_argument("--turntable", type=int, default=0, help="spin the camera around one timestep, in N frames")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(args.frames_dir).glob("frame_*.npz"))
    if not paths:
        raise SystemExit(f"no frame_*.npz found in {args.frames_dir}")

    if args.slice_axis:
        overlay = args.mode == "twosided"
        sl = dict(
            axis=args.slice_axis, at=args.slice_at, size=args.size,
            vmin=args.vmin, vmax=args.vmax, ramp=args.ramp,
            overlay=overlay, softness=args.softness, cold_alpha=args.cold_alpha,
        )

        def slice_cut(temp):
            if not overlay:
                return {}
            t = max(args.trim, 1)
            core = temp[t:-t, t:-t, t:-t]
            hot = args.threshold_pct if args.threshold_pct is not None else 98.0
            return {
                "threshold": float(np.percentile(core, hot)),
                "cold_threshold": float(np.percentile(core, args.cold_pct)),
            }

        if args.still:
            with np.load(paths[args.still_index]) as snap:
                temp = snap["temp"]
                slice_frame(temp, **{**sl, **slice_cut(temp)}).save(args.out)
            print(f"wrote {args.out} from {paths[args.still_index].name}")
            return

        def planes():
            for p in paths:
                with np.load(p) as snap:
                    temp = snap["temp"]
                    yield slice_frame(temp, **{**sl, **slice_cut(temp)})

        encode(planes(), args.out, args.fps)
        print(f"wrote {args.out}: {len(paths)} {args.slice_axis}-slices at {args.slice_at:.2f}")
        return

    kw = dict(
        el=args.el, size=args.size, density=args.density, mode=args.mode,
        sigma=args.sigma, gamma=args.gamma, threshold=args.threshold,
        softness=args.softness, invert=args.invert, cold_alpha=args.cold_alpha,
        trim=args.trim, ramp=args.ramp,
    )

    def cut(temp):
        """The room's mean temperature drifts as it heats, so a fixed cut that
        isolates the plume at the end buries it in warm bulk earlier on."""
        t = args.trim
        core = temp[t:-t, t:-t, t:-t]
        if args.mode == "twosided":
            hot = args.threshold_pct if args.threshold_pct is not None else 99.5
            return {
                "threshold": float(np.percentile(core, hot)),
                "cold_threshold": float(np.percentile(core, args.cold_pct)),
            }
        if args.threshold_pct is None:
            return {}
        return {"threshold": float(np.percentile(core, args.threshold_pct))}

    if args.still:
        with np.load(paths[args.still_index]) as snap:
            temp = snap["temp"]
            img = render_volume(temp, az=args.az, **{**kw, **cut(temp)})
        img.save(args.out)
        print(f"wrote {args.out} from {paths[args.still_index].name}")
        return

    if args.turntable:
        # Parallax from a moving camera reads depth better than a time sequence.
        with np.load(paths[-1]) as snap:
            temp = snap["temp"]
        kw.update(cut(temp))

        def spin():
            for i in range(args.turntable):
                yield render_volume(temp, az=args.az + 360.0 * i / args.turntable, **kw)

        encode(spin(), args.out, args.fps)
        print(f"wrote {args.out}: {args.turntable} views of {paths[-1].name}")
        return

    def gen():
        for i, p in enumerate(paths):
            with np.load(p) as snap:
                temp = snap["temp"]
                yield render_volume(temp, az=args.az + i * args.spin, **{**kw, **cut(temp)})

    encode(gen(), args.out, args.fps)
    print(f"wrote {args.out} from {len(paths)} frames, az {args.az} to {args.az + (len(paths) - 1) * args.spin:.0f}")


if __name__ == "__main__":
    main()
