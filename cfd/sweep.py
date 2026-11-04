"""Parameter sweep driver: many cavity simulations at once, rendered as they run.

Each tile is an independent whole-domain simulation at its own Rayleigh number,
so there is no halo exchange here and nothing to checkpoint. For the decomposed
single-simulation mode, where DDict carries halos across ranks, see halo.py.

Two Dragon pieces, each doing real work:

  Pool.map   distributes whole simulations across every core on every node.
  DDict      carries live per-tile status that the driver polls and displays
             while the sweep is still running.

Frames are rendered by the worker that produced them. An earlier version shipped
snapshots through a Queue to a separate render pool; that moved megabytes per
frame between nodes so a different process could colour-map an array, and the
termination protocol for a cross-node queue proved a reliable source of bugs.
Rendering is cheap and simulation is not, so the data never needs to move.

The same script runs on one node or many; only the launch command changes:

    dragon sweep.py                        # single node
    dragon --hostfile hosts sweep.py       # multi node
"""

from __future__ import annotations

import argparse
import pathlib
import time

import dragon  # noqa: F401  (registers the dragon start method)
import multiprocessing as mp
import numpy as np
from dragon import DDict

import render
from lbm import Cavity, Params

PROGRESS_POLL_SECS = 2.0


def tile_rayleigh(n_tiles: int, lo: float = 1e4, hi: float = 1e7) -> list[float]:
    """Log-spaced Rayleigh numbers so each tile is visibly different physics."""
    return list(np.logspace(np.log10(lo), np.log10(hi), n_tiles))


def simulate_tile(tile_id, ra, cfg, status, out_dir, tile_size):
    """Run one cavity simulation, rendering each frame in place as it is produced."""
    params = Params(n=cfg["n"], ra=ra, u_ref=cfg["u_ref"], swap_period=cfg["swap_period"])
    sim = Cavity(params)
    out = pathlib.Path(out_dir)
    produced = 0

    def on_frame(snap):
        nonlocal produced
        render.render_frame(snap, size=tile_size, vectors=cfg["vectors"]).save(
            out / f"tile{tile_id:02d}_{snap['step']:08d}.png"
        )
        produced += 1
        status[f"tile/{tile_id}"] = {
            "ra": ra,
            "step": snap["step"],
            "nu": snap["nu"],
            "frames": produced,
        }

    t0 = time.perf_counter()
    sim.run(cfg["steps"], frame_every=cfg["frame_every"], on_frame=on_frame)
    elapsed = time.perf_counter() - t0

    status[f"done/{tile_id}"] = elapsed
    return tile_id, ra, elapsed, produced


def report_progress(status, n_tiles, expected_per_tile):
    """Read live status out of the DDict and summarise it on one line."""
    done = frames = 0
    for i in range(n_tiles):
        rec = status.get(f"tile/{i}")
        if rec:
            frames += rec["frames"]
            if rec["frames"] >= expected_per_tile:
                done += 1
    total = n_tiles * expected_per_tile
    pct = 100.0 * frames / total if total else 0.0
    print(f"  progress: {frames}/{total} frames ({pct:.0f}%), {done}/{n_tiles} tiles complete", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--frame-every", type=int, default=200)
    ap.add_argument("--u-ref", type=float, default=0.1)
    ap.add_argument("--swap-period", type=int, default=0)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--vectors", type=int, default=0, help="cell stride between velocity arrows; 0 disables")
    ap.add_argument("--workers", type=int, default=0, help="pool size; 0 lets Dragon choose. Sizing this to available\n cores avoids oversubscribing when there are more tiles than cores.")
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()

    mp.set_start_method("dragon")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Workers on other nodes may run under a different uid on a shared mount, so
    # the output directory has to be writable by them. The driver itself can be
    # placed on a node where it does not own the directory, so this may fail.
    try:
        out_dir.chmod(0o777)
    except OSError:
        pass

    with DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024**2) as status:
        cfg = {
            "n": args.n,
            "steps": args.steps,
            "frame_every": args.frame_every,
            "u_ref": args.u_ref,
            "swap_period": args.swap_period,
            "vectors": args.vectors,
        }
        rayleighs = tile_rayleigh(args.tiles)
        per_tile = args.steps // args.frame_every

        print(f"{args.tiles} tiles, n={args.n}, {args.steps} steps each")
        print(f"Ra from {rayleighs[0]:.1e} to {rayleighs[-1]:.1e}")

        tasks = [
            (i, ra, cfg, status, str(out_dir), args.tile_size)
            for i, ra in enumerate(rayleighs)
        ]

        t0 = time.perf_counter()
        with mp.Pool(args.workers or None) as pool:
            pending = pool.starmap_async(simulate_tile, tasks)
            while not pending.ready():
                time.sleep(PROGRESS_POLL_SECS)
                report_progress(status, args.tiles, per_tile)
            results = pending.get()
        sim_elapsed = time.perf_counter() - t0

        expected = args.tiles * per_tile
        produced = sum(r[3] for r in results)
        on_disk = len(list(out_dir.glob("*.png")))

        print(f"\nsimulation wall time: {sim_elapsed:.1f}s")
        slowest = max(results, key=lambda r: r[2])
        print(f"slowest tile: Ra={slowest[1]:.1e} took {slowest[2]:.1f}s")
        print(f"speedup vs serial: {sum(r[2] for r in results) / sim_elapsed:.1f}x")
        print(f"frames: expected={expected} produced={produced} on_disk={on_disk}")
        print(f"frames written to {out_dir}/")


if __name__ == "__main__":
    main()
