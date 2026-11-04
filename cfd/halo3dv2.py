"""Domain-decomposed 3D cavity: one DDict per node, written to by the producer.

A variant of halo3d.py that drops the shared dictionary entirely. There are
exactly as many dictionaries as there are nodes, each with a single manager
pinned to its own node by Policy, and a rank **writes each face into the
dictionary belonging to the node that will read it**.

halo3d.py routes by ownership of the *producer*: a rank publishes into its own
node's dictionary when the neighbour is local and into a shared cluster-wide
dictionary when it is not, so a seam face crosses the network on the write, the
read, or both depending on where the shared dictionary hashed the key.

This version routes by ownership of the *consumer*. Rank r's left-going face is
consumed by rank r-1, so it is deposited in the dictionary of whichever node
hosts r-1. The consequence is the point of the design:

    every read is node-local, for every rank, including seam ranks.

A rank only ever reads `local_dicts[its own host]`, because by construction its
neighbours put the data there. Only the write side can cross the network, and
only for the two seam faces of a block decomposition.

    dragon --hostfile hostfile --network-prefix '^bridge' halo3dv2.py \
        --ranks 16 --n 128 --config rb --floor-profile bump
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import time

import dragon  # noqa: F401  (registers the 'dragon' start method)
import multiprocessing as mp
import numpy as np
from dragon import DDict, Policy, ProcessGroup, ProcessTemplate, System
from dragon.native.machine import Node

from halo2 import assign_nodes
from halo3d import Slab3D, placement_from_spec, slab_bounds


def node_ddict(host: str, total_mem: int, working_set: int) -> DDict:
    """One single-manager dictionary pinned to `host`."""
    policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=host)
    return DDict(
        None,
        None,
        total_mem,
        managers_per_policy=1,
        policy=[policy],
        working_set_size=working_set,
        wait_for_keys=True,
        timeout=900,
    )


def rank_worker(local_dicts, results, placement, rank, n_ranks, params, steps,
                frame_every, dtype_name, frames_dir):
    host = placement[rank]
    slab = Slab3D(params, rank, n_ranks, dtype=np.dtype(dtype_name))

    # Read from this node's dictionary only. Neighbours deposit here precisely
    # so that this is always true.
    mine = local_dicts[host]

    # Write each face to the node that consumes it: to_left feeds left_nbr,
    # to_right feeds right_nbr.
    left_out = local_dicts[placement[slab.left_nbr]]
    right_out = local_dicts[placement[slab.right_nbr]]

    # Deduplicate by identity; under block placement these are usually all the
    # same object. Every dictionary this rank touches must be advanced in step,
    # or a reader on it blocks forever at a checkpoint this rank never reached.
    used = list({id(d): d for d in (mine, left_out, right_out)}.values())
    for d in used:
        d.checkpoint_id = 0

    slab.collide()
    to_left, to_right = slab.faces()
    left_out[f"halo/{rank}/left"] = to_left
    right_out[f"halo/{rank}/right"] = to_right

    t_exchange = t_compute = 0.0
    start = time.perf_counter()

    for k in range(steps):
        t0 = time.perf_counter()
        from_left = np.asarray(mine[f"halo/{slab.left_nbr}/right"])
        from_right = np.asarray(mine[f"halo/{slab.right_nbr}/left"])
        t1 = time.perf_counter()

        slab.fill_ghosts(from_left, from_right)
        slab.stream()
        slab.collide()
        t2 = time.perf_counter()

        for d in used:
            d.checkpoint()
        to_left, to_right = slab.faces()
        left_out[f"halo/{rank}/left"] = to_left
        right_out[f"halo/{rank}/right"] = to_right
        t3 = time.perf_counter()

        t_exchange += (t1 - t0) + (t3 - t2)
        t_compute += t2 - t1

        if frame_every and (k + 1) % frame_every == 0:
            np.save(f"{frames_dir}/slab_{k + 1:08d}_{rank:03d}.npy", slab.owned_temp())

    results[f"final/{rank}"] = {
        "rank": rank,
        "host": socket.gethostname().split(".")[0],
        "x0": slab.x0,
        "x1": slab.x1,
        "columns": slab.width,
        "elapsed": time.perf_counter() - start,
        "exchange": t_exchange,
        "compute": t_compute,
        "remote_writes": sum(
            1 for nbr in (slab.left_nbr, slab.right_nbr) if placement[nbr] != host
        ),
        "temp": slab.owned_temp(),
        "finite": bool(np.isfinite(slab.rho[slab.owned]).all()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ny", type=int, default=0)
    ap.add_argument("--ra", type=float, default=1e7)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--u-ref", type=float, default=0.1)
    ap.add_argument("--config", default="rb", choices=("side", "rb"))
    ap.add_argument("--floor-profile", default="bump", choices=("uniform", "bump"))
    ap.add_argument("--bump-width", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placement", default="block", choices=("block", "cyclic"))
    ap.add_argument("--ranks-per-host", help="explicit split, e.g. 'studio=20,max=12'")
    ap.add_argument("--node-mem-gb", type=float, default=6.0,
                    help="memory for each node's dictionary; it now carries seam traffic too")
    ap.add_argument("--frames-mem-gb", type=float, default=12.0)
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--frames-dir")
    ap.add_argument("--frame-every", type=int, default=0)
    ap.add_argument("--keep-slabs", action="store_true")
    ap.add_argument("--float64", action="store_true")
    args = ap.parse_args()

    mp.set_start_method("dragon")
    from lbm3d import Params3D

    params = Params3D(
        n=args.n, ny=args.ny, ra=args.ra, u_ref=args.u_ref, config=args.config,
        floor_profile=args.floor_profile, bump_width=args.bump_width, seed=args.seed,
    )
    dtype_name = "float64" if args.float64 else "float32"

    alloc = System()
    hostnames = [Node(nid).hostname for nid in alloc.nodes]
    nnodes = alloc.nnodes
    if args.ranks_per_host:
        placement = placement_from_spec(args.ranks_per_host, hostnames)
        args.ranks = len(placement)
    else:
        placement = assign_nodes(args.ranks, hostnames, args.placement)

    # A face crosses the network only if its consumer sits on another node.
    remote_writes = sum(
        1
        for r in range(args.ranks)
        for nbr in ((r - 1) % args.ranks, (r + 1) % args.ranks)
        if placement[nbr] != placement[r]
    )

    cells = args.n * params.depth * args.n
    print(f"room {args.n}x{params.depth}x{args.n} = {cells / 1e6:.2f}M cells  Ra={args.ra:.0e}")
    for host in hostnames:
        print(f"  {host}: {sum(1 for h in placement if h == host)} ranks")
    print(f"ranks={args.ranks} nodes={nnodes} placement={args.placement}  "
          f"routing=consumer-node  {len(hostnames)} dicts, no shared dict")
    print(f"reads: {2 * args.ranks} node-local (all of them)  "
          f"writes: {2 * args.ranks - remote_writes} local, {remote_writes} remote",
          flush=True)

    frames_dir = ""
    if args.frames_dir:
        import pathlib

        out = pathlib.Path(args.frames_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            out.chmod(0o777)
        except OSError:
            pass
        frames_dir = str(out)

    with contextlib.ExitStack() as stack:
        local_dicts = {
            host: stack.enter_context(
                node_ddict(host, int(args.node_mem_gb * 1024**3), args.working_set)
            )
            for host in hostnames
        }
        results = stack.enter_context(DDict(1, nnodes, int(args.frames_mem_gb * 1024**3), timeout=900))

        grp = ProcessGroup(restart=False)
        for rank, host in enumerate(placement):
            grp.add_process(
                nproc=1,
                template=ProcessTemplate(
                    target=rank_worker,
                    args=(local_dicts, results, placement, rank, args.ranks,
                          params, args.steps, args.frame_every, dtype_name, frames_dir),
                    policy=Policy(placement=Policy.Placement.HOST_NAME, host_name=host),
                ),
            )

        t0 = time.perf_counter()
        grp.init()
        grp.start()
        grp.join()
        grp.close()
        wall = time.perf_counter() - t0

        rows = [results[f"final/{r}"] for r in range(args.ranks)]
        if not all(r["finite"] for r in rows):
            print("DIVERGED: non-finite density", flush=True)

        total_ex = sum(r["exchange"] for r in rows)
        total_cp = sum(r["compute"] for r in rows)
        print(f"wall: {wall:.1f}s   exchange share: {total_ex / (total_ex + total_cp) * 100:.1f}%")
        print(f"throughput: {cells * args.steps / wall / 1e6:.2f} MLUPS")
        print(f"remote writes: {sum(r['remote_writes'] for r in rows)} (expected {remote_writes})")

        if args.frames_dir:
            import pathlib

            out = pathlib.Path(frames_dir)
            steps_saved = sorted({int(p.stem.split("_")[1]) for p in out.glob("slab_*.npy")})
            written = 0
            for s in steps_saved:
                slabs = sorted(out.glob(f"slab_{s:08d}_*.npy"))
                if len(slabs) != args.ranks:
                    continue
                vol = np.zeros((args.n, params.depth, args.n), dtype=np.float32)
                for p in slabs:
                    r = int(p.stem.split("_")[2])
                    x0, x1 = slab_bounds(args.n, r, args.ranks)
                    vol[x0:x1, :, :] = np.load(p)
                np.savez_compressed(out / f"frame_{s:08d}.npz", step=s, ra=args.ra, temp=vol)
                written += 1
            if not args.keep_slabs:
                for p in out.glob("slab_*.npy"):
                    p.unlink()
            print(f"assembled {written} frames in {out}/")


if __name__ == "__main__":
    main()
