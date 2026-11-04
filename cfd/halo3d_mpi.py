"""Domain-decomposed 3D cavity: Dragon launches the ranks, MPI moves the halos.

Same solver as halo3d.py and the same 1D decomposition along x, but the face
exchange goes through MPI_Sendrecv instead of a DDict. Dragon still owns the
run: it stands up the PMIx servers, places the ranks, and carries the results
back in a DDict, so this is a straight swap of the exchange mechanism and the
wall/exchange/compute numbers are directly comparable to halo3d.py.

Two things fall out of the swap:

  * No checkpoints. The DDict version needs a publish/advance/read cycle to
    keep ranks in lockstep on a versioned key. Sendrecv is already a rendezvous
    between exactly the two ranks that share a face, so the lockstep is
    implicit and the working-set machinery disappears.
  * No face routing. halo3d.py sends node-local faces to a per-node dictionary
    and seams to a shared one because routing everything through one DDict
    measured 1.4-3.5x slower. MPI picks the path itself, so there is nothing
    to route.

Placement is unaffected. A PMI group numbers ranks in add_process order, so one
call per run of consecutive same-host ranks reproduces halo3d.py's block and
cyclic layouts exactly, and --placement/--ranks-per-host mean the same thing
here as they do there.

Needs mpi4py built against the same MPI as the nodes:

    MPICC=/opt/homebrew/bin/mpicc pip install --no-binary=mpi4py mpi4py

Export this on the frontend node before launching, or the ranks cannot load the
Open MPI libraries and the run dies waiting for TAUp. Local Services builds each
child's environment from its own, so one export reaches every node and rank, and
Dragon only prepends its own lib dir rather than replacing the value:

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH}

    dragon --hostfile hostfile --network-prefix '^bridge' halo3d_mpi.py \
        --ranks 16 --n 128 --config rb --floor-profile bump
"""

from __future__ import annotations

import argparse

import mpi4py

# Must precede the MPI import: under Dragon the PMIx environment is only in
# place once the process is running, so Init cannot happen at import time.
mpi4py.rc.initialize = False

import os
import socket
import sys
import time

import numpy as np
from mpi4py import MPI


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--frames-mem-gb", type=float, default=12.0)
    ap.add_argument("--frames-dir")
    ap.add_argument("--frame-every", type=int, default=0)
    ap.add_argument("--keep-slabs", action="store_true")
    ap.add_argument("--float64", action="store_true")
    # Set by the driver when it re-invokes this file as the MPI executable.
    ap.add_argument("--worker-dser", help=argparse.SUPPRESS)
    return ap


def make_params(args):
    from lbm3d import Params3D

    return Params3D(
        n=args.n, ny=args.ny, ra=args.ra, u_ref=args.u_ref, config=args.config,
        floor_profile=args.floor_profile, bump_width=args.bump_width, seed=args.seed,
    )


# --------------------------------------------------------------------------
# MPI rank
# --------------------------------------------------------------------------


def run_worker(args) -> None:
    MPI.Init()

    from dragon import DDict

    from halo3d import Slab3D

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    dtype = np.dtype("float64" if args.float64 else "float32")
    slab = Slab3D(make_params(args), rank, n_ranks, dtype=dtype)

    # Slab3D already wraps periodically, so these match the DDict version's
    # neighbours exactly and no Cartesian communicator is needed.
    left_nbr, right_nbr = slab.left_nbr, slab.right_nbr

    slab.collide()
    to_left, to_right = slab.faces()
    from_left = np.empty_like(to_right)
    from_right = np.empty_like(to_left)

    t_exchange = t_compute = 0.0
    start = time.perf_counter()

    for k in range(args.steps):
        t0 = time.perf_counter()
        to_left, to_right = slab.faces()
        # What arrives from the left is that neighbour's right-going face, and
        # vice versa. Sendrecv cannot deadlock, so the pair needs no ordering.
        comm.Sendrecv(to_right, dest=right_nbr, sendtag=0,
                      recvbuf=from_left, source=left_nbr, recvtag=0)
        comm.Sendrecv(to_left, dest=left_nbr, sendtag=1,
                      recvbuf=from_right, source=right_nbr, recvtag=1)
        t1 = time.perf_counter()

        slab.fill_ghosts(from_left, from_right)
        slab.stream()
        slab.collide()
        t2 = time.perf_counter()

        t_exchange += t1 - t0
        t_compute += t2 - t1

        if args.frame_every and (k + 1) % args.frame_every == 0:
            np.save(f"{args.frames_dir}/slab_{k + 1:08d}_{rank:03d}.npy", slab.owned_temp())

    elapsed = time.perf_counter() - start
    comm.Barrier()

    results = DDict.attach(args.worker_dser)
    results[f"final/{rank}"] = {
        "rank": rank,
        "host": socket.gethostname().split(".")[0],
        "x0": slab.x0,
        "x1": slab.x1,
        "columns": slab.width,
        "elapsed": elapsed,
        "exchange": t_exchange,
        "compute": t_compute,
        "temp": slab.owned_temp(),
        "finite": bool(np.isfinite(slab.rho[slab.owned]).all()),
    }
    results.detach()

    MPI.Finalize()


# --------------------------------------------------------------------------
# Dragon driver
# --------------------------------------------------------------------------


def run_driver(args) -> None:
    import contextlib

    import dragon  # noqa: F401  (registers the 'dragon' start method)
    import multiprocessing as mp
    from dragon import DDict, Policy, ProcessGroup, ProcessTemplate, System
    from dragon.infrastructure.facts import PMIBackend
    from dragon.native.machine import Node

    from halo2 import assign_nodes
    from halo3d import placement_from_spec

    mp.set_start_method("dragon")

    params = make_params(args)
    alloc = System()
    hostnames = [Node(nid).hostname for nid in alloc.nodes]
    nnodes = alloc.nnodes

    if args.ranks_per_host:
        placement = placement_from_spec(args.ranks_per_host, hostnames)
        args.ranks = len(placement)
    else:
        placement = assign_nodes(args.ranks, hostnames, args.placement)
    seams = sum(1 for r in range(args.ranks) if placement[r] != placement[(r + 1) % args.ranks])

    cells = args.n * params.depth * args.n
    print(f"room {args.n}x{params.depth}x{args.n} = {cells / 1e6:.2f}M cells  Ra={args.ra:.0e}")
    for host in hostnames:
        print(f"  {host}: {sum(1 for h in placement if h == host)} ranks")
    print(f"ranks={args.ranks} nodes={nnodes} placement={args.placement} exchange=MPI(PMIx)  "
          f"faces: {args.ranks - seams} node-local, {seams} shared", flush=True)

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
        results = stack.enter_context(
            DDict(1, nnodes, int(args.frames_mem_gb * 1024**3), timeout=900)
        )

        worker_args = [
            os.path.abspath(__file__),
            "--worker-dser", results.serialize(),
            "--n", str(args.n), "--ny", str(args.ny), "--ra", repr(args.ra),
            "--steps", str(args.steps), "--u-ref", repr(args.u_ref),
            "--config", args.config, "--floor-profile", args.floor_profile,
            "--bump-width", repr(args.bump_width), "--seed", str(args.seed),
            "--frame-every", str(args.frame_every),
        ]
        if frames_dir:
            worker_args += ["--frames-dir", frames_dir]
        if args.float64:
            worker_args.append("--float64")

        runs: list[list] = []
        for host in placement:
            if runs and runs[-1][0] == host:
                runs[-1][1] += 1
            else:
                runs.append([host, 1])

        grp = ProcessGroup(restart=False, pmi=PMIBackend.PMIX)
        # A PMI group numbers ranks in add_process order, so these runs land
        # rank r on placement[r] and the layout matches halo3d.py's.
        for host, count in runs:
            grp.add_process(
                nproc=count,
                template=ProcessTemplate(
                    target=sys.executable,
                    args=worker_args,
                    cwd=os.getcwd(),
                    env=None,
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

        # Compare seam counts rather than host names: Node().hostname and
        # socket.gethostname() spell the same machine differently.
        actual = {r["rank"]: r["host"] for r in rows}
        actual_seams = sum(1 for r in range(args.ranks)
                           if actual[r] != actual[(r + 1) % args.ranks])
        for host in sorted(set(actual.values())):
            print(f"  {host}: {sum(1 for h in actual.values() if h == host)} ranks (actual)")
        print(f"seams: {actual_seams} actual vs {seams} intended"
              f"{'   <-- LAYOUT MISMATCH' if actual_seams != seams else ''}")

        total_ex = sum(r["exchange"] for r in rows)
        total_cp = sum(r["compute"] for r in rows)
        print(f"wall: {wall:.1f}s   exchange share: {total_ex / (total_ex + total_cp) * 100:.1f}%")
        print(f"throughput: {cells * args.steps / wall / 1e6:.2f} MLUPS")

        if args.frames_dir:
            import pathlib

            from halo3d import slab_bounds

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


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_dser:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
