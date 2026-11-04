"""halo2.py, but with the halo traffic split across node-local dictionaries.

halo2.py places ranks carefully and gains almost nothing, because rank placement
does not decide where a key lives. A DDict spread over two nodes hashes keys to
managers on both, so roughly half of every rank's halos cross the network no
matter which node the rank runs on.

So place the data instead of the ranks. Each node gets its own single-manager
DDict holding the faces exchanged between ranks on that node, and one small
shared DDict carries the few faces that genuinely span nodes, plus the results.
With block placement over 2 nodes only the block seams are shared:

    16 ranks, 2 nodes:  14 faces node-local, 2 shared

Routing is symmetric without any negotiation: rank r-1 picks its right-hand
dictionary with the same predicate rank r uses for its left-hand one, so both
ends of a face always agree.

    dragon --hostfile hosts halo3.py --ranks 16
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

from halo import Slab, global_nusselt
from halo2 import assign_nodes
from lbm import BENCHMARK_NU, Cavity, Params


def rank_worker(shared, local_dicts, results, placement, rank, n_ranks, params, steps):
    """One slab, reading and writing each face through whichever dict is nearest."""
    host = placement[rank]
    left_nbr = (rank - 1) % n_ranks
    right_nbr = (rank + 1) % n_ranks

    left_d = local_dicts[host] if placement[left_nbr] == host else shared
    right_d = local_dicts[host] if placement[right_nbr] == host else shared

    # Deduplicate by identity; DDict handles are not hashable.
    used = [left_d]
    if right_d is not left_d:
        used.append(right_d)
    for d in used:
        d.checkpoint_id = 0

    slab = Slab(params, rank, n_ranks)
    slab.collide()
    to_left, to_right = slab.faces()
    left_d[f"halo/{rank}/left"] = to_left
    right_d[f"halo/{rank}/right"] = to_right

    t_exchange = 0.0
    t_compute = 0.0
    start = time.perf_counter()

    for _ in range(steps):
        t0 = time.perf_counter()
        from_left = np.asarray(left_d[f"halo/{left_nbr}/right"])
        from_right = np.asarray(right_d[f"halo/{right_nbr}/left"])
        t1 = time.perf_counter()

        slab.fill_ghosts(from_left, from_right)
        slab.stream()
        slab.collide()
        t2 = time.perf_counter()

        for d in used:
            d.checkpoint()
        to_left, to_right = slab.faces()
        left_d[f"halo/{rank}/left"] = to_left
        right_d[f"halo/{rank}/right"] = to_right
        t3 = time.perf_counter()

        t_exchange += (t1 - t0) + (t3 - t2)
        t_compute += t2 - t1

    elapsed = time.perf_counter() - start
    temp, ux = slab.owned_fields()
    # Results go to an uncheckpointed dict. A rank whose faces are all node-local
    # never advances its handle on the shared dict, so a put there would target a
    # checkpoint the seam ranks retired long ago.
    results[f"final/{rank}"] = {
        "rank": rank,
        "host": socket.gethostname().split(".")[0],
        "columns": slab.width,
        "x0": slab.x0,
        "x1": slab.x1,
        "elapsed": elapsed,
        "exchange": t_exchange,
        "compute": t_compute,
        "temp": temp,
        "ux": ux,
        "shared_faces": int(left_d is shared) + int(right_d is shared),
        "rho_finite": bool(np.isfinite(slab.rho[slab.owned]).all()),
    }
    for d in used:
        d.detach()
    if all(d is not shared for d in used):
        shared.detach()
    results.detach()


def node_local_ddict(host: str, total_mem: int, managers: int, working_set: int) -> DDict:
    policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=host)
    return DDict(
        None,
        None,
        total_mem,
        managers_per_policy=managers,
        policy=[policy],
        working_set_size=working_set,
        wait_for_keys=True,
        timeout=600,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ra", type=float, default=1e6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--u-ref", type=float, default=0.05)
    ap.add_argument("--swap-period", type=int, default=0)
    ap.add_argument("--placement", choices=["block", "cyclic"], default="block")
    ap.add_argument("--local-managers", type=int, default=1)
    ap.add_argument("--shared-managers", type=int, default=2)
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    mp.set_start_method("dragon")
    params = Params(n=args.n, ra=args.ra, u_ref=args.u_ref, swap_period=args.swap_period)

    alloc = System()
    hostnames = [Node(nid).hostname for nid in alloc.nodes]
    nnodes = alloc.nnodes
    placement = assign_nodes(args.ranks, hostnames, args.placement)

    shared_faces = sum(
        1 for r in range(args.ranks) if placement[r] != placement[(r + 1) % args.ranks]
    )

    with contextlib.ExitStack() as stack:
        local_dicts = {
            host: stack.enter_context(
                node_local_ddict(host, 1024 * 1024 * 1024, args.local_managers, args.working_set)
            )
            for host in hostnames
        }
        shared = stack.enter_context(
            DDict(
                args.shared_managers,
                nnodes,
                nnodes * 512 * 1024 * 1024,
                working_set_size=args.working_set,
                wait_for_keys=True,
                timeout=600,
            )
        )
        results = stack.enter_context(
            DDict(1, nnodes, nnodes * 512 * 1024 * 1024, timeout=600)
        )

        print(
            f"cavity n={args.n} Ra={args.ra:.0e} ranks={args.ranks} nodes={nnodes} steps={args.steps}",
            flush=True,
        )
        print(
            f"placement={args.placement}  faces: {args.ranks - shared_faces} node-local, "
            f"{shared_faces} shared  "
            f"managers: {args.local_managers}/node local, {args.shared_managers}/node shared",
            flush=True,
        )

        grp = ProcessGroup(restart=False)
        for rank, host in enumerate(placement):
            grp.add_process(
                nproc=1,
                template=ProcessTemplate(
                    target=rank_worker,
                    args=(shared, local_dicts, results, placement, rank, args.ranks, params, args.steps),
                    policy=Policy(placement=Policy.Placement.HOST_NAME, host_name=host),
                ),
            )

        t0 = time.perf_counter()
        grp.init()
        grp.start()
        grp.join()
        grp.close()
        wall = time.perf_counter() - t0

        shared.sync_to_newest_checkpoint()
        rows = [results[f"final/{r}"] for r in range(args.ranks)]

        if not all(r["rho_finite"] for r in rows):
            print("DIVERGED: non-finite density", flush=True)

        temp = np.zeros((args.n, args.n))
        ux = np.zeros((args.n, args.n))
        for r in rows:
            temp[r["x0"] : r["x1"], :] = r["temp"]
            ux[r["x0"] : r["x1"], :] = r["ux"]
        nu = global_nusselt(temp, ux, params)

        total_ex = sum(r["exchange"] for r in rows)
        total_cp = sum(r["compute"] for r in rows)
        local_only = [r for r in rows if r["shared_faces"] == 0]
        seam = [r for r in rows if r["shared_faces"] > 0]

        print(f"wall: {wall:.1f}s   exchange share: {total_ex / (total_ex + total_cp) * 100:.1f}%")
        if local_only:
            avg = sum(r["exchange"] for r in local_only) / len(local_only)
            print(f"  node-local ranks ({len(local_only)}): mean exchange {avg:6.1f}s")
        if seam:
            avg = sum(r["exchange"] for r in seam) / len(seam)
            print(f"  seam ranks       ({len(seam)}): mean exchange {avg:6.1f}s")
        print(f"throughput: {args.n * args.n * args.steps / wall / 1e6:.1f} MLUPS")
        print(f"Nu = {nu:.4f}")
        if args.check:
            ref = BENCHMARK_NU.get(args.ra)
            if ref:
                print(f"benchmark Nu = {ref}  error = {abs(nu - ref) / ref * 100:.2f}%")
            serial = Cavity(params)
            serial.run(args.steps)
            print(f"vs serial: max|dT| = {np.abs(temp - serial.temp).max():.3e}")


if __name__ == "__main__":
    main()
