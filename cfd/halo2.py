"""halo.py with explicit rank placement via ProcessGroup, ProcessTemplate and Policy.

halo.py creates plain mp.Process objects and lets Dragon place them. With a 1D
decomposition that is the worst case: if consecutive ranks land on alternating
nodes, every single internal boundary becomes a cross-node exchange, and the
probe puts that at 1609 us versus 412 us on-node.

This version assigns ranks to nodes deliberately. Under block placement a
contiguous run of ranks lives on each node, so only the seams between blocks
cross the network:

    16 ranks, 2 nodes, cyclic:  16 of 16 boundaries cross the network
    16 ranks, 2 nodes, block:    2 of 16

Everything else - the solver, the exchange pattern, the checkpointing - is
imported unchanged from halo.py, so any difference in runtime is placement
and nothing else.

    dragon --hostfile hosts halo2.py --ranks 16 --placement block
    dragon --hostfile hosts halo2.py --ranks 16 --placement cyclic
"""

from __future__ import annotations

import argparse
import time

import dragon  # noqa: F401  (registers the 'dragon' start method)
import multiprocessing as mp
import numpy as np
from dragon import DDict, Policy, ProcessGroup, ProcessTemplate, System
from dragon.native.machine import Node

from halo import Slab, global_nusselt, rank_worker  # noqa: F401  (Slab used by rank_worker)
from lbm import BENCHMARK_NU, Cavity, Params


def assign_nodes(n_ranks: int, hostnames: list[str], strategy: str) -> list[str]:
    """Map each rank to a hostname.

    block  keeps neighbouring ranks together, so only block seams cross nodes.
    cyclic spreads neighbours apart, which is what placement does by default.
    """
    n_nodes = len(hostnames)
    if strategy == "cyclic":
        return [hostnames[r % n_nodes] for r in range(n_ranks)]
    return [hostnames[r * n_nodes // n_ranks] for r in range(n_ranks)]


def count_remote_faces(placement: list[str]) -> int:
    """Adjacent rank pairs that do not share a node, counting the periodic wrap."""
    n = len(placement)
    return sum(1 for r in range(n) if placement[r] != placement[(r + 1) % n])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ra", type=float, default=1e6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--u-ref", type=float, default=0.05)
    ap.add_argument("--swap-period", type=int, default=0)
    ap.add_argument("--placement", choices=["block", "cyclic"], default="block")
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--managers-per-node", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    mp.set_start_method("dragon")
    params = Params(n=args.n, ra=args.ra, u_ref=args.u_ref, swap_period=args.swap_period)

    alloc = System()
    hostnames = [Node(nid).hostname for nid in alloc.nodes]
    nnodes = alloc.nnodes
    placement = assign_nodes(args.ranks, hostnames, args.placement)
    remote = count_remote_faces(placement)

    with DDict(
        args.managers_per_node,
        nnodes,
        nnodes * 1024 * 1024 * 1024,
        working_set_size=args.working_set,
        wait_for_keys=True,
        timeout=600,
    ) as d:
        print(
            f"cavity n={args.n} Ra={args.ra:.0e} tau_f={params.tau_f:.4f}  "
            f"ranks={args.ranks} nodes={nnodes} steps={args.steps}",
            flush=True,
        )
        print(f"placement={args.placement}  remote faces: {remote} of {args.ranks}", flush=True)
        for host in hostnames:
            owned = [r for r, h in enumerate(placement) if h == host]
            print(f"  {host}: ranks {owned[0]}-{owned[-1]} ({len(owned)})" if owned else f"  {host}: none")

        grp = ProcessGroup(restart=False)
        for rank, host in enumerate(placement):
            grp.add_process(
                nproc=1,
                template=ProcessTemplate(
                    target=rank_worker,
                    args=(d, rank, args.ranks, params, args.steps, 0),
                    policy=Policy(placement=Policy.Placement.HOST_NAME, host_name=host),
                ),
            )

        t0 = time.perf_counter()
        grp.init()
        grp.start()
        grp.join()
        grp.close()
        wall = time.perf_counter() - t0

        d.sync_to_newest_checkpoint()
        rows = [d[f"final/{r}"] for r in range(args.ranks)]

        if not all(r["rho_finite"] for r in rows):
            print("DIVERGED: non-finite density", flush=True)

        actual = [r["host"] for r in rows]
        if count_remote_faces(actual) != remote:
            print(f"WARNING: requested {remote} remote faces but got {count_remote_faces(actual)}")

        temp = np.zeros((args.n, args.n))
        ux = np.zeros((args.n, args.n))
        for r in rows:
            temp[r["x0"] : r["x1"], :] = r["temp"]
            ux[r["x0"] : r["x1"], :] = r["ux"]
        nu = global_nusselt(temp, ux, params)

        total_ex = sum(r["exchange"] for r in rows)
        total_cp = sum(r["compute"] for r in rows)
        for r in rows:
            share = r["exchange"] / r["elapsed"] * 100.0
            print(
                f"  rank {r['rank']:>2} on {r['host']:<12} cols={r['columns']:>4}  "
                f"{r['elapsed']:7.1f}s  exchange={r['exchange']:7.1f}s ({share:4.1f}%)"
            )
        print(f"wall: {wall:.1f}s   exchange share: {total_ex / (total_ex + total_cp) * 100:.1f}%")
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
