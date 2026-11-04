"""Measure DDict halo-exchange cost per timestep, with no simulation attached.

The decomposed solver in halo.py pays one DDict round trip per rank per
timestep. At 20k-50k timesteps that cost is the entire question: the halo for
n=512 is only 8 KB, so bandwidth is irrelevant and latency decides whether
domain decomposition is worth doing at all. This measures it in isolation.

Exchange pattern is the one from examples/dragon_data/ddict/ddict_smoothing.py:
read neighbours at checkpoint N, compute, checkpoint(), publish at N+1.

    dragon probe.py --ranks 2 --steps 2000 --halo-bytes 8192
"""

from __future__ import annotations

import argparse
import os
import socket
import time

import dragon  # noqa: F401  (registers the 'dragon' start method)
import multiprocessing as mp
import numpy as np
from dragon import DDict, System


def exchange_worker(d: DDict, rank: int, n_ranks: int, steps: int, halo: int, compute_us: float):
    d.checkpoint_id = 0
    payload = np.zeros(halo // 4, dtype=np.float32)

    left_nbr = rank - 1 if rank > 0 else None
    right_nbr = rank + 1 if rank < n_ranks - 1 else None

    d[f"halo/{rank}/left"] = payload
    d[f"halo/{rank}/right"] = payload

    # Spin for the same wall time a real interior update would take, so the
    # measurement shows whether the exchange hides behind compute or not.
    def burn():
        if compute_us <= 0:
            return
        end = time.perf_counter() + compute_us * 1e-6
        while time.perf_counter() < end:
            pass

    t_wait = 0.0
    t_put = 0.0
    start = time.perf_counter()

    for _ in range(steps):
        t0 = time.perf_counter()
        if right_nbr is not None:
            _ = np.asarray(d[f"halo/{right_nbr}/left"])
        if left_nbr is not None:
            _ = np.asarray(d[f"halo/{left_nbr}/right"])
        t1 = time.perf_counter()

        burn()

        d.checkpoint()
        d[f"halo/{rank}/left"] = payload
        d[f"halo/{rank}/right"] = payload
        t2 = time.perf_counter()

        t_wait += t1 - t0
        t_put += t2 - t1

    elapsed = time.perf_counter() - start
    host = socket.gethostname().split(".")[0]
    d[f"result/{rank}"] = {
        "rank": rank,
        "host": host,
        "pid": os.getpid(),
        "elapsed": elapsed,
        "wait": t_wait,
        "put": t_put - (steps * compute_us * 1e-6),
        "per_step_us": elapsed / steps * 1e6,
    }
    d.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--halo-bytes", type=int, default=8192, help="per halo face; n=512 is 8192")
    ap.add_argument("--compute-us", type=float, default=0.0, help="simulated interior work per step")
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--managers-per-node", type=int, default=1)
    args = ap.parse_args()

    mp.set_start_method("dragon")
    nnodes = System().nnodes

    with DDict(
        args.managers_per_node,
        nnodes,
        nnodes * 1024 * 1024 * 1024,
        working_set_size=args.working_set,
        wait_for_keys=True,
        timeout=300,
    ) as d:
        procs = [
            mp.Process(
                target=exchange_worker,
                args=(d, r, args.ranks, args.steps, args.halo_bytes, args.compute_us),
            )
            for r in range(args.ranks)
        ]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0

        d.sync_to_newest_checkpoint()
        rows = [d[f"result/{r}"] for r in range(args.ranks)]

        hosts = sorted({r["host"] for r in rows})
        print(f"nodes={nnodes} ranks={args.ranks} steps={args.steps} halo={args.halo_bytes}B")
        print(f"placement: {len(hosts)} distinct host(s): {', '.join(hosts)}")
        if len(hosts) < 2 and nnodes > 1:
            print("WARNING: all ranks landed on one node; this is NOT a cross-node measurement")
        print(f"wall: {wall:.2f}s")
        for r in rows:
            print(
                f"  rank {r['rank']} on {r['host']:<12} "
                f"{r['per_step_us']:8.1f} us/step   "
                f"wait={r['wait']:7.2f}s put={r['put']:7.2f}s"
            )
        slowest = max(r["per_step_us"] for r in rows)
        print(f"per-step exchange cost: {slowest:.1f} us")
        for steps, label in ((20000, "20k"), (51000, "51k")):
            print(f"  -> {slowest * steps * 1e-6:7.1f}s of pure exchange over {label} steps")


if __name__ == "__main__":
    main()
