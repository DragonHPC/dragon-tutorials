"""Domain-decomposed thermal LBM: one cavity split across ranks, halos via DDict.

lbm.py runs a whole cavity in one process. This splits a single cavity across
ranks and carries the halo exchange in a Dragon DDict, using checkpointing for
the per-timestep versioning. It is the tightly-coupled counterpart to sweep.py,
which is embarrassingly parallel.

Decomposition is 1D along x: each rank owns a contiguous slab of columns plus
one ghost column on each internal side. Streaming is nearest-neighbour, so a
single ghost column is sufficient and there is no global solve anywhere.

Only the populations that actually cross the boundary are exchanged: of the 9
D2Q9 directions, 3 have cx=+1 and 3 have cx=-1; of the 5 D2Q5 directions, one
each. So a face carries 4 values per cell, not 14.

The key names contain the rank but NOT the timestep. The DDict checkpoint id
supplies the version, so the same four keys are rewritten every step and the
working set holds the last few. Readers block until the neighbour has published
at their checkpoint, which is what keeps ranks in lockstep.

    dragon halo.py --ranks 4 --n 256 --ra 1e5 --steps 5000

Validate against the single-process solver:

    dragon halo.py --ranks 4 --n 64 --ra 1e4 --steps 20000 --check
"""

from __future__ import annotations

import argparse
import socket
import time

import dragon  # noqa: F401  (registers the 'dragon' start method)
import multiprocessing as mp
import numpy as np
from dragon import DDict, System

from lbm import C5, C9, OPP5, OPP9, T_COLD, T_HOT, W5, W9, BENCHMARK_NU, Cavity, Params

# Populations crossing a vertical face, by sign of cx.
RIGHT_F = np.array([i for i, c in enumerate(C9) if c[0] == 1])
LEFT_F = np.array([i for i, c in enumerate(C9) if c[0] == -1])
RIGHT_G = np.array([i for i, c in enumerate(C5) if c[0] == 1])
LEFT_G = np.array([i for i, c in enumerate(C5) if c[0] == -1])


def slab_bounds(n: int, rank: int, n_ranks: int) -> tuple[int, int]:
    base, extra = divmod(n, n_ranks)
    x0 = rank * base + min(rank, extra)
    return x0, x0 + base + (1 if rank < extra else 0)


class Slab:
    """One rank's piece of the cavity, padded with a ghost column each side.

    Index 0 and -1 along x are ghosts; owned columns are [1:-1]. Ghosts on the
    outermost ranks are never read from a neighbour, and the wrap-around that
    np.roll produces there lands only in ghost cells, which are overwritten
    before they are next read.
    """

    def __init__(self, params: Params, rank: int, n_ranks: int):
        self.p = params
        self.rank = rank
        self.n_ranks = n_ranks
        n = params.n
        self.x0, self.x1 = slab_bounds(n, rank, n_ranks)
        self.width = self.x1 - self.x0
        if self.width < 2:
            raise ValueError(f"rank {rank} owns {self.width} columns; use fewer ranks or larger n")

        self.has_hot = self.x0 == 0
        self.has_cold = self.x1 == n
        # Wrap at the outer edges rather than leaving ghosts unfilled: the serial
        # array wraps x=0 into x=n-1, and both are wall cells, so matching that
        # keeps the decomposed result identical to the serial one.
        self.left_nbr = (rank - 1) % n_ranks
        self.right_nbr = (rank + 1) % n_ranks

        shape = (self.width + 2, n)
        solid = np.zeros(shape, dtype=bool)
        solid[:, 0] = solid[:, -1] = True
        if self.has_hot:
            solid[1, :] = True
        if self.has_cold:
            solid[-2, :] = True
        solid[0, :] = solid[-1, :] = False  # ghosts are never collided
        self.solid = solid
        self.fluid = ~solid
        self.owned = np.zeros(shape, dtype=bool)
        self.owned[1:-1, :] = True

        self.rho = np.ones(shape)
        self.ux = np.zeros(shape)
        self.uy = np.zeros(shape)

        gx = np.arange(self.x0 - 1, self.x1 + 1)[:, None] / (n - 1)
        self.temp = T_HOT + (T_COLD - T_HOT) * np.broadcast_to(gx, shape).copy()

        self.f = self._feq(self.rho, self.ux, self.uy)
        self.g = self._geq(self.temp, self.ux, self.uy)
        self.f_post = self.f.copy()
        self.g_post = self.g.copy()
        self.step_count = 0

    def _feq(self, rho, ux, uy):
        cu = 3.0 * (C9[:, 0, None, None] * ux + C9[:, 1, None, None] * uy)
        usq = 1.5 * (ux**2 + uy**2)
        return W9[:, None, None] * rho * (1.0 + cu + 0.5 * cu**2 - usq)

    def _geq(self, temp, ux, uy):
        cu = 3.0 * (C5[:, 0, None, None] * ux + C5[:, 1, None, None] * uy)
        return W5[:, None, None] * temp * (1.0 + cu)

    def wall_temps(self) -> tuple[float, float]:
        if self.p.swap_period <= 0:
            return T_HOT, T_COLD
        phase = np.cos(2.0 * np.pi * self.step_count / self.p.swap_period)
        return T_HOT * phase, T_COLD * phase

    def collide(self):
        """Macroscopic update, BGK collision, and wall reflection."""
        p = self.p
        self.rho = self.f.sum(axis=0)
        force_y = p.gravity * self.temp * self.fluid
        self.ux = (self.f * C9[:, 0, None, None]).sum(axis=0) / self.rho
        self.uy = ((self.f * C9[:, 1, None, None]).sum(axis=0) + 0.5 * force_y) / self.rho
        self.temp = self.g.sum(axis=0)

        feq = self._feq(self.rho, self.ux, self.uy)
        geq = self._geq(self.temp, self.ux, self.uy)
        cy = C9[:, 1, None, None]
        src = (1.0 - 0.5 / p.tau_f) * W9[:, None, None] * 3.0 * cy * force_y

        f_post = self.f - (self.f - feq) / p.tau_f + src
        g_post = self.g - (self.g - geq) / p.tau_g

        f_post[:, self.solid] = f_post[OPP9][:, self.solid]

        t_hot, t_cold = self.wall_temps()
        g_wall = g_post[OPP5].copy()
        if self.has_hot:
            g_wall[:, 1, :] = -g_wall[:, 1, :] + 2.0 * W5[:, None] * t_hot
        if self.has_cold:
            g_wall[:, -2, :] = -g_wall[:, -2, :] + 2.0 * W5[:, None] * t_cold
        g_post[:, self.solid] = g_wall[:, self.solid]

        self.f_post, self.g_post = f_post, g_post

    def faces(self) -> tuple[np.ndarray, np.ndarray]:
        """(to_left, to_right) packed halo faces from the current post-collision state."""
        to_right = np.concatenate(
            [self.f_post[RIGHT_F, -2, :], self.g_post[RIGHT_G, -2, :]], axis=0
        )
        to_left = np.concatenate([self.f_post[LEFT_F, 1, :], self.g_post[LEFT_G, 1, :]], axis=0)
        return to_left, to_right

    def fill_ghosts(self, from_left: np.ndarray, from_right: np.ndarray):
        nf = len(RIGHT_F)
        self.f_post[RIGHT_F, 0, :] = from_left[:nf]
        self.g_post[RIGHT_G, 0, :] = from_left[nf:]
        self.f_post[LEFT_F, -1, :] = from_right[:nf]
        self.g_post[LEFT_G, -1, :] = from_right[nf:]

    def stream(self):
        for i, (cx, cy) in enumerate(C9):
            self.f[i] = np.roll(self.f_post[i], (int(cx), int(cy)), axis=(0, 1))
        for i, (cx, cy) in enumerate(C5):
            self.g[i] = np.roll(self.g_post[i], (int(cx), int(cy)), axis=(0, 1))
        self.step_count += 1

    def owned_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Owned temperature and x-velocity, excluding ghosts."""
        return self.temp[1:-1, :].copy(), self.ux[1:-1, :].copy()


def rank_worker(d: DDict, rank: int, n_ranks: int, params: Params, steps: int, report_every: int):
    d.checkpoint_id = 0
    slab = Slab(params, rank, n_ranks)

    slab.collide()
    to_left, to_right = slab.faces()
    d[f"halo/{rank}/left"] = to_left
    d[f"halo/{rank}/right"] = to_right

    t_exchange = 0.0
    t_compute = 0.0
    start = time.perf_counter()

    for k in range(steps):
        t0 = time.perf_counter()
        from_left = np.asarray(d[f"halo/{slab.left_nbr}/right"])
        from_right = np.asarray(d[f"halo/{slab.right_nbr}/left"])
        t1 = time.perf_counter()

        slab.fill_ghosts(from_left, from_right)
        slab.stream()
        slab.collide()
        t2 = time.perf_counter()

        d.checkpoint()
        to_left, to_right = slab.faces()
        d[f"halo/{rank}/left"] = to_left
        d[f"halo/{rank}/right"] = to_right
        t3 = time.perf_counter()

        t_exchange += (t1 - t0) + (t3 - t2)
        t_compute += t2 - t1

    elapsed = time.perf_counter() - start
    temp, ux = slab.owned_fields()
    d[f"final/{rank}"] = {
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
        "rho_finite": bool(np.isfinite(slab.rho[slab.owned]).all()),
    }
    d.detach()


def global_nusselt(temp: np.ndarray, ux: np.ndarray, params: Params) -> float:
    """Nusselt over the assembled field, using the same definition as lbm.Cavity."""
    solid = np.zeros(temp.shape, dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    dtdx = np.gradient(temp, axis=0)
    local = ux * temp - params.diffusivity * dtdx
    return float(local[~solid].mean() * params.length / (params.diffusivity * (T_HOT - T_COLD)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ra", type=float, default=1e5)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--u-ref", type=float, default=0.05)
    ap.add_argument("--swap-period", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=0)
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--managers-per-node", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="compare Nu against the published benchmark")
    args = ap.parse_args()

    mp.set_start_method("dragon")
    params = Params(n=args.n, ra=args.ra, u_ref=args.u_ref, swap_period=args.swap_period)
    nnodes = System().nnodes

    with DDict(
        args.managers_per_node,
        nnodes,
        nnodes * 1024 * 1024 * 1024,
        working_set_size=args.working_set,
        wait_for_keys=True,
        timeout=600,
    ) as d:
        print(
            f"cavity n={args.n} Ra={args.ra:.0e} tau_f={params.tau_f:.4f} tau_g={params.tau_g:.4f}  "
            f"ranks={args.ranks} nodes={nnodes} steps={args.steps}",
            flush=True,
        )

        procs = [
            mp.Process(target=rank_worker, args=(d, r, args.ranks, params, args.steps, args.report_every))
            for r in range(args.ranks)
        ]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0

        d.sync_to_newest_checkpoint()
        rows = [d[f"final/{r}"] for r in range(args.ranks)]

        if not all(r["rho_finite"] for r in rows):
            print("DIVERGED: non-finite density", flush=True)

        hosts = sorted({r["host"] for r in rows})
        total_ex = sum(r["exchange"] for r in rows)
        total_cp = sum(r["compute"] for r in rows)

        temp = np.zeros((args.n, args.n))
        ux = np.zeros((args.n, args.n))
        for r in rows:
            temp[r["x0"] : r["x1"], :] = r["temp"]
            ux[r["x0"] : r["x1"], :] = r["ux"]
        nu = global_nusselt(temp, ux, params)

        print(f"hosts: {', '.join(hosts)}")
        for r in rows:
            share = r["exchange"] / r["elapsed"] * 100.0
            print(
                f"  rank {r['rank']:>2} on {r['host']:<12} cols={r['columns']:>4}  "
                f"{r['elapsed']:7.1f}s  exchange={r['exchange']:7.1f}s ({share:4.1f}%)  "
                f"compute={r['compute']:7.1f}s"
            )
        print(f"wall: {wall:.1f}s   exchange share: {total_ex / (total_ex + total_cp) * 100:.1f}%")
        mlups = args.n * args.n * args.steps / wall / 1e6
        print(f"throughput: {mlups:.1f} MLUPS")
        print(f"Nu = {nu:.4f}")
        if args.check:
            ref = BENCHMARK_NU.get(args.ra)
            if ref:
                print(f"benchmark Nu = {ref}  error = {abs(nu - ref) / ref * 100:.2f}%")
            serial = Cavity(params)
            serial.run(args.steps)
            dt = np.abs(temp - serial.temp).max()
            du = np.abs(ux - serial.ux).max()
            print(f"serial Nu = {serial.nusselt():.4f}")
            print(f"vs serial: max|dT| = {dt:.3e}  max|du_x| = {du:.3e}")


if __name__ == "__main__":
    main()
