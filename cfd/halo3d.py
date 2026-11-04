"""Domain-decomposed 3D cavity: one room split across ranks, halos via DDict.

The 3D counterpart of halo3.py. lbm3d.py runs a whole room in one process at
about 5 MLUPS, so 128 cubed for 20k steps is a couple of hours; that is the
resolution the volume renderer wants and the reason this exists.

Decomposition is 1D along x, so each halo face is a 2D plane rather than a
line. Of the 19 D3Q19 directions 5 have cx=+1 and 5 have cx=-1; D3Q7 adds one
each way. A face therefore carries 6 values per cell, which at n=128 is about
400 KB - fifty times the 2D halo, still far below what the link can carry.

Dictionary routing follows halo3.py, which measured 1.4-3.5x faster than
routing every face through one cluster-wide DDict: each node gets its own
single-manager dictionary for the faces that stay on it, and a small shared
dictionary carries the block seams.

    dragon --hostfile hosts halo3d.py --ranks 16 --n 128 --config rb --floor-profile bump
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
from halo3 import node_local_ddict
from lbm3d import C7, C19, OPP7, OPP19, T_COLD, T_HOT, W7, W19, Params3D

RIGHT_F = np.array([i for i, c in enumerate(C19) if c[0] == 1])
LEFT_F = np.array([i for i, c in enumerate(C19) if c[0] == -1])
RIGHT_G = np.array([i for i, c in enumerate(C7) if c[0] == 1])
LEFT_G = np.array([i for i, c in enumerate(C7) if c[0] == -1])


def slab_bounds(n: int, rank: int, n_ranks: int) -> tuple[int, int]:
    base, extra = divmod(n, n_ranks)
    x0 = rank * base + min(rank, extra)
    return x0, x0 + base + (1 if rank < extra else 0)


class Slab3D:
    """One rank's slab of the room, padded with a ghost plane on each x face."""

    def __init__(self, params: Params3D, rank: int, n_ranks: int, dtype=np.float32):
        self.p = params
        self.dtype = dtype
        self.rank = rank
        self.n_ranks = n_ranks
        n, ny = params.n, params.depth
        self.x0, self.x1 = slab_bounds(n, rank, n_ranks)
        self.width = self.x1 - self.x0
        if self.width < 2:
            raise ValueError(f"rank {rank} owns {self.width} columns; use fewer ranks or larger n")

        # Periodic in x, as in halo.py: leaving the outermost ghosts unfilled
        # lets np.roll wrap junk into the wall cells.
        self.left_nbr = (rank - 1) % n_ranks
        self.right_nbr = (rank + 1) % n_ranks

        shape = (self.width + 2, ny, n)
        solid = np.zeros(shape, dtype=bool)
        solid[:, :, 0] = solid[:, :, -1] = True  # floor and ceiling
        solid[:, 0, :] = solid[:, -1, :] = True  # depth walls
        if self.x0 == 0:
            solid[1, :, :] = True
        if self.x1 == n:
            solid[-2, :, :] = True
        solid[0, :, :] = solid[-1, :, :] = False  # ghosts are never collided
        self.solid = solid
        self.fluid = ~solid
        self.owned = np.zeros(shape, dtype=bool)
        self.owned[1:-1, :, :] = True

        self.rho = np.ones(shape, dtype=dtype)
        self.ux = np.zeros(shape, dtype=dtype)
        self.uy = np.zeros(shape, dtype=dtype)
        self.uz = np.zeros(shape, dtype=dtype)

        z = np.arange(n, dtype=dtype)[None, None, :] / (n - 1)
        if params.config == "rb":
            self.temp = (T_HOT + (T_COLD - T_HOT) * np.broadcast_to(z, shape)).astype(dtype)
            # Draw the seed noise for the whole room and slice, so the initial
            # state does not depend on how many ranks it was cut into.
            rng = np.random.default_rng(params.seed)
            noise = rng.normal(0.0, 1e-3, (n, ny, n)).astype(dtype)
            idx = np.arange(self.x0 - 1, self.x1 + 1) % n
            self.temp = self.temp + noise[idx] * self.fluid
        else:
            # Side-heated: the gradient runs across x instead, and the
            # conduction state is stable, so it needs no seed to start.
            xs = np.arange(self.x0 - 1, self.x1 + 1, dtype=dtype) / (n - 1)
            self.temp = np.broadcast_to(
                (T_HOT + (T_COLD - T_HOT) * xs)[:, None, None], shape
            ).astype(dtype)
        self.t_floor = self._floor_slice()

        self.f = self._feq(self.rho, self.ux, self.uy, self.uz)
        self.g = self._geq(self.temp, self.ux, self.uy, self.uz)
        self.f_post = self.f.copy()
        self.g_post = self.g.copy()
        self.step_count = 0

    def _floor_slice(self) -> np.ndarray:
        """This slab's portion of the floor temperature, including ghost planes."""
        n, ny = self.p.n, self.p.depth
        if self.p.floor_profile != "bump":
            return np.full((self.width + 2, ny), T_HOT, dtype=self.dtype)
        xs = np.arange(self.x0 - 1, self.x1 + 1)
        x = (xs - (n - 1) / 2) / ((n - 1) / 2)
        y = (np.arange(ny) - (ny - 1) / 2) / ((ny - 1) / 2)
        r2 = x[:, None] ** 2 + y[None, :] ** 2
        bump = np.exp(-r2 / (2.0 * self.p.bump_width**2))
        return (T_COLD + (T_HOT - T_COLD) * bump).astype(self.dtype)

    def _feq(self, rho, ux, uy, uz):
        cu = 3.0 * (
            C19[:, 0, None, None, None] * ux
            + C19[:, 1, None, None, None] * uy
            + C19[:, 2, None, None, None] * uz
        )
        usq = 1.5 * (ux**2 + uy**2 + uz**2)
        return (W19[:, None, None, None] * rho * (1.0 + cu + 0.5 * cu**2 - usq)).astype(self.dtype)

    def _geq(self, temp, ux, uy, uz):
        cu = 4.0 * (
            C7[:, 0, None, None, None] * ux
            + C7[:, 1, None, None, None] * uy
            + C7[:, 2, None, None, None] * uz
        )
        usq = 2.0 * (ux**2 + uy**2 + uz**2)
        return (W7[:, None, None, None] * temp * (1.0 + cu + 0.5 * cu**2 - usq)).astype(self.dtype)

    def collide(self):
        p = self.p
        self.rho = self.f.sum(axis=0)
        force_z = p.gravity * self.temp * self.fluid
        self.ux = (self.f * C19[:, 0, None, None, None]).sum(axis=0) / self.rho
        self.uy = (self.f * C19[:, 1, None, None, None]).sum(axis=0) / self.rho
        self.uz = ((self.f * C19[:, 2, None, None, None]).sum(axis=0) + 0.5 * force_z) / self.rho
        self.temp = self.g.sum(axis=0)

        feq = self._feq(self.rho, self.ux, self.uy, self.uz)
        geq = self._geq(self.temp, self.ux, self.uy, self.uz)
        cz = C19[:, 2, None, None, None]
        src = (1.0 - 0.5 / p.tau_f) * W19[:, None, None, None] * 3.0 * cz * force_z

        f_post = self.f - (self.f - feq) / p.tau_f + src
        g_post = self.g - (self.g - geq) / p.tau_g

        f_post[:, self.solid] = f_post[OPP19][:, self.solid]

        g_wall = g_post[OPP7].copy()
        if p.config == "rb":
            g_wall[:, :, :, 0] = -g_wall[:, :, :, 0] + 2.0 * W7[:, None, None] * self.t_floor[None, :, :]
            g_wall[:, :, :, -1] = -g_wall[:, :, :, -1] + 2.0 * W7[:, None, None] * T_COLD
        else:
            # The heated walls lie on the decomposition axis, so only the two
            # end ranks carry them; everyone else sees plain bounce-back.
            if self.x0 == 0:
                g_wall[:, 1, :, :] = -g_wall[:, 1, :, :] + 2.0 * W7[:, None, None] * T_HOT
            if self.x1 == p.n:
                g_wall[:, -2, :, :] = -g_wall[:, -2, :, :] + 2.0 * W7[:, None, None] * T_COLD
        g_post[:, self.solid] = g_wall[:, self.solid]

        self.f_post, self.g_post = f_post, g_post

    def faces(self) -> tuple[np.ndarray, np.ndarray]:
        to_right = np.concatenate(
            [self.f_post[RIGHT_F, -2, :, :], self.g_post[RIGHT_G, -2, :, :]], axis=0
        )
        to_left = np.concatenate(
            [self.f_post[LEFT_F, 1, :, :], self.g_post[LEFT_G, 1, :, :]], axis=0
        )
        return to_left, to_right

    def fill_ghosts(self, from_left: np.ndarray, from_right: np.ndarray):
        nf = len(RIGHT_F)
        self.f_post[RIGHT_F, 0, :, :] = from_left[:nf]
        self.g_post[RIGHT_G, 0, :, :] = from_left[nf:]
        self.f_post[LEFT_F, -1, :, :] = from_right[:nf]
        self.g_post[LEFT_G, -1, :, :] = from_right[nf:]

    def stream(self):
        for i, c in enumerate(C19):
            self.f[i] = np.roll(self.f_post[i], (int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))
        for i, c in enumerate(C7):
            self.g[i] = np.roll(self.g_post[i], (int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))
        self.step_count += 1

    def owned_temp(self) -> np.ndarray:
        return self.temp[1:-1, :, :].copy()


def rank_worker(shared, local_dicts, results, placement, rank, n_ranks, params, steps, frame_every, dtype_name, frames_dir):
    host = placement[rank]
    slab = Slab3D(params, rank, n_ranks, dtype=np.dtype(dtype_name))

    left_d = local_dicts[host] if placement[slab.left_nbr] == host else shared
    right_d = local_dicts[host] if placement[slab.right_nbr] == host else shared
    used = [left_d] + ([right_d] if right_d is not left_d else [])
    for d in used:
        d.checkpoint_id = 0

    slab.collide()
    to_left, to_right = slab.faces()
    left_d[f"halo/{rank}/left"] = to_left
    right_d[f"halo/{rank}/right"] = to_right

    t_exchange = t_compute = 0.0
    start = time.perf_counter()

    for k in range(steps):
        t0 = time.perf_counter()
        from_left = np.asarray(left_d[f"halo/{slab.left_nbr}/right"])
        from_right = np.asarray(right_d[f"halo/{slab.right_nbr}/left"])
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

        if frame_every and (k + 1) % frame_every == 0:
            # Each rank writes its own slab as it goes. Holding frames in memory
            # until the end loses the whole run if anything dies late, and the
            # write is a fraction of a second against minutes of compute.
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
        "temp": slab.owned_temp(),
        "finite": bool(np.isfinite(slab.rho[slab.owned]).all()),
    }


def placement_from_spec(spec: str, hostnames: list[str]) -> list[str]:
    """Expand 'studio=20,max=12' into a per-rank host list.

    Matched by name rather than position: System().nodes does not order hosts
    by capability, so a positional split can hand the most ranks to the
    smallest machine. Under lockstep that would gate every rank.
    """
    out: list[str] = []
    for item in spec.split(","):
        key, _, count = item.partition("=")
        matches = [h for h in hostnames if key.strip().lower() in h.lower()]
        if len(matches) != 1:
            raise SystemExit(f"'{key}' matched {len(matches)} of {hostnames}; need exactly one")
        out.extend([matches[0]] * int(count))
    return out


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
    ap.add_argument("--frames-mem-gb", type=float, default=12.0, help="total memory for the frame store")
    ap.add_argument("--working-set", type=int, default=4)
    ap.add_argument("--frames-dir")
    ap.add_argument("--frame-every", type=int, default=0)
    ap.add_argument("--keep-slabs", action="store_true", help="do not delete per-rank slabs after assembly")
    ap.add_argument("--float64", action="store_true")
    args = ap.parse_args()

    mp.set_start_method("dragon")
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
    seams = sum(1 for r in range(args.ranks) if placement[r] != placement[(r + 1) % args.ranks])

    cells = args.n * params.depth * args.n
    print(f"room {args.n}x{params.depth}x{args.n} = {cells / 1e6:.2f}M cells  Ra={args.ra:.0e}")
    for host in hostnames:
        print(f"  {host}: {sum(1 for h in placement if h == host)} ranks")
    print(f"ranks={args.ranks} nodes={nnodes} placement={args.placement}  "
          f"faces: {args.ranks - seams} node-local, {seams} shared", flush=True)

    frames_dir = ""
    if args.frames_dir:
        import pathlib

        out = pathlib.Path(args.frames_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Workers elsewhere may run under a different uid on the shared mount, so
        # widen the mode. Best effort only: the driver can land on a node that
        # does not own the directory, or that holds a stale NFS handle for it.
        try:
            out.chmod(0o777)
        except OSError:
            pass
        frames_dir = str(out)

    with contextlib.ExitStack() as stack:
        local_dicts = {
            host: stack.enter_context(node_local_ddict(host, 4 * 1024**3, 1, args.working_set))
            for host in hostnames
        }
        shared = stack.enter_context(
            DDict(1, nnodes, nnodes * 2 * 1024**3, working_set_size=args.working_set,
                  wait_for_keys=True, timeout=900)
        )
        results = stack.enter_context(DDict(1, nnodes, int(args.frames_mem_gb * 1024**3), timeout=900))

        grp = ProcessGroup(restart=False)
        for rank, host in enumerate(placement):
            grp.add_process(
                nproc=1,
                template=ProcessTemplate(
                    target=rank_worker,
                    args=(shared, local_dicts, results, placement, rank, args.ranks,
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

        if args.frames_dir:
            import pathlib

            out = pathlib.Path(frames_dir)
            # Bounds are recomputed rather than read from the results, so a
            # directory of slabs from an interrupted run can still be assembled.
            steps_saved = sorted({int(p.stem.split("_")[1]) for p in out.glob("slab_*.npy")})
            written = 0
            for s in steps_saved:
                slabs = sorted(out.glob(f"slab_{s:08d}_*.npy"))
                if len(slabs) != args.ranks:  # a partial frame from an interrupted run
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
