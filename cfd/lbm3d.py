"""Thermal lattice Boltzmann solver for a 3D differentially heated cavity.

The 3D counterpart of lbm.py. A box of fluid with a hot wall at x=0, a cold
wall at x=n-1, gravity along -z, and no-slip adiabatic walls everywhere else.
D3Q19 for the flow, D3Q7 for temperature, Boussinesq buoyancy coupling them.

Axes are (x, y, z): x spans hot to cold, z is vertical, y is the depth of the
room. This is the field the volume renderer projects through, so it exists for
a visual reason as much as a physical one - the 2D solver has no interior to
look into.

Validation here is the 2D limit rather than a published 3D benchmark: with
--periodic-y and a shallow box the solution becomes y-invariant and must
reproduce lbm.py's Nusselt number, which is itself validated to under 1%
against de Vahl Davis. That exercises the lattice, the collision and the x/z
boundary conditions without relying on numbers I cannot check.

    python lbm3d.py --ra 1e5 --n 64 --steps 20000
    python lbm3d.py --ra 1e4 --n 48 --ny 4 --periodic-y --check-2d
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import time
from dataclasses import dataclass

import numpy as np

T_HOT = 0.5
T_COLD = -0.5


def _build_d3q19():
    """Rest vector, then the 6 face neighbours, then the 12 edge neighbours."""
    face = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edge = [
        v
        for v in itertools.product((-1, 0, 1), repeat=3)
        if sum(abs(c) for c in v) == 2
    ]
    vectors = [(0, 0, 0)] + face + edge
    weights = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
    return np.array(vectors, dtype=np.int32), np.array(weights)


C19, W19 = _build_d3q19()

# D3Q7 for temperature: rest plus the 6 face neighbours. Its second moment gives
# c_s^2 = 1/4, not the 1/3 of D3Q19, which is why tau_g below is 4*alpha + 0.5.
C7 = np.array(
    [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int32,
)
W7 = np.array([1 / 4] + [1 / 8] * 6)


def _opposites(vectors: np.ndarray) -> np.ndarray:
    lookup = {tuple(v): i for i, v in enumerate(vectors)}
    return np.array([lookup[tuple(-v)] for v in vectors])


OPP19 = _opposites(C19)
OPP7 = _opposites(C7)


@dataclass
class Params3D:
    n: int = 64  # extent in x and z
    ny: int = 0  # depth; 0 means a cube
    ra: float = 1e5
    pr: float = 0.71
    u_ref: float = 0.05
    swap_period: int = 0
    periodic_y: bool = False
    config: str = "side"  # 'side' heats a wall, 'rb' heats the floor
    floor_profile: str = "uniform"  # 'bump' concentrates the heat in the middle
    bump_width: float = 0.4
    seed: int = 0

    @property
    def depth(self) -> int:
        return self.ny if self.ny else self.n

    @property
    def length(self) -> int:
        return self.n - 2

    @property
    def viscosity(self) -> float:
        return self.u_ref * self.length * np.sqrt(self.pr / self.ra)

    @property
    def diffusivity(self) -> float:
        return self.u_ref * self.length / np.sqrt(self.pr * self.ra)

    @property
    def gravity(self) -> float:
        return self.u_ref**2 / self.length

    @property
    def tau_f(self) -> float:
        return 3.0 * self.viscosity + 0.5

    @property
    def tau_g(self) -> float:
        return 4.0 * self.diffusivity + 0.5


class Cavity3D:
    def __init__(self, params: Params3D, dtype=np.float64):
        self.p = params
        self.dtype = dtype
        n, ny = params.n, params.depth
        shape = (n, ny, n)

        self.solid = np.zeros(shape, dtype=bool)
        self.solid[0, :, :] = self.solid[-1, :, :] = True
        self.solid[:, :, 0] = self.solid[:, :, -1] = True
        if not params.periodic_y:
            self.solid[:, 0, :] = self.solid[:, -1, :] = True
        self.fluid = ~self.solid

        self.rho = np.ones(shape, dtype=dtype)
        self.ux = np.zeros(shape, dtype=dtype)
        self.uy = np.zeros(shape, dtype=dtype)
        self.uz = np.zeros(shape, dtype=dtype)

        if params.config == "rb":
            self.t_floor = self._floor_temperature()
            # Hot floor, cold ceiling. The conduction state is an unstable
            # equilibrium, so it needs a seed or nothing ever lifts off.
            z = np.arange(n, dtype=dtype)[None, None, :] / (n - 1)
            self.temp = (T_HOT + (T_COLD - T_HOT) * np.broadcast_to(z, shape)).astype(dtype)
            rng = np.random.default_rng(params.seed)
            self.temp = self.temp + rng.normal(0.0, 1e-3, shape).astype(dtype) * self.fluid
        else:
            self.t_floor = None
            x = np.arange(n, dtype=dtype)[:, None, None] / (n - 1)
            self.temp = (T_HOT + (T_COLD - T_HOT) * np.broadcast_to(x, shape)).astype(dtype)

        self.f = self._feq(self.rho, self.ux, self.uy, self.uz)
        self.g = self._geq(self.temp, self.ux, self.uy, self.uz)
        self.step_count = 0

    def _feq(self, rho, ux, uy, uz):
        cu = 3.0 * (
            C19[:, 0, None, None, None] * ux
            + C19[:, 1, None, None, None] * uy
            + C19[:, 2, None, None, None] * uz
        )
        usq = 1.5 * (ux**2 + uy**2 + uz**2)
        return (W19[:, None, None, None] * rho * (1.0 + cu + 0.5 * cu**2 - usq)).astype(self.dtype)

    def _geq(self, temp, ux, uy, uz):
        # Second order in u. The first-order form left a velocity-dependent
        # error that showed up as a few percent on Nu against the 2D solver.
        cu = 4.0 * (
            C7[:, 0, None, None, None] * ux
            + C7[:, 1, None, None, None] * uy
            + C7[:, 2, None, None, None] * uz
        )
        usq = 2.0 * (ux**2 + uy**2 + uz**2)
        return (W7[:, None, None, None] * temp * (1.0 + cu + 0.5 * cu**2 - usq)).astype(self.dtype)

    def _stream(self, dist, lattice):
        out = np.empty_like(dist)
        for i, c in enumerate(lattice):
            out[i] = np.roll(dist[i], (int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))
        return out

    def _floor_temperature(self) -> np.ndarray:
        """Floor temperature over (x, y), at full hot-wall strength.

        A uniform floor sheds plumes from everywhere at once. Concentrating the
        heat in the middle gives one persistent column rising through the room,
        which is both easier to read and closer to a heater on a floor.
        """
        n, ny = self.p.n, self.p.depth
        if self.p.floor_profile != "bump":
            return np.full((n, ny), T_HOT, dtype=self.dtype)
        x = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
        y = (np.arange(ny) - (ny - 1) / 2) / ((ny - 1) / 2)
        r2 = x[:, None] ** 2 + y[None, :] ** 2
        bump = np.exp(-r2 / (2.0 * self.p.bump_width**2))
        return (T_COLD + (T_HOT - T_COLD) * bump).astype(self.dtype)

    def wall_temps(self) -> tuple[float, float]:
        if self.p.swap_period <= 0:
            return T_HOT, T_COLD
        phase = np.cos(2.0 * np.pi * self.step_count / self.p.swap_period)
        return T_HOT * phase, T_COLD * phase

    def step(self):
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

        t_hot, t_cold = self.wall_temps()
        g_wall = g_post[OPP7].copy()
        if self.p.config == "rb":
            floor = self.t_floor * (t_hot / T_HOT)
            g_wall[:, :, :, 0] = -g_wall[:, :, :, 0] + 2.0 * W7[:, None, None] * floor[None, :, :]
            g_wall[:, :, :, -1] = -g_wall[:, :, :, -1] + 2.0 * W7[:, None, None] * t_cold
        else:
            g_wall[:, 0, :, :] = -g_wall[:, 0, :, :] + 2.0 * W7[:, None, None] * t_hot
            g_wall[:, -1, :, :] = -g_wall[:, -1, :, :] + 2.0 * W7[:, None, None] * t_cold
        g_post[:, self.solid] = g_wall[:, self.solid]

        self.f = self._stream(f_post, C19)
        self.g = self._stream(g_post, C7)
        self.step_count += 1

    def nusselt(self) -> float:
        """Average Nusselt number, taken along whichever axis carries the heat."""
        p = self.p
        t_hot, t_cold = self.wall_temps()
        delta = t_hot - t_cold
        if abs(delta) < 1e-9:
            return float("nan")
        if p.config == "rb":
            flux = self.uz * self.temp - p.diffusivity * np.gradient(self.temp, axis=2)
        else:
            flux = self.ux * self.temp - p.diffusivity * np.gradient(self.temp, axis=0)
        return float(flux[self.fluid].mean() * p.length / (p.diffusivity * delta))

    def speed(self) -> np.ndarray:
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)

    def snapshot(self) -> dict:
        return {
            "step": self.step_count,
            "ra": self.p.ra,
            "nu": self.nusselt(),
            "temp": self.temp.astype(np.float32),
            "speed": self.speed().astype(np.float32),
        }

    def run(self, steps: int, report_every: int = 0, frame_every: int = 0, on_frame=None):
        for k in range(steps):
            self.step()
            if report_every and (k + 1) % report_every == 0:
                print(f"  step {k + 1:>7d}  Nu={self.nusselt():.4f}  umax={self.speed().max():.5f}", flush=True)
            if frame_every and on_frame and (k + 1) % frame_every == 0:
                on_frame(self.snapshot())
            if not np.isfinite(self.rho).all():
                raise FloatingPointError(f"diverged at step {k + 1}; lower u_ref or raise n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=1e5)
    ap.add_argument("--n", type=int, default=64, help="extent in x and z")
    ap.add_argument("--ny", type=int, default=0, help="depth; 0 means a cube")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--u-ref", type=float, default=0.05)
    ap.add_argument("--swap-period", type=int, default=0)
    ap.add_argument("--periodic-y", action="store_true", help="no walls in y; used for the 2D limit")
    ap.add_argument("--config", default="side", choices=("side", "rb"), help="heat a side wall or the floor")
    ap.add_argument("--floor-profile", default="uniform", choices=("uniform", "bump"))
    ap.add_argument("--bump-width", type=float, default=0.4, help="radius of the hot patch, as a fraction of half-width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=0)
    ap.add_argument("--frames-dir")
    ap.add_argument("--frame-every", type=int, default=0)
    ap.add_argument("--float32", action="store_true", help="halve memory; needed at large n")
    ap.add_argument("--steady", action="store_true", help="run until Nu stops moving instead of a fixed step count")
    ap.add_argument("--steady-tol", type=float, default=1e-4)
    ap.add_argument("--check-2d", action="store_true", help="compare against the 2D solver")
    args = ap.parse_args()

    params = Params3D(
        n=args.n, ny=args.ny, ra=args.ra, u_ref=args.u_ref,
        swap_period=args.swap_period, periodic_y=args.periodic_y,
        config=args.config, seed=args.seed,
        floor_profile=args.floor_profile, bump_width=args.bump_width,
    )
    sim = Cavity3D(params, dtype=np.float32 if args.float32 else np.float64)

    cells = args.n * params.depth * args.n
    mem = sim.f.nbytes + sim.g.nbytes
    print(f"cavity {args.n}x{params.depth}x{args.n} = {cells / 1e6:.2f}M cells  Ra={args.ra:.0e}")
    print(f"tau_f={params.tau_f:.4f} tau_g={params.tau_g:.4f}  populations {mem / 1e6:.0f} MB")

    on_frame = None
    if args.frames_dir:
        out = pathlib.Path(args.frames_dir)
        out.mkdir(parents=True, exist_ok=True)
        out.chmod(0o777)

        def on_frame(snap):
            np.savez_compressed(out / f"frame_{snap['step']:08d}.npz", **snap)

    t0 = time.perf_counter()
    if args.steady:
        from lbm import run_to_steady

        nu3, converged = run_to_steady(sim, tol=args.steady_tol, max_steps=args.steps)
        steps_done = sim.step_count
    else:
        sim.run(args.steps, report_every=args.report_every, frame_every=args.frame_every, on_frame=on_frame)
        nu3, converged, steps_done = sim.nusselt(), None, args.steps
    elapsed = time.perf_counter() - t0

    print(f"{elapsed:.1f}s for {steps_done} steps  ({cells * steps_done / elapsed / 1e6:.2f} MLUPS)")
    if converged is False:
        print(f"WARNING: not converged within {args.steps} steps")
    print(f"Nu = {nu3:.4f}")

    if args.check_2d:
        from lbm import Cavity, Params, run_to_steady

        ref = Cavity(Params(n=args.n, ra=args.ra, u_ref=args.u_ref, swap_period=args.swap_period))
        if args.steady:
            nu2, conv2 = run_to_steady(ref, tol=args.steady_tol, max_steps=args.steps)
        else:
            ref.run(args.steps)
            nu2, conv2 = ref.nusselt(), None
        print(f"2D Nu = {nu2:.4f} ({ref.step_count} steps)   3D Nu = {nu3:.4f} ({steps_done} steps)")
        print(f"difference = {abs(nu3 - nu2) / nu2 * 100:.2f}%")
        if conv2 is False:
            print("WARNING: 2D reference not converged")
        mid = params.depth // 2
        print(f"y-variation of temp: {np.abs(sim.temp - sim.temp[:, mid : mid + 1, :]).max():.3e}")


if __name__ == "__main__":
    main()
