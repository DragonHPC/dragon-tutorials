"""Thermal lattice Boltzmann solver for the differentially heated square cavity.

Natural convection in a square cavity: hot left wall, cold right wall, adiabatic
top and bottom, no-slip everywhere. This is the de Vahl Davis (1983) benchmark,
which gives published Nusselt numbers to validate against.

D2Q9 for the flow, D2Q5 for temperature, coupled by a Boussinesq buoyancy force.
Both collisions are BGK. Streaming is nearest-neighbour only and there is no
global pressure solve, which is why this parallelises without an all-to-all.

Pure numpy, single node, no Dragon dependency. Run directly:

    python lbm.py --ra 1e5 --n 128
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import time
from dataclasses import dataclass

import numpy as np

# D2Q9 lattice for the flow field.
C9 = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int32,
)
W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# D2Q5 lattice for temperature.
C5 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)
W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
OPP5 = np.array([0, 3, 4, 1, 2])

T_HOT = 0.5
T_COLD = -0.5

# de Vahl Davis (1983) benchmark, average Nusselt number at the hot wall, Pr=0.71.
BENCHMARK_NU = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}


@dataclass
class Params:
    n: int = 128
    ra: float = 1e5
    pr: float = 0.71
    u_ref: float = 0.05  # buoyancy velocity in lattice units; keeps Mach low
    swap_period: int = 0  # steps per full hot/cold cycle; 0 holds the walls steady

    @property
    def length(self) -> int:
        """Fluid extent, excluding the solid boundary layer."""
        return self.n - 2

    @property
    def viscosity(self) -> float:
        return self.u_ref * self.length * np.sqrt(self.pr / self.ra)

    @property
    def diffusivity(self) -> float:
        return self.u_ref * self.length / np.sqrt(self.pr * self.ra)

    @property
    def gravity(self) -> float:
        """g*beta*dT, from the definition of the buoyancy velocity."""
        return self.u_ref**2 / self.length

    @property
    def tau_f(self) -> float:
        return 3.0 * self.viscosity + 0.5

    @property
    def tau_g(self) -> float:
        return 3.0 * self.diffusivity + 0.5


class Cavity:
    def __init__(self, params: Params):
        self.p = params
        n = params.n

        self.solid = np.zeros((n, n), dtype=bool)
        self.solid[0, :] = self.solid[-1, :] = True
        self.solid[:, 0] = self.solid[:, -1] = True
        self.fluid = ~self.solid

        self.rho = np.ones((n, n))
        self.ux = np.zeros((n, n))
        self.uy = np.zeros((n, n))

        # Start from a linear conduction profile so the transient is physical.
        x = np.arange(n)[:, None] / (n - 1)
        self.temp = T_HOT + (T_COLD - T_HOT) * np.broadcast_to(x, (n, n)).copy()

        self.f = self._feq(self.rho, self.ux, self.uy)
        self.g = self._geq(self.temp, self.ux, self.uy)
        self.step_count = 0

    def _feq(self, rho, ux, uy):
        cu = 3.0 * (C9[:, 0, None, None] * ux + C9[:, 1, None, None] * uy)
        usq = 1.5 * (ux**2 + uy**2)
        return W9[:, None, None] * rho * (1.0 + cu + 0.5 * cu**2 - usq)

    def _geq(self, temp, ux, uy):
        cu = 3.0 * (C5[:, 0, None, None] * ux + C5[:, 1, None, None] * uy)
        return W5[:, None, None] * temp * (1.0 + cu)

    def _stream(self, dist, lattice):
        out = np.empty_like(dist)
        for i, (cx, cy) in enumerate(lattice):
            out[i] = np.roll(dist[i], (int(cx), int(cy)), axis=(0, 1))
        return out

    def step(self):
        p = self.p

        self.rho = self.f.sum(axis=0)
        # Buoyancy acts on the fluid; the half-force shift is the Guo correction.
        force_y = p.gravity * self.temp * self.fluid
        self.ux = (self.f * C9[:, 0, None, None]).sum(axis=0) / self.rho
        self.uy = ((self.f * C9[:, 1, None, None]).sum(axis=0) + 0.5 * force_y) / self.rho
        self.temp = self.g.sum(axis=0)

        feq = self._feq(self.rho, self.ux, self.uy)
        geq = self._geq(self.temp, self.ux, self.uy)

        # Guo forcing term, first order in velocity is enough at these Mach numbers.
        cy = C9[:, 1, None, None]
        src = (1.0 - 0.5 / p.tau_f) * W9[:, None, None] * 3.0 * cy * force_y

        f_post = self.f - (self.f - feq) / p.tau_f + src
        g_post = self.g - (self.g - geq) / p.tau_g

        # Bounce-back: solid nodes reflect their populations before streaming.
        f_post[:, self.solid] = f_post[OPP9][:, self.solid]

        # Temperature: anti-bounce-back on the hot/cold walls gives Dirichlet,
        # plain bounce-back on top/bottom gives zero flux (adiabatic).
        t_hot, t_cold = self.wall_temps()
        g_wall = g_post[OPP5].copy()
        g_wall[:, 0, :] = -g_wall[:, 0, :] + 2.0 * W5[:, None] * t_hot
        g_wall[:, -1, :] = -g_wall[:, -1, :] + 2.0 * W5[:, None] * t_cold
        g_post[:, self.solid] = g_wall[:, self.solid]

        self.f = self._stream(f_post, C9)
        self.g = self._stream(g_post, C5)
        self.step_count += 1

    def wall_temps(self) -> tuple[float, float]:
        """Hot/cold wall temperatures, optionally swapping over time.

        With swap_period set, the walls cycle through zero and exchange roles, so
        the flow stalls and reverses instead of settling into a steady roll.
        Note this makes Ra a peak value, not a constant.
        """
        if self.p.swap_period <= 0:
            return T_HOT, T_COLD
        phase = np.cos(2.0 * np.pi * self.step_count / self.p.swap_period)
        return T_HOT * phase, T_COLD * phase

    def nusselt(self) -> float:
        """Average Nusselt number over the cavity.

        Nu = <u_x*T - alpha*dT/dx> * L / (alpha * dT), averaged over the fluid.
        """
        p = self.p
        t_hot, t_cold = self.wall_temps()
        delta = t_hot - t_cold
        if abs(delta) < 1e-9:  # walls level mid-cycle, Nu is undefined
            return float("nan")
        dtdx = np.gradient(self.temp, axis=0)
        local = self.ux * self.temp - p.diffusivity * dtdx
        return float(local[self.fluid].mean() * p.length / (p.diffusivity * delta))

    def snapshot(self) -> dict:
        """Compact render-ready state. float32 keeps queue traffic down."""
        speed = np.sqrt(self.ux**2 + self.uy**2)
        vort = np.gradient(self.uy, axis=0) - np.gradient(self.ux, axis=1)
        return {
            "step": self.step_count,
            "ra": self.p.ra,
            "nu": self.nusselt(),
            "u_ref": self.p.u_ref,
            "temp": self.temp.astype(np.float32),
            "ux": self.ux.astype(np.float32),
            "uy": self.uy.astype(np.float32),
            "speed": speed.astype(np.float32),
            "vort": vort.astype(np.float32),
        }

    def run(self, steps: int, report_every: int = 0, frame_every: int = 0, on_frame=None):
        for k in range(steps):
            self.step()
            if report_every and (k + 1) % report_every == 0:
                print(f"  step {k + 1:>7d}  Nu={self.nusselt():.4f}  umax={self._umax():.5f}")
            if frame_every and on_frame and (k + 1) % frame_every == 0:
                on_frame(self.snapshot())
            if not np.isfinite(self.rho).all():
                raise FloatingPointError(f"diverged at step {k + 1}; lower u_ref or raise n")

    def _umax(self) -> float:
        return float(np.sqrt(self.ux**2 + self.uy**2).max())


def run_to_steady(sim, tol: float = 1e-4, window: int = 500, max_steps: int = 400_000):
    """Step until the Nusselt number stops moving, and report whether it did.

    A fixed step count is not comparable across configurations: time to steady
    state grows with n and with 1/u_ref, so the same number of steps can be
    converged in one run and far from it in another.
    """
    prev = None
    while sim.step_count < max_steps:
        sim.run(window)
        nu = sim.nusselt()
        if prev is not None and abs(nu - prev) <= tol * max(abs(nu), 1e-12):
            return nu, True
        prev = nu
    return sim.nusselt(), False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=1e5, help="Rayleigh number")
    ap.add_argument("--n", type=int, default=128, help="lattice size (square)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--report-every", type=int, default=5000)
    ap.add_argument("--swap-period", type=int, default=0, help="steps per hot/cold cycle; 0 = steady walls")
    ap.add_argument("--u-ref", type=float, default=0.05, help="buoyancy velocity; higher develops faster but raises Mach")
    ap.add_argument("--frames-dir", type=str, default="", help="write .npz snapshots here")
    ap.add_argument("--frame-every", type=int, default=0)
    args = ap.parse_args()

    p = Params(n=args.n, ra=args.ra, swap_period=args.swap_period, u_ref=args.u_ref)
    print(f"Ra={p.ra:.0e}  Pr={p.pr}  n={p.n}  nu={p.viscosity:.5f}  alpha={p.diffusivity:.5f}")
    print(f"tau_f={p.tau_f:.4f}  tau_g={p.tau_g:.4f}  Ma={p.u_ref * np.sqrt(3):.3f}  turnover~{p.length / p.u_ref:.0f} steps")
    if p.tau_f < 0.51 or p.tau_g < 0.51:
        print("  WARNING: relaxation time near 0.5, expect instability")

    sim = Cavity(p)

    on_frame = None
    if args.frames_dir:
        out_dir = pathlib.Path(args.frames_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter = itertools.count()

        def on_frame(snap):
            np.savez_compressed(out_dir / f"frame_{next(counter):06d}.npz", **snap)

    t0 = time.perf_counter()
    sim.run(args.steps, args.report_every, args.frame_every, on_frame)
    elapsed = time.perf_counter() - t0

    nu = sim.nusselt()
    mlups = args.steps * p.n * p.n / elapsed / 1e6
    print(f"\nNu = {nu:.4f}")
    ref = BENCHMARK_NU.get(args.ra)
    if ref and not p.swap_period:  # benchmark only applies to steady walls
        print(f"de Vahl Davis reference = {ref:.4f}   error = {100 * abs(nu - ref) / ref:.2f}%")
    print(f"{elapsed:.1f}s for {args.steps} steps  ({mlups:.2f} MLUPS)")


if __name__ == "__main__":
    main()
