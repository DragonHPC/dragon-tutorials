---
name: cfd-solvers
description: 'Work on the CFD solver family in this repo (cfd/): the 2D and 3D thermal lattice Boltzmann kernels (lbm.py, lbm3d.py), the decomposed halo-exchange solvers (halo.py, halo2.py, halo3.py, halo3d.py, halo3dv2.py, halo3d_mpi.py), the ensemble sweep (sweep.py), the transport probe (probe.py), and the renderers (render.py, vol3d.py). USE FOR: changing or extending these solvers; choosing a halo-exchange strategy; interpreting or reproducing their measured performance; validating physics against benchmarks; tuning volume-render transfer functions; running them on the two-node Mac cluster. DO NOT USE FOR: general Dragon API questions — see the dragon-api skill for DDict, ProcessGroup, Policy, multiprocessing, launching and cleanup.'
---

# CFD Solvers

The Dragon API itself is covered by the **dragon-api** skill. This file is only
about *these programs*: what they are, what has been measured, and which ideas
have already been tried and failed.

Everything numeric below is measured, not estimated. Entries marked
**SUPERSEDED** were valid on older hardware or an older configuration.

## 1. The programs

| file | what it is |
|---|---|
| `lbm.py` | 2D thermal lattice Boltzmann, D2Q9 flow + D2Q5 temperature, single process |
| `lbm3d.py` | 3D, D3Q19 + D3Q7, side-heated or heated-floor |
| `sweep.py` | **ensemble**: many independent simulations, `mp.Pool` + DDict status |
| `halo.py` | **decomposed 2D**, one domain across ranks, single DDict |
| `halo2.py` | same, explicit rank placement via ProcessGroup + Policy |
| `halo3.py` | same, node-local DDicts for local faces + small shared DDict for seams |
| `halo3d.py` | decomposed 3D, node-local + shared routing — the production solver |
| `halo3dv2.py` | 3D, **consumer-node routing**: one DDict per node, no shared dict |
| `halo3d_mpi.py` | 3D, Dragon launches the ranks, `MPI_Sendrecv` moves the halos |
| `probe.py` | exchange latency with no physics attached |
| `render.py` | 2D colour mapping, montage, mp4 |
| `vol3d.py` | 3D slices and oblique volume rendering |

`halo3dv2.py` and `halo3d_mpi.py` both import `Slab3D` **directly from
halo3d.py**, so the physics is not merely equivalent — it is the same code
object. That is what makes them controlled comparisons of transport alone.

## 2. Physics and validation

- 2D solver vs de Vahl Davis (1983): Ra=10⁴, n=64, steady state gives
  **Nu = 2.2324** against the benchmark 2.2430 — **0.47%**. Ra=10⁵ at n=128 is
  within 0.97%.
- `halo.py` decomposition is exact: ranks 1/2/4 all give Nu = 2.2316, with
  max|ΔT| vs serial = 1.245e-07.
- 3D decomposition is **bitwise** exact: 1 vs 4 ranks gives max|ΔT| = 0.000e+00,
  and all four captured fields across DDict/MPI × two repetitions were
  bit-identical at 128³ float32 after 4000 steps.
- Good visuals: Ra=10⁷, n=512, `u_ref=0.1`, `tau_f=0.541`.

**OPEN**: the 3D solver's Nusselt number sits ~5% below the validated 2D solver
in the 2D limit. Velocity fields agree to 4e-04, so the flow is right and the
visuals are sound, but three hypotheses for the thermal discrepancy failed under
test. **Do not quote a Nusselt number from the 3D solver.**

**Ra=10⁸ 2D imagery is spectacular and physically wrong** (Nu ≈ 200–500 where
~25–30 is expected). Visuals only.

## 3. Decomposition design

1D slab decomposition along x. Each rank owns `[x0, x1)` plus one ghost plane on
each face.

- **Make the decomposition periodic in x** (`left_nbr = (r-1) % P`). Leaving the
  outer ghosts unfilled let `np.roll` wrap junk into wall cells and produced a
  12% Nu error.
- **Only populations are exchanged, never temperature.** Consequently a
  slab-local `np.gradient` reads ghost temperature that was never updated, which
  makes Nu rank-dependent. **Nu must be reduced on the assembled field.**
- Of the 19 D3Q19 directions, 5 have cx=+1 and 5 have cx=-1; D3Q7 adds one each
  way. A face therefore carries 6 values per cell — about 400 KB at n=128.
- The face a rank sends left is consumed by its left neighbour, and vice versa.
  The pairing that matters:

  ```
  from_left  <- left_nbr's  to_right
  from_right <- right_nbr's to_left
  ```

- `--config side` puts the heated walls **on the decomposition axis**, so only
  the two end ranks carry them. The flag was once accepted long before it did
  anything, silently running a heated-floor case under a misleading name. After
  the fix, 1 vs 4 ranks gives max|ΔT| = 0.000e+00 with walls at +0.497/−0.497.

## 4. The four exchange strategies

| solver | routing | reads | writes |
|---|---|---|---|
| `halo.py` | one cluster-wide DDict | ~half remote | ~half remote |
| `halo3.py`, `halo3d.py` | node-local dict for local faces, shared dict for seams | local except seams | local except seams |
| `halo3dv2.py` | **consumer-node**: write into the dict of the node that reads | **all local** | local except seams |
| `halo3d_mpi.py` | `MPI_Sendrecv` | n/a | library-routed |

`halo3dv2.py` is the interesting design. Each rank resolves three handles:

```python
mine      = local_dicts[host]                          # every read
left_out  = local_dicts[placement[slab.left_nbr]]      # to_left  feeds left_nbr
right_out = local_dicts[placement[slab.right_nbr]]     # to_right feeds right_nbr
```

Reads are then unconditional and provably node-local, because the neighbours
deposited the data there by construction. Verified serially: 800/800 reads
node-local at 16 ranks, both block and cyclic, field bit-identical to serial.
Under block placement only **4 of 32 writes per step** leave the node, and that
count is independent of rank count (two ring seams, two directions).

**Checkpoint rule for v2**: a rank may touch up to three dictionaries and must
advance *each distinct one* exactly once per iteration — deduplicate by `id()`.
A seam rank that only *writes* into the far node's dictionary is still a client
of it and must advance it, or readers there block forever. See dragon-api §3.

## 5. Measured performance

### The kernel is memory-bandwidth bound — this frames everything

K independent `python lbm.py` processes, n=256, no Dragon and no communication:

| K | per-process MLUPS | aggregate |
|---|---|---|
| 1 | 9.76 | 9.76 |
| 4 | 6.89 | 27.6 |
| 8 | 5.39 | 43.2 |
| 16 | 2.96 | 47.3 |

Aggregate saturates near **47 MLUPS** regardless of process count. Adding a node
adds memory bandwidth, which this kernel is starved of, and adds latency, which
lockstep cannot hide. Every transport result below is bounded by this.

**UNEXPLAINED**: `halo` r2→r3 per-rank compute drops 13.0 → 6.6 MLUPS, steeper
than the K=2→K=3 drop (8.39 → 7.01) in the no-Dragon test.

### 2D decomposed (n=512, 4000 steps, Ra=10⁶) — SUPERSEDED, pre-bridge

| | 1 node | 2 nodes |
|---|---|---|
| 8 ranks | 32.4 s | 37.9 s |
| 16 ranks | **20.6 s** | 30.9 s |
| 32 ranks | — | 116.5 s (96% exchange) |

Serial is 88.5 s. The second node made 2D *worse*: `probe.py` measures an
exchange at **412 µs within a node and 1609 µs across**, 4×, and with a barrier
every timestep that dominates.

### Placement experiments (2D, n=512/4000, 2 nodes) — SUPERSEDED, pre-bridge

| ranks | block | cyclic |
|---|---|---|
| 8 | 39.3 | 38.4 |
| 16 | 34.6 | 32.2 |
| 32 | 99.5 | 119.6 |

Cutting remote faces 8× bought nothing at r8/r16. Data placement did:

| | 16 ranks | 32 ranks |
|---|---|---|
| one cluster-wide DDict | 34.6 s | 99.5 s |
| node-local + shared | **22.9 s** | **28.5 s** |

1.4× and 3.5×. On the Thunderbolt bridge the node-local configuration at 16
ranks measures **20.3–20.7 s** rather than 22.9.

**Manager counts** (r16, n=512, 4000 steps, 2 reps): raising `--local-managers`
in block placement is monotonically worse — 20.7 / 21.0 / 22.2 s for 1/2/4.
Keep it at 1. `--shared-managers 2` is worth 12% in *cyclic* (31.6 → 27.8 s)
where the shared dict actually carries cross-node traffic, and is irrelevant in
block, which has only 2 shared keys.

### 3D — the second node wins (128³, 400 steps)

| configuration | throughput |
|---|---|
| serial (`lbm3d.py`) | 3.80 MLUPS |
| 1 node, 16 ranks | 19.73 |
| 1 node, 20 ranks | 19.72 (bandwidth ceiling) |
| 2 nodes, 16 ranks (8/8) | 21.63 |
| 2 nodes, 32 ranks (20/12) | **25.71** |

5.2× on one node, 6.8× on two. At 256³: 23.75 MLUPS with exchange share down
from 38% to 21%. A full 256³ production run held **24.4 MLUPS across 40,000
steps and 7h 40m** with no drift and no non-finite values. Short 400-step
benchmarks predicted long runs well (19.73 → 20.20 at 128³, 23.75 → 24.4 at
256³).

### DDict vs MPI (n=128, 4000 steps, 16 ranks block, 5 alternating reps)

| metric | version | median | mean | min | max | spread |
|---|---|---|---|---|---|---|
| wall (s) | DDict | **305.70** | 305.68 | 305.30 | 305.90 | 0.20% |
| wall (s) | MPI | **318.10** | 316.82 | 313.20 | 320.00 | 2.14% |
| MLUPS | DDict | 27.44 | 27.44 | 27.42 | 27.47 | 0.18% |
| MLUPS | MPI | 26.37 | 26.48 | 26.21 | 26.78 | 2.16% |
| exchange % | DDict | 38.40 | 38.40 | 38.30 | 38.50 | 0.52% |
| exchange % | MPI | 40.60 | 41.26 | 36.90 | 46.20 | 22.91% |

**DDict faster by 4.06%, and the distributions do not overlap** — DDict's
slowest run beats MPI's fastest by 7.3 s. DDict is also far more repeatable.
Fair reading: Dragon's DDict is competitive with MPI on a tightly-coupled
nearest-neighbour exchange, MPI's home ground, while needing no MPI install, no
PMI launcher and no executable entry point.

**RETRACTED**: an earlier pair measured 403.5 s (DDict) vs 397.1 s (MPI) and was
briefly read as MPI winning. Both were contaminated by orphaned processes; the
same configuration on a clean cluster runs 305 s.

### halo3d vs halo3dv2 (same protocol, 5 alternating reps)

| metric | orig | v2 |
|---|---|---|
| wall median | 311.00 s | 311.10 s |
| wall spread | 1.06% | **0.35%** |
| exchange % median | 40.20 | **38.60** |
| exchange % spread | 7.46% | **1.55%** |
| MLUPS median | 26.98 | 26.96 |

**Wall time is a dead heat** (+0.03%, ranges fully overlap). Consumer-node
routing does cut exchange share by 1.6 points and makes it **five times more
stable**, but that does not reach wall time because the kernel is
bandwidth-bound. v2 is the more *predictable* program, not the faster one.

Note orig measured 305.70 s in one campaign and 311.00 s three days later —
about 2% of day-to-day drift. **Never compare across campaigns.**

### Ensemble (36 tiles, n=256)

```
1 node (20 workers)   359.7 s
2 nodes (32 workers)  219.9 s     1.64x
```

Reproducible to under 1%, zero frames lost across four runs. The ensemble
collects the extra bandwidth and pays no latency, which is exactly why it scales
where the decomposed 2D solver does not.

## 6. The two parallel shapes

The project implements the same physics two ways because they stress different
things:

- **Ensemble** (`sweep.py`): independent simulations, no inter-worker
  communication, one `mp.Pool`, one DDict holding live per-tile status the
  driver polls.
- **Decomposed** (`halo*.py`): one domain cut across ranks, halo faces exchanged
  **every timestep**.

Work done between communications differs by about four orders of magnitude —
~40 s per message for the ensemble, ~1 ms for the decomposed solver. That single
ratio explains nearly every result above.

## 7. Geometry matters downstream

Temperature distributions are nothing alike:

| | median | p1 | p99 |
|---|---|---|---|
| heated floor | −0.119 | −0.414 | +0.008 |
| side-heated | 0.000 | −0.328 | +0.328 |

The heated floor is a cold bulk with a thin hot plume; the side-heated room is
symmetric. **Rendering settings do not carry across** — cuts that isolate a
plume in one give a lopsided, half-empty picture in the other.

## 8. Rendering

numpy + Pillow with a hand-built lookup table; no matplotlib, because every
worker would pay its import cost to do a colour map. Video uses imageio's
bundled ffmpeg.

The 3D renderer rotates the volume with `scipy.ndimage.map_coordinates` and
composites front to back. **Opacity is keyed to a high-pass of the temperature
field, not to temperature itself** — a convecting box is smooth and fills the
volume, so compositing temperature directly produces uniform fog.

Hard-won rules:

- **A side-heated cavity at Ra=10⁶ has nothing inside the room to look at.** It
  is steady and laminar and all structure is in two thin wall boundary layers.
  No transfer function can invent interior structure. Ra=10⁷ develops lobes and
  sheds blobs, which a slice shows well.
- **Use per-frame percentile cuts, not fixed temperatures.** The room's mean
  temperature drifts, so a fixed cut that isolates a plume at the end sits below
  the bulk earlier on — it became a glowing cube halfway through a run.
- **Key colour to distance past the cut**, not to absolute temperature. A
  diverging map washed the plume to white once the drifting cut landed on the
  map's pale midpoint.
- **Decode frames back out of the encoded mp4 and look at them.** Three movies
  passed every cheap check — correct size, correct frame count, exit code zero —
  and were unusable. The `chk_*_{10,50,99}.png` stills are exactly that check.
- **Verification has to vary the same axis the output does.** The camera rotated
  about the wrong axes for most of the project, tumbling the room end over end.
  Three *time* samples of a rotating movie say almost nothing; four azimuths
  found it in under a minute.
- **A slice is strictly better for reading physics.** `--slice` composites
  nothing, so nothing can hide. A plume stem a few cells across is sharp in a
  slice and nearly invisible in a volume render, because a thin feature
  accumulates almost no opacity along a view ray. Volume renders exist to show
  the flow is three-dimensional.
- `--vectors` arrows are scaled to a quarter of `u_ref`, because measured peak
  speeds are only 14–24% of it.

## 9. Running

See dragon-api for launch mechanics. Solver-specific points:

```bash
# 3D production, ranks split to match core counts
dragon --hostfile hostfile --network-prefix '^bridge' halo3d.py \
    --ranks-per-host "studio=20,max=12" --n 256 --ra 1e7 --steps 40000 \
    --config rb --floor-profile bump --frames-dir out --frame-every 400

# verify a decomposition against a serial run
dragon halo.py --ranks 4 --n 128 --check
```

- **Weight ranks to core counts** (`studio=20,max=12`, not an even 16/16).
  The exchange is lockstep, so oversubscribing the smaller node gates *every*
  rank. Worth 19%.
- Each rank writes its own slab every frame and the driver assembles
  `frame_*.npz` at the end, so an interrupted run can still be assembled from
  whatever slabs survived.
- Frame assembly recomputes bounds rather than reading them from results, by
  design, so a directory of orphaned slabs is still usable.
- `--frame-every 4000` with `--steps 4000` writes exactly one final frame — the
  cheap way to capture a field for an A/B field comparison.
- `halo3d_mpi.py` additionally needs the MPI library path exported on the
  frontend; see dragon-api §4.

## 10. The cluster these numbers come from

| node | cores | memory |
|---|---|---|
| studio | 28 logical / 20 performance | 96 GB |
| max | 16 logical / 12 performance | 48 GB |

- Thunderbolt bridge `bridge0`, 10.10.10.1 / 10.10.10.2, MTU 9000, measured
  **37.47 Gbit/s** single-stream — 94% of TB4 line rate. Office LAN is
  2.30 Gbit/s, so the bridge is 16.3×.
- Dragon-level p2p latency ~**259 µs**, flat from 2 B to 1024 B, stepping to
  ~290–297 µs at 2–4 KB. Use `p2p_lat.py`, not ping.
- **Do not use ICMP as a latency proxy.** macOS loopback ping RTT alone is
  0.204 ms. ICMP RTT doubled once with throughput unchanged and was wrongly
  called a regression.
- The TB bridge is worth **11.4%** at r16 (20.3 vs 22.9 s), with exchange time
  13.1 → 10.1 s and compute unchanged. A **second cable gives nothing** — the
  halo is 8 KB per rank per step, so it is round-trip bound, not bandwidth
  bound.
- NFS: studio is the server; `/Users/leekent/Shared` on studio, `/Volumes/Shared`
  on max.

## 11. Ideas already tried that did not work

- **Pushing snapshots to render workers through an `mp.Queue`.** Three
  successive termination protocols lost frames or hung. Rendering is cheap and
  simulation is not, so the data never needed to move. Removing the queue fixed
  it and made it faster.
- **Rank placement to improve locality.** Nearly useless here; data placement is
  decisive.
- **More managers on a node-local dictionary.** Monotonically worse.
- **A second Thunderbolt cable.** Within noise.
- **Cyclic placement at scale.** Worse than block at r32 (119.6 vs 99.5 s).
