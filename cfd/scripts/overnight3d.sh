#!/bin/bash
# Overnight 256^3 Rayleigh-Benard run across both nodes, then render.
# Ranks are split by hostname to match core counts: the halo exchange is
# lockstep, so oversubscribing the smaller node would gate every rank.
cd /Volumes/Shared || exit 1
source env3.12/bin/activate

OUT=${OUT:-/Volumes/Shared/movies/rb256}
N=${N:-256}
STEPS=${STEPS:-40000}
FRAME_EVERY=${FRAME_EVERY:-400}
RANKS=${RANKS:-"studio=20,max=12"}
LOG=${LOG:-/Volumes/Shared/movies/rb256.log}
# Without this the default regex matches every NIC and HSTA pairs ip_addrs by
# index, which differs per node. '^bridge' covers bridge0 and any later bridgeN.
NETPREFIX=${NETPREFIX:-'^bridge'}

bash perf/kclean.sh >/dev/null 2>&1
# A fresh directory each run: deleting and recreating the same path leaves the
# other node holding a stale NFS handle for it.
OUT="${OUT}_$(date +%H%M%S)"
mkdir -p "$OUT" && chmod 777 "$OUT"

echo "=== $(date) starting 256^3 ===" | tee "$LOG"
# dragon must be a background job with stdin detached: it calls setpgid, and in
# the foreground of a non-interactive shell that earns a stop signal (rc 147).
dragon --hostfile hostfile --network-prefix "$NETPREFIX" halo3d.py \
  --ranks-per-host "$RANKS" --n "$N" --ra 1e7 --u-ref 0.1 --steps "$STEPS" \
  --config rb --floor-profile bump --bump-width 0.3 \
  --frames-dir "$OUT" --frame-every "$FRAME_EVERY" >>"$LOG" 2>&1 </dev/null &
sim_pid=$!
wait "$sim_pid"
rc=$?
echo "=== $(date) simulation exited rc=$rc ===" | tee -a "$LOG"

frames=$(ls "$OUT"/frame_*.npz 2>/dev/null | wc -l | tr -d " ")
slabs=$(ls "$OUT"/slab_*.npy 2>/dev/null | wc -l | tr -d " ")
echo "frames=$frames  leftover slabs=$slabs" | tee -a "$LOG"

if [[ $frames -eq 0 ]]; then
  echo "no frames; nothing to render" | tee -a "$LOG"
  exit 1
fi

# Transfer-function settings are untested at this resolution, so emit a few
# stills alongside the movie rather than betting the whole render on one guess.
for g in 2.0 3.0; do
  for s in 4 7; do
    python vol3d.py "$OUT" --still --size 700 --gamma "$g" --sigma "$s" \
      --density 0.5 --trim 5 --az 40 --el 20 --ramp inferno \
      --out "/Volumes/Shared/movies/rb256_g${g}_s${s}.png" >>"$LOG" 2>&1
  done
done
echo "=== $(date) stills done ===" | tee -a "$LOG"

python vol3d.py "$OUT" --out /Volumes/Shared/movies/rb256_movie.mp4 \
  --size 800 --gamma 2.5 --sigma 6 --density 0.5 --trim 5 \
  --az 35 --el 20 --spin 0.9 --ramp inferno --fps 12 >>"$LOG" 2>&1
echo "=== $(date) time movie done ===" | tee -a "$LOG"

python vol3d.py "$OUT" --turntable 72 --out /Volumes/Shared/movies/rb256_turntable.mp4 \
  --size 800 --gamma 2.5 --sigma 6 --density 0.5 --trim 5 \
  --el 20 --ramp inferno --fps 15 >>"$LOG" 2>&1
echo "=== $(date) turntable done ===" | tee -a "$LOG"

ls -la /Volumes/Shared/movies/rb256* | tee -a "$LOG"
