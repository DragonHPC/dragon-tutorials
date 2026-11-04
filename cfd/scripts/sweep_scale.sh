#!/bin/bash
# Ensemble scaling for sweep.py: one node vs two, repeated.
# Worker counts are sized to each configuration's cores to avoid oversubscription.
cd /Volumes/Shared || exit 1
source env3.12/bin/activate

TILES=${TILES:-36}
N=${N:-256}
STEPS=${STEPS:-6000}
FRAME_EVERY=${FRAME_EVERY:-500}
TILE_SIZE=${TILE_SIZE:-256}
REPS=${REPS:-2}
ONE_WORKERS=${ONE_WORKERS:-20}
TWO_WORKERS=${TWO_WORKERS:-32}
LIMIT=${LIMIT:-1500}
# Without this the default regex matches every NIC and HSTA pairs ip_addrs by
# index, which differs per node. '^bridge' covers bridge0 and any later bridgeN.
NETPREFIX=${NETPREFIX:-'^bridge'}

run_capped() {
  local out_file=$1
  shift
  "$@" >"$out_file" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if ((waited >= LIMIT)); then
      kill -9 "$pid" 2>/dev/null
      pkill -9 -f sweep.py 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 5
    waited=$((waited + 5))
  done
  wait "$pid"
}

run() {
  local label="$1" hf="$2" workers="$3"
  for ((i = 0; i < REPS; i++)); do
    bash perf/kclean.sh >/dev/null 2>&1
    rm -rf /Volumes/Shared/ens
    local tmp rc
    tmp=$(mktemp)
    if [[ -n $hf ]]; then
      run_capped "$tmp" dragon --hostfile hostfile --network-prefix "$NETPREFIX" sweep.py --tiles "$TILES" --n "$N" \
        --steps "$STEPS" --frame-every "$FRAME_EVERY" --workers "$workers" \
        --tile-size "$TILE_SIZE" --out /Volumes/Shared/ens
    else
      run_capped "$tmp" dragon sweep.py --tiles "$TILES" --n "$N" \
        --steps "$STEPS" --frame-every "$FRAME_EVERY" --workers "$workers" \
        --tile-size "$TILE_SIZE" --out /Volumes/Shared/ens
    fi
    rc=$?
    if ((rc == 124)); then
      printf "%-12s rep%d  HUNG (killed after %ss)\n" "$label" "$i" "$LIMIT"
      rm -f "$tmp"
      continue
    fi
    local w f
    w=$(grep -oE 'simulation wall time: [0-9.]+' "$tmp" | grep -oE '[0-9.]+$')
    f=$(grep -oE '^frames: .*' "$tmp")
    printf "%-12s rep%d  wall=%-8s %s\n" "$label" "$i" "${w:-FAIL}" "${f:-no frame line}"
    rm -f "$tmp"
  done
}

echo "=== ensemble: tiles=$TILES n=$N steps=$STEPS reps=$REPS ==="
run "1node" "" "$ONE_WORKERS"
run "2node" "yes" "$TWO_WORKERS"

rm -rf /Volumes/Shared/ens
bash perf/kclean.sh >/dev/null 2>&1
