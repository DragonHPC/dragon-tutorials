#!/bin/bash
# Scaling matrix for the decomposed halo solver, repeated to average out noise.
cd /Volumes/Shared || exit 1
source env3.12/bin/activate

N=${N:-512}
STEPS=${STEPS:-4000}
RA=${RA:-1e6}
REPS=${REPS:-3}
ONE_RANKS=${ONE_RANKS:-"2 4 8 16"}
TWO_RANKS=${TWO_RANKS:-"4 8 16 32"}
LIMIT=${LIMIT:-300}
# Without this the default regex matches every NIC and HSTA pairs ip_addrs by
# index, which differs per node. '^bridge' covers bridge0 and any later bridgeN.
NETPREFIX=${NETPREFIX:-'^bridge'}

# Dragon intermittently hangs at two-node startup, so cap each run rather than
# letting one stall consume the whole matrix. macOS has no GNU timeout.
run_capped() {
  local out_file=$1
  shift
  "$@" >"$out_file" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if ((waited >= LIMIT)); then
      kill -9 "$pid" 2>/dev/null
      pkill -9 -f halo.py 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 2
    waited=$((waited + 2))
  done
  wait "$pid"
}

run() {
  local label="$1" hf="$2" ranks="$3"
  local times=()
  for ((i = 0; i < REPS; i++)); do
    bash perf/kclean.sh >/dev/null 2>&1
    local tmp rc
    tmp=$(mktemp)
    if [[ -n $hf ]]; then
      run_capped "$tmp" dragon --hostfile hostfile --network-prefix "$NETPREFIX" halo.py --ranks "$ranks" --n "$N" --ra "$RA" --steps "$STEPS"
    else
      run_capped "$tmp" dragon halo.py --ranks "$ranks" --n "$N" --ra "$RA" --steps "$STEPS"
    fi
    rc=$?
    if ((rc == 124)); then
      printf "%-18s rep%d  HUNG (killed after %ss)\n" "$label" "$i" "$LIMIT"
      rm -f "$tmp"
      continue
    fi
    local w x
    w=$(grep -oE '^wall: [0-9.]+' "$tmp" | awk '{print $2}')
    x=$(grep -oE 'exchange share: [0-9.]+' "$tmp" | awk '{print $3}')
    [[ -n $w ]] && times+=("$w")
    printf "%-18s rep%d  wall=%-7s exch=%s%%\n" "$label" "$i" "${w:-FAIL}" "${x:-?}"
    rm -f "$tmp"
  done
  if ((${#times[@]})); then
    printf "%-18s BEST  %s\n" "$label" "$(printf '%s\n' "${times[@]}" | sort -n | head -1)"
  else
    printf "%-18s BEST  none completed\n" "$label"
  fi
}

echo "=== n=$N steps=$STEPS Ra=$RA reps=$REPS ==="
if [[ ${SERIAL:-1} == 1 ]]; then
  echo "--- serial baseline ---"
  for ((i = 0; i < REPS; i++)); do
    python lbm.py --ra "$RA" --n "$N" --steps "$STEPS" 2>&1 | tail -1
  done
fi

if [[ -n $ONE_RANKS ]]; then
  echo "--- single node ---"
  for r in $ONE_RANKS; do run "1node-r$r" "" "$r"; done
fi

if [[ -n $TWO_RANKS ]]; then
  echo "--- two nodes ---"
  for r in $TWO_RANKS; do run "2node-r$r" "yes" "$r"; done
fi

bash perf/kclean.sh >/dev/null 2>&1
