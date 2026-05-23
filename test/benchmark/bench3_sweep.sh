#!/usr/bin/env bash
# bench3_sweep.sh — run bench3.sh across every test/ op, sequentially & SOLO
# (concurrent sims contend on CPU → false hangs), appending a TSV to
# .benchmarks/. Resumable: ops already present in the TSV are skipped.
# Orphan-spike cleanup runs INSIDE this script (the rtk Bash hook mangles a
# `pkill` typed on the caller command line — see gtx_sweep.sh).
#
# Usage: bench3_sweep.sh [per_sim_timeout_sec]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TMO="${1:-180}"
OUTDIR="$REPO/.benchmarks"; mkdir -p "$OUTDIR"
TSV="$OUTDIR/bench3_raw.tsv"
HDR="op	osizeB	t_py	t_sp	t_iss	py_ref	sp_ref	iss_ref	iss_zero	py_eq_sp	py_eq_iss"
[ -f "$TSV" ] || printf '%s\n' "$HDR" > "$TSV"

cleanup() { pkill -9 -f 'data/bin/spike.*extension=gtx' 2>/dev/null; pkill -9 -f 'GTX_ISS' 2>/dev/null; sleep 1; }

# op list: every test/<OP>/n1s16 that has a kernel + golden
mapfile -t OPS < <(
  for d in "$REPO"/test/*/n1s16; do
    op=$(basename "$(dirname "$d")")
    c=$(ls "$d"/n1s16_*.c 2>/dev/null | head -1); [ -z "$c" ] && continue
    b=$(basename "$c" .c)
    { [ -f "$d/data/${b}_ref.txt" ] || [ -f "$d/data/${b}_result.hex" ]; } && echo "$op"
  done | sort -u
)

total=${#OPS[@]}; n=0
echo "[sweep] $total ops, per-sim timeout ${TMO}s, TSV=$TSV" >&2
for OP in "${OPS[@]}"; do
  n=$((n+1))
  if awk -F'\t' -v o="$OP" 'NR>1 && $1==o {f=1} END{exit !f}' "$TSV"; then
    echo "[$n/$total] $OP  (skip — already in TSV)" >&2; continue
  fi
  cleanup
  row=$(timeout $((TMO*3 + 60)) bash "$HERE/bench3.sh" "$OP" "$TMO" 2>>"$OUTDIR/bench3_sweep.err")
  [ -z "$row" ] && row="$OP	?	ERR	ERR	ERR	ERR	ERR	ERR	ERR	ERR	ERR"
  printf '%s\n' "$row" >> "$TSV"
  echo "[$n/$total] $row" >&2
done
cleanup
echo "[sweep] DONE — $(($(wc -l < "$TSV")-1)) rows in $TSV" >&2
