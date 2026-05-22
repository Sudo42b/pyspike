#!/usr/bin/env bash
# Sequential SOLO golden-verification sweep over GTX ops. Reads an ordered op
# list from $1 (one op per line). Per-op timeout isolates hangs.
#
# Run ops SOLO — concurrent pyspike instances contend on CPU and show up as
# false hangs. Orphan spike cleanup runs INSIDE this script (the rtk-rewrite
# Bash hook mangles a `pkill` typed on the caller command line).
#
# Usage: gtx_sweep.sh <oplist_file> [per_op_timeout_sec]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="${1:?need oplist file}"
TMO="${2:-100}"
RESULTS="${GTX_ART_DIR:-/tmp}/gtx_sweep_results.tsv"
: > "$RESULTS"

cleanup() { pkill -9 -f 'data/bin/spike.*extension=gtx' 2>/dev/null; sleep 1; }

n=0
total=$(grep -cvE '^\s*$' "$LIST")
while IFS= read -r OP; do
  [ -z "$OP" ] && continue
  n=$((n+1))
  cleanup
  line=$(timeout "$TMO" bash "$SCRIPT_DIR/gtx_verify.sh" "$OP" 2>&1 | grep -E '^\[' | head -1)
  if [ -z "$line" ]; then
    line="[$OP] HANG/TIMEOUT (>${TMO}s)"
  fi
  printf '%s\n' "$line" | tee -a "$RESULTS"
  printf '[%d/%d] %s\n' "$n" "$total" "$line" >&2
done < "$LIST"
cleanup
echo "SWEEP-DONE total=$n"
