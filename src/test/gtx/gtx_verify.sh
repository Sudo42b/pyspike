#!/usr/bin/env bash
# Drive one GTX op golden verification: build -> run under pyspike -> dump DDR
# -> compare with verify.py (FP16 ULP tolerance, not a raw byte diff).
#
# Usage: gtx_verify.sh <OP> [ulp] [atol]      e.g. gtx_verify.sh CLAMP
#
# Env:   GTX_ART_DIR   artifact dir for the built elf / dump / log (default /tmp)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OP="$1"
ULP="${2:-1}"
ATOL="${3:-0.001}"
ART="${GTX_ART_DIR:-/tmp}"

ND="$REPO/test/$OP/n1s16"
KSRC=$(ls "$ND"/n1s16_*.c 2>/dev/null | head -1)
if [ -z "$KSRC" ]; then echo "[$OP] NO-KERNEL"; exit 2; fi
BASE=$(basename "$KSRC" .c)            # e.g. n1s16_clamp
INPUT="$ND/data/${BASE}_input.txt"
# Canonical golden is _ref.txt (@-addressed). _result.hex is a prior ISS dump
# that is stale/corrupt for some ops (e.g. CONV_2D_DW / CONV_3D, where it holds
# nan/eps garbage), so prefer _ref.txt and fall back to _result.hex only when
# _ref.txt is absent.
if [ -f "$ND/data/${BASE}_ref.txt" ]; then
  GOLDEN="$ND/data/${BASE}_ref.txt"
else
  GOLDEN="$ND/data/${BASE}_result.hex"
fi
if [ ! -f "$INPUT" ];  then echo "[$OP] NO-INPUT $INPUT";  exit 2; fi
if [ ! -f "$GOLDEN" ]; then echo "[$OP] NO-GOLDEN $GOLDEN"; exit 2; fi

OUT="$ART/gtxv_${BASE}.elf"
DUMP="$ART/gtxv_${BASE}.dump"
LOG="$ART/gtxv_${BASE}.log"

# 1) build (quiet)
if ! bash "$SCRIPT_DIR/build_kernel.sh" "$KSRC" "$OUT" >"$LOG" 2>&1; then
  echo "[$OP] BUILD-FAIL (see $LOG)"; tail -5 "$LOG"; exit 3
fi

# 2) output size = golden byte count (hex chars / 2, '@' lines stripped)
OSIZE=$(grep -vE '^@|^$' "$GOLDEN" | tr -d '\n' | wc -c); OSIZE=$((OSIZE / 2))

# 3) run under pyspike, dump DDR
t0=$(date +%s)
GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$DUMP" \
  GTX_DDR_DUMP_ADDR=0x37f000000 GTX_DDR_DUMP_SIZE="$OSIZE" GTX_DDR_REVERSED=1 \
  UV_LINK_MODE=copy uv run --no-sync pyspike --extlib=riscv.gtx --extension=gtx \
  --device=gtx_ddr,0x370000000 "$OUT" \
  >>"$LOG" 2>&1
rc=$?
t1=$(date +%s)
if [ ! -s "$DUMP" ]; then echo "[$OP] NO-DUMP rc=$rc ${t1}s (see $LOG)"; tail -5 "$LOG"; exit 4; fi

# 4) golden without '@' address lines
grep -vE '^@' "$GOLDEN" > "${DUMP}.golden"

# 5) verify with FP16 tolerance
V=$(python3 "$REPO/vendor/gtx_cpp_reference/gtx/verify.py" "$DUMP" "${DUMP}.golden" \
      --fp16 --ulp "$ULP" --atol "$ATOL" 2>&1)
res=$(echo "$V" | grep -E 'Result:' | awk '{print $2}')
mm=$(echo "$V" | grep -E 'Mismatches' | head -1 | awk '{print $NF}')
elems=$(echo "$V" | grep -E 'FP16 elements' | awk '{print $NF}')
echo "[$OP] $res  mismatch=$mm/$elems  ${OSIZE}B  $((t1-t0))s  ulp=$ULP"
# full report only on FAIL
if [ "$res" != "PASS" ]; then echo "$V" | sed -n '/First mismatches/,/Result:/p' | head -14; fi
