#!/usr/bin/env bash
# verify_golden.sh — align pyspike / vendor-spike / ISS to a numpy ground-truth
# golden for one GTX op. Golden priority: <kernel>_numpy_golden.txt (from
# test/gen_golden.py) else <kernel>_ref.txt. Reports, per simulator, PASS/FAIL
# vs the numpy golden (FP16 ulp=1) plus whether ISS produced non-zero output
# (ISS only runs the credit-gated compute ops as zeros on this corpus).
#
# Usage: verify_golden.sh <OP> [per_sim_timeout_sec]
#   OP = directory name (e.g. ABS, RMS_NORM). Emits a one-line summary.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OP="$1"; TMO="${2:-180}"
ART="${GTX_ART_DIR:-/tmp/vgolden}"; mkdir -p "$ART"

SPK=/mnt/e/14_NIGHTLY/gtx_spike/riscv-isa-sim/build
ISS="$REPO/vendor/simulator/GTX_ISS"
VERIFY="$REPO/vendor/gtx_cpp_reference/gtx/verify.py"

ND="$REPO/test/$OP/n1s16"
KSRC=$(ls "$ND"/n1s16_*.c 2>/dev/null | head -1)
[ -z "$KSRC" ] && { echo "[$OP] NO-KERNEL"; exit 0; }
BASE=$(basename "$KSRC" .c)
INPUT="$ND/data/${BASE}_input.txt"
# golden priority: numpy ground truth, else existing ref
if [ -f "$ND/data/${BASE}_numpy_golden.txt" ]; then
  GOLDEN="$ND/data/${BASE}_numpy_golden.txt"; GSRC="numpy"
elif [ -f "$ND/data/${BASE}_ref.txt" ]; then
  GOLDEN="$ND/data/${BASE}_ref.txt"; GSRC="ref"
else echo "[$OP] NO-GOLDEN"; exit 0; fi
[ -f "$INPUT" ] || { echo "[$OP] NO-INPUT"; exit 0; }

OSIZE=$(grep -vE '^@|^$' "$GOLDEN" | tr -d '\n' | wc -c); OSIZE=$((OSIZE/2))
OSIZE_HEX=$(printf '0x%x' "$OSIZE")
NLINES=$(grep -vEc '^@|^$' "$GOLDEN")
grep -vE '^@' "$GOLDEN" > "$ART/${BASE}.golden"

ELF_K="$ART/${BASE}.k.elf"; ELF_U="$ART/${BASE}.uni.elf"
bash "$REPO/src/test/gtx/build_kernel.sh" "$KSRC" "$ELF_K" >"$ART/${BASE}.build.log" 2>&1 || true
bash "$HERE/benchmark/build_uni.sh" "$KSRC" "$ELF_U" >>"$ART/${BASE}.build.log" 2>&1 || true

match() {  # <dump> -> PASS / FAIL:<mm> / NODUMP
  [ -s "$1" ] || { echo NODUMP; return; }
  head -n "$NLINES" "$1" > "$1.cut" 2>/dev/null
  local V r m
  V=$(python3 "$VERIFY" "$1.cut" "$ART/${BASE}.golden" --fp16 --ulp 1 --atol 0.001 2>/dev/null)
  r=$(echo "$V" | grep -E 'Result:' | awk '{print $2}')
  m=$(echo "$V" | grep -E 'Mismatches' | head -1 | awk '{print $NF}')
  [ "$r" = PASS ] && echo PASS || { [ -n "$r" ] && echo "FAIL:$m" || echo ERR; }
}

DPY="$ART/${BASE}.py.hex"; DSP="$ART/${BASE}.sp.hex"; DISS="$ART/${BASE}.iss.hex"
GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$DPY" GTX_DDR_DUMP_ADDR=0x37f000000 \
  GTX_DDR_DUMP_SIZE=$OSIZE GTX_DDR_REVERSED=1 UV_LINK_MODE=copy \
  timeout "$TMO" uv run --no-sync pyspike --extlib=riscv.gtx --extension=gtx \
  --device=gtx_ddr,0x370000000 "$ELF_K" >"$ART/${BASE}.py.log" 2>&1
GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$DSP" GTX_DDR_DUMP_ADDR=0x37f000000 \
  GTX_DDR_DUMP_SIZE=$OSIZE GTX_DDR_REVERSED=1 LD_LIBRARY_PATH="$SPK" \
  timeout "$TMO" "$SPK/spike" --extension=gtx_npu "$ELF_K" >"$ART/${BASE}.sp.log" 2>&1
timeout "$TMO" "$ISS" -I "$ELF_U" -S 0x370000000 -L "$INPUT" -B 0x37f000000 \
  -E "$OSIZE_HEX" -T "$DISS" -V -l 0 -F "$ART/${BASE}.sock" >"$ART/${BASE}.iss.log" 2>&1

ISSZERO=NO; [ -s "$DISS" ] && ! LC_ALL=C grep -qvE '^0*$' "$DISS" && ISSZERO=YES
printf '[%s] golden=%s(%dB)  py=%s  sp=%s  iss=%s  iss_zero=%s\n' \
  "$OP" "$GSRC" "$OSIZE" "$(match "$DPY")" "$(match "$DSP")" "$(match "$DISS")" "$ISSZERO"
