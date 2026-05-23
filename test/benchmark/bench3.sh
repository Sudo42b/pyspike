#!/usr/bin/env bash
# bench3.sh — 3-simulator benchmark for one GTX op: pyspike / vendor-spike / ISS.
# Builds the proven elf per backend, runs each, times it, and compares output to
# the golden ref (FP16 ULP) and cross-compares pyspike vs spike vs ISS.
#
# Usage: bench3.sh <OP> [per_sim_timeout_sec]
# Emits ONE TSV row on stdout (tab-separated); diagnostics on stderr.
# Columns:
#   op osize_B t_py t_sp t_iss py_ref sp_ref iss_ref iss_zero py_eq_sp py_eq_iss
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OP="$1"; TMO="${2:-180}"
ART="${GTX_ART_DIR:-/tmp/bench3}"; mkdir -p "$ART"

SPK=/mnt/e/14_NIGHTLY/gtx_spike/riscv-isa-sim/build
ISS="$REPO/vendor/simulator/GTX_ISS"
VERIFY="$REPO/vendor/gtx_cpp_reference/gtx/verify.py"

ND="$REPO/test/$OP/n1s16"
KSRC=$(ls "$ND"/n1s16_*.c 2>/dev/null | head -1)
[ -z "$KSRC" ] && { echo -e "$OP\tNO-KERNEL"; exit 0; }
BASE=$(basename "$KSRC" .c)
INPUT="$ND/data/${BASE}_input.txt"
[ -f "$ND/data/${BASE}_ref.txt" ] && GOLDEN="$ND/data/${BASE}_ref.txt" || GOLDEN="$ND/data/${BASE}_result.hex"
{ [ -f "$INPUT" ] && [ -f "$GOLDEN" ]; } || { echo -e "$OP\tNO-DATA"; exit 0; }

OSIZE=$(grep -vE '^@|^$' "$GOLDEN" | tr -d '\n' | wc -c); OSIZE=$((OSIZE/2))
OSIZE_HEX=$(printf '0x%x' "$OSIZE")
NLINES=$(grep -vEc '^@|^$' "$GOLDEN")
grep -vE '^@' "$GOLDEN" > "$ART/${BASE}.golden"

ELF_K="$ART/${BASE}.k.elf"      # build_kernel (tohost) — pyspike + spike
ELF_U="$ART/${BASE}.uni.elf"    # full-startup + exit_shim — ISS
bash "$HERE/../../src/test/gtx/build_kernel.sh" "$KSRC" "$ELF_K" >"$ART/${BASE}.build.log" 2>&1 || true
bash "$HERE/build_uni.sh" "$KSRC" "$ELF_U" >>"$ART/${BASE}.build.log" 2>&1 || true

secs() { echo "scale=1; ($2-$1)/1" | bc; }

# verify_match <dump> -> echoes PASS / FAIL:<mm> / ERR  (FP16 ulp=1)
verify_match() {
  local dump="$1" golden="$2"
  [ -s "$dump" ] || { echo "NODUMP"; return; }
  # truncate dump to golden line count (spike over-dumps)
  head -n "$NLINES" "$dump" > "$dump.cut" 2>/dev/null
  local V res mm
  V=$(python3 "$VERIFY" "$dump.cut" "$golden" --fp16 --ulp 1 --atol 0.001 2>/dev/null)
  res=$(echo "$V" | grep -E 'Result:' | awk '{print $2}')
  mm=$(echo "$V" | grep -E 'Mismatches' | head -1 | awk '{print $NF}')
  if [ "$res" = "PASS" ]; then echo "PASS"; elif [ -n "$res" ]; then echo "FAIL:$mm"; else echo "ERR"; fi
}

# ---- 1) pyspike (build_kernel elf) --------------------------------------
DPY="$ART/${BASE}.py.hex"; : > "$DPY"
t0=$(date +%s.%N)
GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$DPY" GTX_DDR_DUMP_ADDR=0x37f000000 \
  GTX_DDR_DUMP_SIZE=$OSIZE GTX_DDR_REVERSED=1 UV_LINK_MODE=copy \
  timeout "$TMO" uv run --no-sync pyspike --extlib=riscv.gtx --extension=gtx \
  --device=gtx_ddr,0x370000000 "$ELF_K" >"$ART/${BASE}.py.log" 2>&1
rc=$?; t1=$(date +%s.%N)
if [ $rc -eq 124 ]; then T_PY="TIMEOUT"; else T_PY=$(secs "$t0" "$t1"); fi

# ---- 2) vendor spike (build_kernel elf) ---------------------------------
DSP="$ART/${BASE}.sp.hex"; : > "$DSP"
t0=$(date +%s.%N)
GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$DSP" GTX_DDR_DUMP_ADDR=0x37f000000 \
  GTX_DDR_DUMP_SIZE=$OSIZE GTX_DDR_REVERSED=1 LD_LIBRARY_PATH="$SPK" \
  timeout "$TMO" "$SPK/spike" --extension=gtx_npu "$ELF_K" >"$ART/${BASE}.sp.log" 2>&1
rc=$?; t1=$(date +%s.%N)
if [ $rc -eq 124 ]; then T_SP="TIMEOUT"; else T_SP=$(secs "$t0" "$t1"); fi

# ---- 3) SystemC ISS (uni elf) -------------------------------------------
DISS="$ART/${BASE}.iss.hex"; : > "$DISS"
t0=$(date +%s.%N)
timeout "$TMO" "$ISS" -I "$ELF_U" -S 0x370000000 -L "$INPUT" -B 0x37f000000 \
  -E "$OSIZE_HEX" -T "$DISS" -V -l 0 -F "$ART/${BASE}.sock" >"$ART/${BASE}.iss.log" 2>&1
rc=$?; t1=$(date +%s.%N)
if [ $rc -eq 124 ]; then T_ISS="TIMEOUT"; else T_ISS=$(secs "$t0" "$t1"); fi

# ---- comparisons --------------------------------------------------------
PY_REF=$(verify_match "$DPY" "$ART/${BASE}.golden")
SP_REF=$(verify_match "$DSP" "$ART/${BASE}.golden")
ISS_REF=$(verify_match "$DISS" "$ART/${BASE}.golden")
# ISS all-zero?
if [ -s "$DISS" ]; then
  if LC_ALL=C grep -qvE '^0*$' "$DISS"; then ISS_ZERO="NO"; else ISS_ZERO="YES"; fi
else ISS_ZERO="NODUMP"; fi
# cross: pyspike vs spike / iss (byte-exact?)
head -n "$NLINES" "$DSP" > "$DSP.cut" 2>/dev/null
if [ -s "$DPY" ] && [ -s "$DSP.cut" ] && cmp -s "$DPY" "$DSP.cut"; then PY_EQ_SP="EXACT"; else PY_EQ_SP=$(verify_match "$DSP" "$DPY" 2>/dev/null || echo "DIFF"); fi
if [ -s "$DPY" ] && [ -s "$DISS" ] && cmp -s "$DPY" "$DISS"; then PY_EQ_ISS="EXACT"; else PY_EQ_ISS="DIFF"; fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$OP" "$OSIZE" "$T_PY" "$T_SP" "$T_ISS" "$PY_REF" "$SP_REF" "$ISS_REF" "$ISS_ZERO" "$PY_EQ_SP" "$PY_EQ_ISS"
