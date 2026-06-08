#!/usr/bin/env bash
# Build a GGML test kernel (test/<OP>/n1s16/<kernel>.c) into a Spike-runnable
# ELF that terminates via HTIF `tohost` (so pyspike's atexit DDR dump fires).
#
# This is the pyspike-native replacement for the stale, committed test .elf
# (which were built with an older crt lacking the tohost symbol and therefore
# never terminated cleanly under pyspike).
#
# Usage:
#   build_kernel.sh <kernel.c> <out.elf>            # build only
#   build_kernel.sh --run <kernel.c> <out.elf> <input.txt> <golden.hex>
#                                                   # build, run under pyspike, diff golden
#
# External sources (override via env if your layout differs):
#   GTX_FIRMWARE     gtx-firmware root (intrinsics + headers)
#   GTX_KERNEL_INC   include dir holding gtx/address.h
#   CROSS_CC         riscv toolchain gcc
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

CC="${CROSS_CC:-riscv64-unknown-elf-gcc}"
GFW="${GTX_FIRMWARE:-/home/sw.lee/supergate_sw/device/gtx-firmware}"
KINC="${GTX_KERNEL_INC:-/home/sw.lee/supergate_sw/device/gtx_kernel/dsppp/src/include}"
INTRIN="$GFW/src/gtx/intrinsics"

RUN=0
if [ "${1:-}" = "--run" ]; then RUN=1; shift; fi

KSRC="$1"; OUT="$2"
KDIR="$(cd "$(dirname "$KSRC")" && pwd)"

CFLAGS="-march=rv64g_xgtxnpu -mabi=lp64d -mcmodel=large -O3 -g -ffreestanding \
  -nostartfiles -ffunction-sections -fdata-sections -std=c11 \
  -DGTX_MAIN_OFFSET=0x370000000ULL \
  -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
  -Wno-missing-field-initializers -Wno-strict-aliasing \
  -Wno-incompatible-pointer-types -Wno-compare-distinct-pointer-types"
INCS="-I$KDIR -I$GFW/include/gtx/intrinsics -I$GFW/include -I$GFW/include/gtx -I$KINC"
# Extra include dir (e.g. the ggml_ops_c corpus root, which holds gtx_isa_compat.h
# pulled in by exp/sqrt/softmax/norm/... kernels).
INCS="$INCS ${EXTRA_INC:+-I$EXTRA_INC}"
SRCS="$KSRC $INTRIN/intrin_level1.c $INTRIN/intrin_level2.c $INTRIN/intrin_level3.c $HERE/spike_crt.S"

# shellcheck disable=SC2086
"$CC" $CFLAGS $INCS -T "$HERE/spike_gtx_link.ld" -nostdlib -Wl,--gc-sections \
  $SRCS -lgcc -o "$OUT"
echo "[build] $OUT  (tohost: $(readelf -s "$OUT" 2>/dev/null | grep -c ' tohost$'))"

if [ "$RUN" -eq 1 ]; then
  INPUT="$3"; GOLDEN="$4"
  # Output size = byte count of golden data (hex chars / 2, '@' lines stripped).
  OSIZE=$(grep -vE '^@|^$' "$GOLDEN" | tr -d '\n' | wc -c)
  OSIZE=$((OSIZE / 2))
  TMPOUT="$(mktemp)"
  echo "[run] dumping 0x37f000000 +$OSIZE B"
  GTX_DDR_SIZE=2G GTX_DDR_INIT="$INPUT" GTX_DDR_DUMP="$TMPOUT" \
    GTX_DDR_DUMP_ADDR=0x37f000000 GTX_DDR_DUMP_SIZE="$OSIZE" GTX_DDR_REVERSED=1 \
    UV_LINK_MODE=copy uv run --no-sync pyspike --extlib=riscv.gtx --extension=gtx \
    --device=gtx_ddr,0x370000000 --device=gtx_spm,0x10000 "$OUT"
  grep -vE '^@' "$GOLDEN" > "${TMPOUT}.golden"
  if diff -q "$TMPOUT" "${TMPOUT}.golden" >/dev/null 2>&1; then
    echo "[PASS] $OUT matches golden"
  else
    echo "[FAIL] $OUT differs from golden ($(diff "$TMPOUT" "${TMPOUT}.golden" | grep -c '^[<>]') lines)"
    exit 1
  fi
fi
