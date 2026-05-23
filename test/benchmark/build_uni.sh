#!/usr/bin/env bash
# build_uni.sh — build ONE "universal" elf that runs on all three GTX sims
# (SystemC-ISS, vendor spike, pyspike). Full gtx-firmware startup (ISS needs it)
# + exit_shim (--wrap=main writes tohost so spike/pyspike exit; startup _Exit
# self-loop halts the ISS). See memory gtx-3sim-benchmark-iss.
#
# Usage: build_uni.sh <kernel.c> <out.elf>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GFW="${GTX_FIRMWARE:-/home/sw.lee/supergate_sw/device/gtx-firmware}"
KINC="${GTX_KERNEL_INC:-/home/sw.lee/supergate_sw/device/gtx_kernel/dsppp/src/include}"
CC="${CROSS_CC:-/opt/riscv/bin/riscv64-unknown-elf-gcc}"

KSRC="$1"; OUT="$2"
KDIR="$(cd "$(dirname "$KSRC")" && pwd)"

CFLAGS="-march=rv64g_xgtxnpu -mabi=lp64d -mcmodel=large -O3 -g -ffreestanding \
  -nostartfiles -ffunction-sections -fdata-sections -std=c11 \
  -DGTX_MAIN_OFFSET=0x370000000ULL \
  -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
  -Wno-missing-field-initializers -Wno-strict-aliasing \
  -Wno-incompatible-pointer-types -Wno-compare-distinct-pointer-types"

INCS="-I$KDIR -I$KINC -I$GFW/include -I$GFW/include/gtx \
  -I$GFW/include/gtx/intrinsics -I$GFW/include/intrinsics -I$GFW/include/drivers"

SRCS="$KSRC \
  $GFW/src/gtx/intrinsics/intrin_level1.c \
  $GFW/src/gtx/intrinsics/intrin_level2.c \
  $GFW/src/gtx/intrinsics/intrin_level3.c \
  $GFW/src/gtx/sc_print.c \
  $GFW/src/startup/startup.c \
  $GFW/src/startup/vector_table.c \
  $GFW/src/riscv/riscv-irq.c \
  $GFW/src/riscv/embeddev_riscv.c \
  $HERE/exit_shim.c"

# shellcheck disable=SC2086
"$CC" $CFLAGS $INCS -T "$GFW/linker.ld" -nostdlib -Wl,--gc-sections \
  -Wl,--wrap=main $SRCS -lgcc -o "$OUT" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[build_uni] $OUT  (tohost: $(readelf -s "$OUT" 2>/dev/null | grep -c ' tohost$'))"
exit $rc
