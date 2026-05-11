# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GTX NPU kernel implementations for GGML tensor operations. Each operation is a bare-metal C kernel targeting the GTX NPU architecture (RISC-V rv64g + custom NPU extensions). Kernels are validated against reference outputs using GTX ISS (Instruction Set Simulator) or Spike with the gtx_npu extension.

## Build & Test Commands

```bash
# Build + run + compare a single kernel
./run_tests_n1s16.sh n1s16_abs

# Run all ~90 kernels
./run_tests_n1s16.sh --all

# Build only (no ISS execution)
./run_tests_n1s16.sh --build-only

# Use Spike instead of GTX ISS
./run_tests_n1s16.sh --spike n1s16_abs
./run_tests_n1s16.sh --spike --all

# Promote ISS output as new golden reference
./run_tests_n1s16.sh --update-ref n1s16_abs
./run_tests_n1s16.sh --update-ref --all

# Generate test data
./run_tests_n1s16.sh --generate

# Compare results with Python FP16 scalar reference
python3 compare_all_ops.py
python3 compare_intrinsic_vs_scalar.py
```

**Toolchain**: `riscv64-unknown-elf-gcc` with `-march=rv64g -mabi=lp64d -mcmodel=large -O0 -g -ffreestanding -nostartfiles -std=c11`

**External dependencies** (paths hardcoded in run_tests_n1s16.sh):
- GTX firmware: `/home/sw.lee/supergate_sw/device/gtx-firmware` (headers, intrinsics, linker script)
- GTX ISS: `../simulator/GTX_ISS`
- Spike: `../build/spike`

## Architecture

### Memory Hierarchy (3-level, explicit DMA)

```
DDR (off-chip)  →  L2 SPM (shared per NEST)  →  L1 SPM banks (per SPU)
```

- **DDR addresses**: Input A @ `0x1000000`, Input B @ `0x2000000`, Result @ `0xf000000`
- **L2 SPM**: On-chip shared memory, addressed from `0x000000`
- **L1 SPM banks**: 4 banks per SPU — typically BANK_A=`0x00000`, BANK_B=`0x20000`, BANK_R=`0x50000`

All data movement is explicit via DMA intrinsics (`__load`, `__store`, `__load_cr`, `__store_cr`).

### Execution Model

```
__split()
  __start_plan(nest_id)
    __start_shared()       ← DDR↔L2 DMA (runs on NEST controller)
    __end_shared()
    for tid in 0..15:
      __start_thread(tid)  ← L2↔L1 DMA + compute (runs on SPU)
      __end_thread(tid)
  __end_plan(nest_id)
__join()
```

- **NEST**: A group of 16 SPUs. Shared context handles DDR↔L2 transfers.
- **SPU thread**: Each SPU processes its slice of rows. `ROWS_PER_SPU = HEIGHT / (NEST_NUM × SPU_NUM_PER_NEST)`.
- **Credit system**: `__load_cr`/`__store_cr` + `__credit_chk` synchronize shared↔thread contexts. `__credit_ld(tid, nest_id)` signals the last load in a thread is complete.

### Kernel Directory Layout

```
OPERATION/
├── n1s1/                          # 1 NEST × 1 SPU variant (scalar reference)
│   └── n1s1_operation.c
└── n1s16/                         # 1 NEST × 16 SPUs (primary target)
    ├── n1s16_operation.c          # Kernel source
    ├── n1s16_operation.elf        # Compiled binary (git-tracked)
    └── data/
        ├── n1s16_operation_input.txt    # Hex input (@address format)
        ├── n1s16_operation_ref.txt      # Golden reference (@address + hex)
        ├── n1s16_operation_result.hex   # ISS/Spike output (hex only)
        └── n1s16_operation_cmd.txt      # Optional: operation parameters
```

### Data Format

**Input/Reference files** (`_input.txt`, `_ref.txt`): `@` prefix lines denote DDR addresses, followed by hex data.
```
@1000000
3f3ebbe4b732bf353d763dfc3fad3a7c...
@2000000
c0b3c384b4cbc0e9b7ce3db3c16737c4...
```

**Result files** (`_result.hex`): Raw hex lines (no `@` prefix). Comparison strips `@` lines from ref before diffing.

**Encoding**: 4 hex chars = 1 FP16 value (big-endian). A typical line has 32 chars = 16 FP16 values.

### Key Headers

- **`fp16_utils.h`**: Integer-only FP16 arithmetic (no float/double — would cause EBREAK on ISS/RTL). Provides `fp16_compare()`, `fp16_to_f32bits()`, `f32bits_to_fp16()`, `FP16_READ/WRITE()` macros.
- **`ggml.h`**: GGML tensor operation reference documentation (100KB, read-only reference).
- **`intrin.h`** (from gtx-firmware): GTX NPU intrinsics — DMA, vector ops, synchronization.

### Common Intrinsics

| Category | Examples |
|----------|---------|
| Vector-vector | `__add_vv()`, `__sub_vv()`, `__mul_vv()`, `__div_vv()` |
| Vector-scalar | `__add_vs()`, `__mul_vs()`, `__max_vs()` |
| Unary | `__neg()`, `__abs()`, `__exp()`, `__sqrt()` |
| Matrix | `__mm()` (matrix multiply) |
| DMA | `__load()`, `__store()`, `__load_cr()`, `__store_cr()` |
| Sync | `__split()`, `__join()`, `__credit_chk()`, `__credit_ld()` |
| SPM config | `__set_spm_addr(ADDR_R, ADDR_unused, ADDR_B, ADDR_A)` |

### Writing a New Kernel

1. Create `OPNAME/n1s16/n1s16_opname.c` following the split→plan→shared→thread pattern
2. Generate test data in `OPNAME/n1s16/data/` (input.txt + ref.txt)
3. Add entries to `run_tests_n1s16.sh`: `OUTPUT_SIZES`, `KERNEL_DIR`, `ALL_KERNELS`
4. Run: `./run_tests_n1s16.sh n1s16_opname`

### Critical Conventions

- **No floating-point C code**: The ISS/RTL traps on host FP instructions. Use `fp16_utils.h` for any FP manipulation in C.
- **Credit on last iteration**: Always call `__credit_ld(tid, nest_id)` on the last loop iteration (`r == ROWS_PER_SPU - 1`) and use `__store_cr` for the last store. Missing credits cause hangs.
- **Bank addressing**: `__set_spm_addr(R_bank, unused, B_bank, A_bank)` — the order is R, unused, B, A.
- **ELF files are git-tracked**: Pre-built ELFs allow testing without the toolchain.
