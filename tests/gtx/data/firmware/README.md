# P7 NJIT-04: vendor 84-op firmware build location

P5/P6 shipped 12 .elf via hand-written `.S` sources at
`tests/gtx/data/elf/`. P7 NJIT-04 attempts to build the remaining 72
from `vendor/gtx_cpp_reference/test/<OP>/n1s16/n1s16_<op>.c` using the
`/opt/riscv/` cross-toolchain when available.

## Build status (this checkout)

- `/opt/riscv/bin/riscv64-unknown-elf-gcc` — present (toolchain binary OK).
- `vendor/gtx_cpp_reference/gtx-firmware/` submodule — empty (no
  `include/`, `linker.ld`, or intrinsic sources).
- Alternative GFW at `/home/sw.lee/supergate_sw/device/gtx-firmware/` —
  present, but source tree does NOT include `gtx/address.h` and other
  headers that vendor `n1s16_<op>.c` kernels reference.
- **Result on this checkout:** 0 of 72 vendor kernels build successfully;
  the test harness gracefully skips them at Tier 3 (`no .elf for op X`).

## How to populate this directory (on a developer machine with a fully
populated GFW)

```bash
# 1. Ensure GFW headers + linker.ld + intrinsics are at GTX_FIRMWARE.
export GTX_FIRMWARE=/path/to/gtx-firmware

# 2. Run vendor build script (builds .elf in vendor/.../n1s16/<op>.elf).
cd vendor/gtx_cpp_reference/test
bash run_tests_n1s16.sh --build-only

# 3. Copy each <vendor_dir>/n1s16/n1s16_<op>.elf to firmware/<op>.elf.
# Example:
for op_dir in $(ls -d */); do
    op_name="${op_dir,,}"
    op_name="${op_name%/}"
    src=$(find "${op_dir}" -name "n1s16_*.elf" | head -1)
    if [[ -n "$src" ]]; then
        cp "$src" "../../tests/gtx/data/firmware/${op_name}.elf"
    fi
done

# 4. Re-run sweep:
pytest tests/gtx/test_regression_fw_full_sweep.py --no-cov -q
```

## Test discovery

`tests/gtx/test_regression_fw_full_sweep.py::_find_elf` searches both:
- `tests/gtx/data/firmware/<op>.elf` (P7 location for vendor builds)
- `tests/gtx/data/elf/<op>.elf` (P5/P6 legacy location for the 12
  pre-built .elf)

Either match wins; missing-on-both triggers graceful skip.
