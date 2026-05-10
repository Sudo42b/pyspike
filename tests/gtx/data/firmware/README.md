# Firmware test fixtures (`tests/gtx/data/firmware/`)

This directory holds **hand-built firmware `.elf`** test fixtures used by the
GTX NPU regression suite. **Vendor pre-built `.elf` files are NOT stored
here** — see "Contract 3 — Vendor `.elf` import procedure" below.

This directory is **excluded from the Python wheel** (D-07 + VTW-04). Only
goldens at `../golden/*.hex` ship to users.

----

## Contract 1 — BE FP16 vs LE FP16 byte-order boundary

The NPU + DDR HW model represents FP16 values in two byte orders depending
on the data source:

- **LE FP16 (pyspike default)** — `np.float16.view(np.uint8)` produces
  `[low_byte, high_byte]`. Pure-Python NumPy backend matches the host's
  little-endian byte order. All P3/P4/P5/P6 hand-built tests use this.
- **BE FP16 (vendor reference)** — vendor `_ref.txt` golden files store
  FP16 values in 32-byte HW DDR bus-word order, parsed right-to-left by
  the SystemC HW simulation that produced them. See
  `vendor/gtx_cpp_reference/gtx/CLAUDE.md` "DDR Hex 파일 바이트 순서"
  for the authoritative vendor specification.

Quoting `vendor/gtx_cpp_reference/gtx/CLAUDE.md` (verbatim):

> ### FP16 바이트 순서
>
> **모든 L1/L0 연산 — Little-Endian (SystemC TLM 일치):**
> ```cpp
> uint16_t raw = spu.l1[off] | ((uint16_t)spu.l1[off + 1] << 8);  // 읽기
> spu.l1[off]     = fp16 & 0xFF;                                    // 쓰기
> spu.l1[off + 1] = (fp16 >> 8) & 0xFF;
> ```
>
> ### DDR Hex 파일 바이트 순서
>
> GTX HW DDR 버스는 256-bit(32-byte) 폭. SystemC HW sim
> (`Memory::readDataFile`)은 hex 라인을 우→좌로 파싱한다 (rightmost byte →
> mem[0]). 3개 프로젝트가 DDR 데이터를 공유하기 위한 규약:
>
> - **기본값:** left-to-right (표준 hex, `objcopy` 등과 호환)
> - **`GTX_DDR_REVERSED=1`:** right-to-left (SystemC HW sim / gtx-risc-vp와 호환)
> - **★ HW sim 데이터 사용 시 반드시 `GTX_DDR_REVERSED=1` 필요**

When pyspike loads / dumps a vendor-derived `.elf`, the
`GTX_DDR_REVERSED=1` env var flips the parser direction so that BE
goldens compare byte-exact against pyspike's LE-default output. The
read/write paths in `src/main/python/riscv/gtx/ddr.py` consult this
env var per call (`ddr_init_from_file` ddr.py:110 + `ddr_dump_to_file`
ddr.py:145), with no module-level cache.

----

## Contract 2 — `GTX_DDR_REVERSED=1` auto-application

`GTX_DDR_REVERSED=1` is set **automatically and inline** by the regression
sweep harness for vendor pre-built `.elf` paths. Hand-built P5/P6 `.elf`
files (LE FP16 native) are unaffected.

Implementation: `tests/gtx/test_regression_fw_full_sweep.py:382-387`
(D-10 inline `subprocess.run(env=...)` env block). The inline check is:

```python
vendor_root_for_env = pathlib.Path(
    os.environ.get("GTX_VENDOR_TEST_DIR", "/mnt/e/14_NIGHTLY/pyspike/test/")
)
is_vendor_elf = elf_path.is_relative_to(vendor_root_for_env)
# ...
if is_vendor_elf:
    env["GTX_DDR_REVERSED"] = "1"
```

`autouse` fixture / pytest marker patterns were rejected (D-10) because
they leak `GTX_DDR_REVERSED` across non-vendor tests, breaking
`test_ddr_modes.py` which exercises both LE and BE paths intentionally.

The same `is_vendor_elf` discriminator gates four other vendor-only env
wirings (Plan 08-04):

- `GTX_DDR_DUMP_ADDR=0xf000000` (vendor uses BASE_DDR_RESULT, not P5/P6's `0x100`)
- `GTX_DDR_INIT=<vendor input.txt>` (vendor `.elf` requires operand pre-staging)
- `GTX_NO_EXIT=1` (let WJOIN return 0 so multi-tile for-loops complete)
- `GTX_DDR_DUMP_SIZE` from `OP_DUMP_SIZE_OVERRIDE` (full-region dump for vendor)

----

## Contract 3 — Vendor `.elf` import procedure

Vendor pre-built `.elf` (79 files) and `_ref.txt` (84 files) live OUTSIDE
this repo at `${GTX_VENDOR_TEST_DIR}` (default
`/mnt/e/14_NIGHTLY/pyspike/test/`).

**Setup**:

```bash
# 1. Set env var (or accept default)
export GTX_VENDOR_TEST_DIR=/mnt/e/14_NIGHTLY/pyspike/test/

# 2. Confirm vendor tree is populated
ls $GTX_VENDOR_TEST_DIR/ABS/n1s16/n1s16_abs.elf

# 3. Generate truncated goldens (always; ~3 KB total, committed)
python scripts/import_vendor_golden.py --all

# 4. Optional: generate full-region goldens for local divergence
#    investigation (.gitignored under tests/gtx/data/golden_full/;
#    ~MB scale per op).
python scripts/import_vendor_golden.py --all --full
```

**CI**: `--all` is sufficient. Full-region (`--full`) is local dev only.

The `--all` mode honors `VENDOR_TO_PYSPIKE_OPS_LOWER` (Plan 08-02) so the
9 P6 hand-aliased goldens (e.g., `add_vv.hex`, `mul_vv.hex`) preserve their
md5 invariant when re-imported — the canonical pyspike op_name wins over
vendor-stem-lowercased aliases.

----

## Contract 4 — `_find_elf` search priority

`tests/gtx/test_regression_fw_full_sweep.py:_find_elf`
(test_regression_fw_full_sweep.py:183-219) resolves an op's `.elf` in
this order (P8 08-04 — VENDOR-FIRST for this sweep harness specifically):

1. `${GTX_VENDOR_TEST_DIR}/<OP_DIR>/n1s16/n1s16_<vendor_stem>.elf`
                                                    (D-13 default = `/mnt/e/14_NIGHTLY/pyspike/test/`)
2. `tests/gtx/data/firmware/<elf_stem>.elf`        (P5/P6 hand-built — wheel-excluded)
3. `tests/gtx/data/elf/<elf_stem>.elf`             (P5/P6 legacy location)

**Precedence rule** (sweep-specific): vendor pre-built wins on collision
because this sweep tests against vendor `_ref.txt` goldens, which require
the multi-tile firmware to produce the full DDR output. Hand-built P5/P6
`.S` kernels output a single row at `0x100` and would always mismatch the
vendor golden; they are exercised by `test_regression_fw_full.py` /
`test_regression_fw_mm.py` (separate test modules, unaffected).

The vendor candidate uses `VENDOR_HOST_TREE_STEM_OVERRIDE` to handle
naming variations (e.g., vendor `n1s16_mul.elf` vs. hand-built
`mul_vv.elf`; vendor `n1s16_div_vv.elf` requires the explicit override).

**Override**: setting `GTX_VENDOR_TEST_DIR=/some/other/path` redirects
candidate 1 only. Candidates 2 and 3 are repo-relative and unaffected.

----

## Wheel size impact statement (D-07 + VTW-04)

This directory (`tests/gtx/data/firmware/`) is excluded from the wheel via:

- `MANIFEST.in:18`: `prune tests/gtx/data/firmware`
- `pyproject.toml:127-128` `[tool.setuptools.exclude-package-data]`:
  `"*" = ["tests/gtx/data/firmware/*", "tests/gtx/data/firmware/**/*"]`

What ships in the wheel:
- `tests/gtx/data/golden/*.hex` (~89 truncated goldens, ~4 KB total) — included
- `tests/gtx/data/elf/*.elf` (12 P5/P6 hand-built fixtures) — included
- `tests/gtx/data/firmware/*.elf` — **excluded**

Base wheel size after `python -m build --wheel`: ≤ 50 MB (cibuildwheel
guard preserved). Vendor `.elf` are NEVER bundled — users with a vendor
checkout set `GTX_VENDOR_TEST_DIR` to use them. The
`test_wheel_excludes_firmware_dir` sentinel
(`tests/gtx/test_wheel_data_present.py`) asserts namelist exclusion at
build time.

----

## P7 build status (legacy preserved)

P7 NJIT-04 attempts to build the remaining 72 vendor kernels from
`vendor/gtx_cpp_reference/test/<OP>/n1s16/n1s16_<op>.c` using the
`/opt/riscv/` cross-toolchain when available.

**Build status (this checkout)**:

- `/opt/riscv/bin/riscv64-unknown-elf-gcc` — present (toolchain binary OK).
- `vendor/gtx_cpp_reference/gtx-firmware/` submodule — empty (no
  `include/`, `linker.ld`, or intrinsic sources).
- Alternative GFW at `/home/sw.lee/supergate_sw/device/gtx-firmware/` —
  present, but source tree does NOT include `gtx/address.h` and other
  headers that vendor `n1s16_<op>.c` kernels reference.
- **Result on this checkout:** 0 of 72 vendor kernels build successfully
  from the in-tree GFW path. Plan 08-02 / 08-04 wired the vendor
  pre-built `.elf` discovery instead (Contract 3 + Contract 4 above).

**How to populate this directory** (on a developer machine with a fully
populated GFW):

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
