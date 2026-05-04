# Stack Research — Python RoCC NPU Functional Model on pyspike

**Domain:** Bit-exact functional model of a RISC-V RoCC NPU coprocessor (NEST(4)×SPU(16) GTX), pure-Python on top of pyspike's existing pybind11 trampolines, shipped inside the existing `spike` manylinux2014_x86_64 wheel.

**Researched:** 2026-05-04

**Overall confidence:** HIGH (NumPy/Python decisions verified against current docs and pyspike's existing constraints; bit-exactness recommendations cross-checked against the C++ reference in `gtx_npu.h` and `verify.py`).

**Scope guard:** This document does NOT re-specify the existing pybind11 trampoline layer (`py_extension_t`, `py_rocc_t`, `@riscv.isa.register`, `PYSPIKE_LIBS`, cibuildwheel matrix). Those are already validated (see `.planning/codebase/STACK.md`, `INTEGRATIONS.md`). Recommendations here are strictly additive — what to add to compute, model memory, verify, and ship the GTX NPU port.

---

## TL;DR — Decision Summary

| Question | Answer | Confidence |
|---|---|---|
| NumPy major version | **NumPy 1.26.x (1.26.4)** — pinned upper bound `numpy>=1.20,<2.0` | HIGH |
| FP16 storage / FP16 compute | **Storage: `np.uint16` byte arrays. Compute: FP32 internal (`np.float32`) + single trailing `np.float16` cast.** Match `gtx_fp32_to_16` RNE behavior. | HIGH |
| Memory hierarchy representation | **Per-bank `np.uint8` ndarrays (one per L0/L1/L2 bank), wrapped in a thin Python class.** No mega-array. Use `memoryview` slices for byte writes. | HIGH |
| Endianness | **Store FP16 as little-endian raw bytes in L0/L1.** Use `np.frombuffer(buf, dtype='<u2')` (or `'<f2'`) to interpret. NEVER use `arr.view(np.float16)` blindly — it's host-native (LE on x86_64 but a portability footgun). | HIGH |
| FP16 hex I/O | **Plain stdlib `bytes.fromhex` + `np.frombuffer` for input; `np.ndarray.tobytes() + .hex()` for output.** Reuse the existing `verify.py` parser. | HIGH |
| Disassembler integration | **Reuse pyspike's existing `riscv.disasm.disasm_insn_t` + `arg` decorators exposed in `mod_disasm`.** No new lib. | HIGH |
| Performance backend | **Pure NumPy first.** Defer numba/cython/JAX. Hot-path (gemm) uses `np.einsum` or `np.matmul` on cast-to-fp32 views. | HIGH |
| Test framework | **pytest (already configured) + `pytest.mark.parametrize` over discovered .elf files via `conftest.py`.** No new plugin needed. Optionally `pytest-xdist` for parallel ELF replay. | HIGH |
| Asset packaging | **`importlib.resources.files()` (3.9+) with `importlib_resources` backport for 3.8.** Add `package_data` glob in `pyproject.toml [tool.setuptools.package-data]`. | HIGH |
| New runtime deps to add | **`numpy>=1.20,<2.0` only.** No scipy, no numba, no torch in v1. | HIGH |

---

## 1. NumPy Version Baseline

### Recommendation: `numpy>=1.20,<2.0` (target NumPy 1.26.4)

**Pinning rationale:**

1. **Python 3.8 is a hard constraint.** `pyproject.toml` declares `requires-python = ">=3.8"` and ships `cp38-manylinux_x86_64` wheels. NumPy 2.0 dropped Python 3.8; NumPy 2.3 also moved off `manylinux2014` to `manylinux_2_28`. To keep pyspike's existing five-Python-version wheel matrix intact, NumPy must be **1.26.x or older** for cp38, and the upper bound `<2.0` keeps a single dependency floor across all five interpreters. (Source: NumPy release notes for 2.0 and 2.3.)
2. **`np.float16` semantics have been stable since NumPy 1.20.** RNE rounding, subnormal generation (gradual underflow), NaN propagation, and `±Inf` handling on `np.float16` are IEEE 754 binary16 conformant in 1.20+. No 2.x-only fix is required for our workload.
3. **NumPy 1.26.4 is the last stable 1.x release** (LTS-ish; `manylinux2014` wheels for cp38–cp312 published on PyPI). Use it as the development target; allow the floor at 1.20 because the bytes/strides API we rely on is older than that.

**Anti-recommendation: do NOT add `numpy>=2.0`.** Doing so would either:
- force pyspike to drop cp38 wheels (regression of `PYS-EXT-06`), or
- require maintaining two requirement sets (cp38 → numpy<2; cp39+ → numpy>=2), doubling test surface for no functional gain.

**Code policy:** prefer APIs available since NumPy 1.20 — `np.frombuffer`, `np.ndarray.tobytes`, `np.einsum`, `np.matmul`, `astype(copy=False)`. Avoid 2.0-only features (`np.bool_` rename consequences, copy semantics changes, `np.in1d` removal, etc.).

### FP16 corner-case behavior to be aware of (NumPy 1.20 → 1.26)

| Case | NumPy `np.float16` behavior | Match with `gtx_fp32_to_16`? |
|---|---|---|
| Subnormal (exp==0, mant!=0) | Generated correctly via gradual underflow on FP32→FP16 cast | YES — both round-to-nearest-even |
| `nan` propagation | NaN payload preserved up to the 10-bit fraction; quiet NaN by default | YES (verify.py treats NaN as max ULP, not equal — see PITFALLS.md) |
| `±Inf` on overflow | Cast `np.float32(65520.0).astype(np.float16)` → `+inf` (RNE rounds 65504 < x < 65520 to inf) | YES — matches `gtx_fp32_to_16` overflow branch (`f_exp >= 0x1F → 0x7C00`) |
| `-0.0` preservation | Preserved through all unary ops | YES |
| Round-half-to-even | Default; e.g. `0.000244140625 + ulp/2` ties to even mantissa | YES — `gtx_fp32_to_16` implements RNE explicitly |

**Confidence: HIGH** — NumPy 1.26 docs and source confirm IEEE 754 binary16 RNE; manually checked against `gtx.h:117-151`. One subtlety: NumPy generates **quiet NaN** with payload zero on FP32→FP16 cast unless the input was already a NaN with a non-zero high mantissa bit, while `gtx_fp32_to_16` preserves the **upper 10 bits** of the 23-bit mantissa (`mant >> 13`). For nominal arithmetic this is irrelevant; for NaN-bit-exact tests, see PITFALLS.md.

---

## 2. FP16 Storage vs Compute Strategy

### Recommendation: **Store as `np.uint16` raw bits; compute in `np.float32`; cast back via `np.float16`.**

```python
# Read N FP16 values from L1 byte buffer at offset off (little-endian)
raw = np.frombuffer(spu.l1, dtype='<u2', count=N, offset=off)   # uint16 view
fp32 = raw.view(np.float16).astype(np.float32, copy=False)      # widen for compute

# Compute (always FP32 to match HW accumulator)
acc = (a_fp32 * b_fp32).sum(dtype=np.float32)                   # explicit dtype kwarg

# Cast back through np.float16 — single round, RNE
out_fp16 = np.float16(acc)                                      # equivalent to gtx_fp32_to_16
out_raw = np.array([out_fp16]).view(np.uint16)                  # uint16 representation
spu.l1[off]     = out_raw[0] & 0xFF                             # LE byte 0
spu.l1[off + 1] = (out_raw[0] >> 8) & 0xFF                      # LE byte 1
```

### Why FP32 internal accumulation, not native FP16

The C++ NPU **explicitly mandates FP32 accumulation** for VSUM/DOT and for `mxe_accum`:

> "VSUM 정밀도: FP32 내부 누적 후 1회 FP16 변환. 레퍼런스 매칭 필요 시 행별 분할 후 FP16 부분합 재합산." — `gtx/CLAUDE.md:130`

This is non-negotiable for bit-exactness. Two failure modes if you skip it:

1. **`np.sum(arr_fp16)` accumulates in FP16.** With 16 SPUs reducing 384KB of FP16, FP16 accumulation hits roundoff at ~2048 + ε and produces results that diverge from the C++ reference by tens of ULP, far outside `--ulp 1`.
2. **`np.einsum('ij,jk->ik', a_fp16, b_fp16)`** with FP16 inputs uses FP16 internal accumulation in NumPy 1.x — verified by source. **Always pass `dtype=np.float32` to einsum/matmul reductions** or pre-cast operands to FP32.

### Why a single trailing FP16 cast (not multiple intermediate casts)

`gtx_fp32_to_16` rounds **once** at the end of a fused operation. If your Python code does:

```python
acc = np.float32(0)
for x in row_fp16:
    acc = np.float16(acc + np.float32(x))   # WRONG — rounds every iter
```

…you'll insert N extra rounding steps that the HW does not perform, producing systematic ULP drift. Correct pattern:

```python
acc = np.float32(0)
for x in row_fp16:
    acc = acc + np.float32(x)               # FP32 accumulate
out = np.float16(acc)                       # round once
```

### Row-split partial sums (special case)

For VSUM with row tiling, the spec says "행별 분할 후 FP16 부분합 재합산" — partial sums are cast to FP16 between rows then re-accumulated. This is **a HW-specific accumulator topology**, not a NumPy artifact. Implement explicitly:

```python
partial_fp16 = np.empty(num_rows, dtype=np.float16)
for r in range(num_rows):
    partial_fp16[r] = np.float16(rows_fp32[r].sum(dtype=np.float32))   # FP32 then FP16
result_fp16 = np.float16(partial_fp16.astype(np.float32).sum(dtype=np.float32))
```

**Confidence: HIGH** — verified against `gtx_npu.h:117-151` (`gtx_fp32_to_16`) and the spec note in `gtx/CLAUDE.md:130`.

### Anti-recommendations

| Avoid | Why |
|---|---|
| `np.sum(arr, axis=…)` on a `float16` array without `dtype=np.float32` | Silently accumulates in FP16 — ULP drift. |
| `np.dot(a_fp16, b_fp16)` | FP16 internal product on 1.x — fails ULP gate. |
| `np.float16` arithmetic in tight loops | Slow (software-emulated on most x86_64) AND wrong-precision. Cast up first. |
| `bfloat16` (via ml_dtypes / jax) | The HW is FP16, not BF16. Different exponent width → different overflow/subnormal behavior. |

---

## 3. Memory Hierarchy Representation

### Recommendation: **One `np.uint8` ndarray per memory bank, wrapped in a thin descriptor class. No mega-array.**

```python
class SPU:
    __slots__ = ("l0", "l1", "lspr")
    def __init__(self):
        self.l0  = np.zeros(GTX_L0_SIZE,  dtype=np.uint8)   # 1 KB
        self.l1  = np.zeros(GTX_L1_SIZE,  dtype=np.uint8)   # 384 KB
        self.lspr = np.zeros(0x400, dtype=np.uint64)        # 1024 SPRs

class NEST:
    __slots__ = ("l2", "nspr", "spus")
    def __init__(self):
        self.l2   = np.zeros(GTX_L2_SIZE, dtype=np.uint8)   # 16 MB
        self.nspr = np.zeros(0x400, dtype=np.uint64)
        self.spus = [SPU() for _ in range(GTX_SPUS_PER_NEST)]

class GtxNpu:
    def __init__(self):
        self.gspr  = np.zeros(0x400, dtype=np.uint64)
        self.nests = [NEST() for _ in range(GTX_NUM_NESTS)]
        self.ddr   = None   # lazily allocated; 4GB is too big to pre-zero
```

### Why per-bank, not one big array

1. **Total size of one big array is ~4.4 GB if you include DDR**, ~64 MB just for L1 (4×16×384 KB) + 64 MB for L2 (4×16 MB) + 64 KB for L0. A pre-allocated mega-array forces 4 GB resident memory at process start; per-bank lets DDR be lazy (`ensure_ddr` in `gtx_npu_core.cc`).
2. **Slicing is cheaper at the per-bank level.** `nests[n].spus[s].l1[off:off+N]` is O(1), zero-copy. With a mega-array you'd compute global offsets (`base + n*nest_stride + s*spu_stride + off`) every access — error-prone and harder to debug.
3. **Per-bank arrays line up with HW SRAM banks.** 16 L2 banks × 1 MB matches the HW description (`GTX_L2_NUM_BANKS = 16`). If a future debug build wants per-bank ECC or contention modeling, the structure already maps.

### Use `memoryview` for byte-level writes

```python
# Inside an op — assume l1 is np.uint8 ndarray
mv = memoryview(spu.l1)             # zero-copy, no NumPy machinery
mv[off]     = fp16_raw & 0xFF       # plain Python int ops, fastest path
mv[off + 1] = (fp16_raw >> 8) & 0xFF
```

`memoryview` over a `np.uint8` ndarray is faster than direct ndarray indexing for **scalar byte stores** because it bypasses NumPy's scalar-coercion path. For bulk writes use `arr[off:off+2N] = packed_bytes` instead.

### When to use views vs copies

| Operation | Use | Why |
|---|---|---|
| Read a strided FP16 vector from L1 | `np.frombuffer(spu.l1, dtype='<f2', count=N, offset=off)` | Zero-copy, returns a view |
| Compute on FP16 vector | `view.astype(np.float32, copy=False)` | Cheap widening; may be a view if dtypes are compatible |
| Write FP16 result to L1 | Build a `np.float16` array, `.tobytes()`, then `spu.l1[off:off+2N] = bytes_obj` | Single bulk store; LE bytes guaranteed on x86_64 |
| Bulk DMA copy | `np.copyto(dst_view, src_view)` | NumPy's optimized memcpy path |

**Confidence: HIGH**

---

## 4. Endianness Handling

### Recommendation: **Always be explicit. Use `'<u2'` / `'<f2'` dtype strings everywhere FP16 raw bits cross the L0/L1/DDR boundary.**

### The trap: implicit native byte order on `view()`

```python
# DANGER — looks correct, is host-dependent
buf = np.zeros(8, dtype=np.uint8)
buf[0] = 0xC0; buf[1] = 0x40            # SystemC LE: this is 0x40C0 (= 2.0 in fp16)
val = buf.view(np.float16)[0]            # works on x86_64 (LE host), fails on big-endian
```

`arr.view(np.float16)` is **native byte order**. On x86_64 (LE host) it agrees with the GTX TLM convention by accident, but if anyone ever runs the test suite on a big-endian platform (or imports a numpy array from a different system), the bytes are reinterpreted as if they were big-endian. Two mitigations:

1. **Use explicit endianness in dtype:** `np.frombuffer(buf, dtype='<f2')` always reads as little-endian, regardless of host.
2. **Document the assumption:** assert `sys.byteorder == 'little'` at module import (the project is manylinux2014_x86_64-only, so this always holds — but the assert is a tripwire).

### Reading FP16 from L1 (LE per `gtx/CLAUDE.md:106-110`)

```python
# Canonical pattern — matches the C++ pattern in gtx_npu.h
def read_fp16_le(buf: np.ndarray, off: int) -> np.float16:
    return np.frombuffer(buf, dtype='<f2', count=1, offset=off)[0]

def read_fp16_le_vec(buf: np.ndarray, off: int, n: int) -> np.ndarray:
    return np.frombuffer(buf, dtype='<f2', count=n, offset=off)
```

### Writing FP16 to L1 (LE per `gtx/CLAUDE.md:106-110`)

```python
def write_fp16_le_vec(buf: np.ndarray, off: int, vals: np.ndarray):
    """vals must be np.float16. Writes as little-endian bytes."""
    # Force LE: tobytes() is native-byte-order; on x86_64 that's LE, but be explicit:
    le = vals.astype('<f2', copy=False)
    buf[off:off + 2 * vals.size] = np.frombuffer(le.tobytes(), dtype=np.uint8)
```

### Manual byte writes (matching C++ idiom exactly)

For tests that mirror the C++ `spu.l1[off]= fp16 & 0xFF; spu.l1[off+1]= (fp16>>8)&0xFF` pattern explicitly (e.g. when you want the test to read like the reference), this is bit-equivalent to the `'<u2'` dtype:

```python
raw = int(np.float16(2.0).view(np.uint16))   # 0x4000
buf[off]     = raw & 0xFF                     # 0x00
buf[off + 1] = (raw >> 8) & 0xFF              # 0x40
# Read back via either path:
assert int.from_bytes(buf[off:off+2], 'little') == raw
assert np.frombuffer(buf, dtype='<u2', count=1, offset=off)[0] == raw
```

### DDR hex file convention (different from L1)

Note the discrepancy: `verify.py` parses DDR hex as **big-endian FP16 pairs** (`r_raw = (result[off] << 8) | result[off + 1]` at `verify.py:235`), but L1/L0 in-memory is LE. This is correct — DDR hex files are an external textual representation (one MSB-first hex string per 32-byte bus word), while in-memory SRAM mirrors SystemC TLM (LE). The Python NPU port must:

- Read/write **DDR hex files** with the existing `verify.py:fp16_to_fp32 / fp32_to_fp16` helpers (BE pair).
- Read/write **L1/L0 buffers** with `'<f2'` (LE).
- The DMA path in `gtx_npu_dma.cc` is the seam between the two — it explicitly translates byte order during transfer.

The `GTX_DDR_REVERSED` mode (`gtx/CLAUDE.md:115-119`) further reverses **whole bus words** (32-byte rotation), not individual FP16 byte order. Implement as a separate `np.flip(bus_word_bytes)` step before/after the LE-FP16 read/write.

**Confidence: HIGH** — verified against `gtx/CLAUDE.md:104-119`, `verify.py:217-244`, and `gtx_npu.h:89-114`.

---

## 5. FP16 Hex File I/O

### Recommendation: **Reuse the existing `verify.py` parser. Use stdlib `bytes.fromhex` + `np.frombuffer` for new I/O.**

The existing tools are already correct:

| Need | Reuse from `gtx_spike` | Move to where in pyspike |
|---|---|---|
| Parse DDR hex file (one 32-byte hex line per bus word) | `verify.py:parse_hex_file` | `riscv/gtx/hexio.py` |
| Parse `@addr` sectioned hex | `verify_ref.py:parse_hex_file` | `riscv/gtx/hexio.py` |
| FP16 ↔ FP32 conversion | `verify.py:fp16_to_fp32`, `fp32_to_fp16` | Optional: replace with `np.frombuffer(..., dtype='<f2')` + `astype(np.float32)` since NumPy ≥1.20 conforms to IEEE 754 |
| ULP distance | `verify.py:fp16_ulp_distance` | `riscv/gtx/verify.py` |

### When to keep verify.py's pure-Python conversion vs use NumPy

- **Keep pure-Python** for the verify CLI tool (no NumPy import at startup → faster failure path; users can run `python -m riscv.gtx.verify` without importing the whole numpy/spike stack).
- **Use NumPy** inside the NPU model itself (where NumPy is already loaded).

A stricter-than-stdlib alternative is `ml_dtypes` (TF/JAX-maintained library exposing `bfloat16`/`float8`/`float16` dtypes), but **do not add it** — NumPy 1.26 has correct binary16 already, and `ml_dtypes` adds a wheel-build dependency for zero benefit on FP16.

### Anti-recommendation: do **not** use `struct.pack("e", ...)` for FP16

Python's `struct` `'e'` format (Python 3.6+) implements FP16, but:
- It uses host-native byte order by default (need `<e` for explicit LE).
- It can't vectorize — each call is one value.

`np.frombuffer(buf, dtype='<f2')` is 50–500× faster on N>16 vectors and produces a NumPy array directly usable by downstream ops.

**Confidence: HIGH**

---

## 6. Disassembler Integration

### Recommendation: **Reuse pyspike's existing `riscv.disasm.disasm_insn_t` machinery. Do not introduce a new disassembler.**

pyspike already exposes (per `.planning/codebase/INTEGRATIONS.md` and the `c9cf7c4` commit message "map RoCC extension surface in pyspike binding layer"):

- `mod_disasm` submodule with `disasm_insn_t`
- `arg` decorators in `mod_disasm` (the Python equivalents of spike's `arg_t` operand renderers — `xrd`, `xrs1`, `xrs2`, `imm_*`, etc.)
- `riscv.isa.ROCC.get_disasms(self) -> List[disasm_insn_t]` virtual method, dispatched via the `py_rocc_t` trampoline

### Pattern for GTX

Translate `gtx_npu_disasm.inc` (a static C++ table of `disasm_insn_t` rows) into a Python list returned from `get_disasms`:

```python
from riscv.disasm import disasm_insn_t, arg

class GtxNpu(isa.ROCC):
    def get_disasms(self):
        return [
            # opcode_match, opcode_mask, mnemonic, [arg_renderers...]
            disasm_insn_t(0x0b | (0x40 << 25), 0xfe00707f, "gtx.dma.load",
                          [arg.xrd, arg.xrs1, arg.xrs2]),
            disasm_insn_t(0x0b | (0x10 << 25), 0xfe00707f, "gtx.sasmd.add",
                          [arg.xrd, arg.xrs1, arg.xrs2]),
            # ... one entry per (funct7, funct3) combo in gtx_npu_disasm.inc
        ]
```

### Custom argument renderers

If a GTX instruction has fields that don't map to existing `arg.xrd`/`xrs1`/`xrs2`/`imm_*` (e.g. an SPR address embedded in `funct7`), you have two options:

1. **Define a Python `arg` callable.** spike's `arg_t` is an abstract base whose `to_string(insn) -> str` is overridable; pyspike's `mod_disasm` exposes this so a Python callable can act as an `arg`. Verify via `dir(riscv.disasm)` that `arg` accepts a Python callable; if not, fall back to (2).
2. **Pre-format the mnemonic string** in `get_disasms` to bake the field into the literal name (e.g. `"gtx.spr.0x100"` instead of `"gtx.spr"` + `imm`). Less elegant but always works.

### Anti-recommendations

- **Do not introduce capstone or another disassembler.** spike's disasm is canonical; capstone has no GTX knowledge.
- **Do not hand-format mnemonic strings inside `custom0()`.** That's runtime cost; `get_disasms` is called once during registration and cached.

**Confidence: MEDIUM-HIGH** — High that the integration path exists (commit `c9cf7c4` and `.planning/codebase/INTEGRATIONS.md` confirm it). Medium that all GTX argument flavors map cleanly to existing `arg` factories — there's a small chance one or two GTX-specific renderers (e.g. NEST/SPU mask format) need a custom `arg` callable; verify during implementation.

---

## 7. Python 3.8 vs 3.12 — `np.float16` Differences

### Verdict: **No material differences in `np.float16` semantics across cp38–cp312 when paired with NumPy 1.26.x.**

NumPy 1.26 produces identical FP16 RNE rounding, subnormal generation, and NaN/Inf handling on all CPython 3.8–3.12. The C-level FP16 conversion code in NumPy hasn't changed materially since 1.20. Differences worth being aware of (none of which affect bit-exactness):

| Area | 3.8 | 3.12 | Notes for our work |
|---|---|---|---|
| `int(np.float16(x))` for inf/nan | OverflowError / ValueError | Same | Wrap in try/except in trace logs |
| `repr(np.float16(0.1))` | `'0.0999755859375'` (NumPy formats fully) | Same | Cosmetic; not in bit-exactness path |
| `math.fsum` on cast values | Identical | Identical | Not used (we use NumPy) |
| `struct.pack('<e', ...)` (FP16) | Available (3.6+) | Available | Use `np.frombuffer` instead anyway |
| `importlib.resources.files()` | **Not in stdlib** — needs backport | Available | See packaging section |
| `typing.TypeAlias` | Not available | Available (3.10+) | Use string annotations or `from __future__ import annotations` |

**Bit-exactness corollary:** since `np.float16` rounding is identical across the matrix, **the test suite need not parametrize over Python versions** for ULP correctness. Test once on the developer's primary Python (cp311), trust cibuildwheel's matrix to build wheels for the others.

**Confidence: HIGH**

---

## 8. Performance — NumPy Tricks for NEST(4)×SPU(16)

### Recommendation: **Pure NumPy. No numba / cython / JAX in v1.**

The constraints in `PROJECT.md:122-141` are explicit: no new C++ code, no new runtime deps beyond NumPy. The performance budget is "회귀가 한 세션 내(수십 분 수준) 끝나야 실용" — minutes, not seconds. NumPy alone clears that bar if you avoid Python-level loops over FP16 elements.

### High-leverage patterns

**(a) GEMM via `np.matmul` with explicit FP32 dtype:**

```python
# a_fp16: (M, K), b_fp16: (K, N) — both LE-FP16 in L1
a_fp32 = np.frombuffer(spu.l1, dtype='<f2', count=M*K, offset=a_off).reshape(M, K).astype(np.float32)
b_fp32 = np.frombuffer(spu.l1, dtype='<f2', count=K*N, offset=b_off).reshape(K, N).astype(np.float32)
out = a_fp32 @ b_fp32           # FP32 internal — matches HW MM accumulator
out_fp16 = out.astype(np.float16)
```

`@` calls into BLAS (OpenBLAS on manylinux2014). For 16×16×K operations this is overkill but still 10-100× faster than a Python loop.

**(b) Vectorized SPU broadcast via `np.broadcast_to`:**

When mode-2 (P-only) loop broadcasts across NEST L2:

```python
# Single source, broadcast to all 16 SPUs in one NEST
src_view = np.frombuffer(nest.l2, dtype='<f2', count=N, offset=src_off)
# No copy; broadcasted view fed directly to per-SPU compute
broadcast = np.broadcast_to(src_view, (GTX_SPUS_PER_NEST, N))
```

**(c) Batch over (NEST, SPU) with axis-0 stacking:**

When the same op runs on all (NEST, SPU) pairs (mode-4, P+T):

```python
# Stack all 64 SPUs' inputs into a (4, 16, K) array, run one matmul, scatter back.
ins = np.stack([nest.spus[s].l1_fp32_view(off, K)
                for nest in self.nests for s in range(GTX_SPUS_PER_NEST)])
ins = ins.reshape(GTX_NUM_NESTS, GTX_SPUS_PER_NEST, K)
outs = ins @ weights_fp32        # (4, 16, K) @ (K, N) → (4, 16, N), single BLAS call
```

**(d) Avoid: Python-level loops over FP16 SPM bytes.** A loop like `for i in range(N): spu.l1[off+2*i] = ... ` will dominate runtime even on N=1024.

### When pure NumPy is not enough — defer plan

If a hot path is identified after measurement (use `cProfile` + `snakeviz`):

1. **First**, restructure to push the loop into NumPy (broadcasting, einsum, fancy indexing).
2. **Second**, use `np.vectorize` only for development convenience (it's not faster than a Python loop — it's slower because of dtype overhead).
3. **Third**, only if (1) and (2) fail, consider `numba` in a separate v2 milestone — NOT v1, because numba adds an LLVM-runtime wheel-build dependency that would force re-validating the cibuildwheel pipeline.
4. **Last resort**, drop into a `.cc` file under `src/main/cpp/` and reuse the existing pybind11 build — but this contradicts the "pure-Python" project decision (`PROJECT.md` Key Decisions row 1).

### Anti-recommendations

| Avoid | Why |
|---|---|
| `numba` in v1 | Adds LLVM toolchain to wheel build; not needed for minutes-scale regression. |
| `cython` in v1 | Adds compile step; defeats "pure Python" decision. |
| `jax` / `torch` | Massive deps; FP16 semantics differ (XLA fusing changes rounding); incompatible with manylinux2014 baseline. |
| `np.vectorize` for performance | Misleading name — it's a Python-loop wrapper. |
| `multiprocessing` for NEST parallelism | The model is deterministic and serial; parallelism is for tests, not the model. Use `pytest-xdist` for that. |

**Confidence: HIGH**

---

## 9. Testing — pytest Setup for ISA Regression

### Recommendation: **pytest (already in `[project.optional-dependencies].dev`) + `conftest.py` discovery + `pytest.mark.parametrize`. Add `pytest-xdist` for parallel ELF replay.**

### What's already there

`pyproject.toml` already declares: `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mypy`, `pytest-pylint`, `pytest-repeat`, `pytest-timeout`. No new plugin is strictly required.

### Adding the GTX regression suite

```python
# tests/gtx/conftest.py
import pathlib
import pytest

GTX_FW_DIR = pathlib.Path(__file__).parent / "data" / "firmware"

def pytest_generate_tests(metafunc):
    if "gtx_elf" in metafunc.fixturenames:
        elfs = sorted(GTX_FW_DIR.glob("*.elf"))
        # Use the .elf filename as test ID for clean reporting
        metafunc.parametrize("gtx_elf", elfs, ids=[e.name for e in elfs])
```

```python
# tests/gtx/test_regression.py
def test_firmware_matches_golden(gtx_elf, tmp_path):
    golden = gtx_elf.with_suffix(".golden.hex")
    result = run_pyspike_npu(gtx_elf, dump_to=tmp_path / "result.hex")
    from riscv.gtx.verify import compare_fp16
    stats = compare_fp16(result.read_bytes(), golden.read_bytes(),
                         ulp_tol=1, atol=0.001)
    assert stats["mismatches"] == 0, stats
```

### Useful additions

| Plugin | Why | Recommended? |
|---|---|---|
| `pytest-xdist` | Run ELF regressions in parallel: `pytest -n auto`. Each ELF is independent → 4-8× speedup on a workstation. | **YES, add to `[dev]`** |
| `pytest-randomly` | Catches order-dependent state leaks. NPU state lives on a fresh `GtxNpu` instance per test, so low value here. | NO (defer) |
| `pytest-cases` | Separate cases-from-tests is overkill — `pytest.mark.parametrize` over discovered ELFs is simpler. | NO |
| `pytest-benchmark` | Useful in v2 for perf regression; not bit-exactness. | Defer |
| `hypothesis` | Property-based tests for FP16 op correctness (e.g. `op(0.0) == 0.0` for ABS). Very high leverage for the verify_ref-style scalar ops. | **YES, add to `[dev]`** for the unit-test layer |

### Hypothesis for ULP-tolerant op tests

```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.floats(min_value=-65504, max_value=65504, allow_nan=False, width=16))
def test_relu_matches_reference(x):
    a = np.float16(x)
    spike_out = run_npu_relu(a)
    ref_out = np.maximum(a.astype(np.float32), 0.0).astype(np.float16)
    assert spike_out == ref_out  # bit-exact for ReLU
```

`hypothesis.strategies.floats(width=16)` generates `np.float16`-representable values including subnormals, ±0, ±inf, NaN — perfect for the verify_ref op tests.

### What NOT to do

- **Don't write a custom test discovery system.** `pytest_generate_tests` already does it.
- **Don't put `.elf` files in `tests/data/` outside the package.** They need to ship with the wheel — see packaging section below.
- **Don't mock `processor_t` / `rocc_insn_t`** — use real spike via the existing pyspike test harness (see `tests/test_extension.py`).

**Confidence: HIGH**

---

## 10. Packaging — Shipping `.elf` and `.hex` Assets

### Recommendation: **Add a `riscv/gtx/data/` directory with firmware/golden assets, declare it in `pyproject.toml [tool.setuptools.package-data]`, access via `importlib.resources.files()` (with `importlib_resources` backport for cp38).**

### Layout

```
src/main/python/riscv/gtx/
├── __init__.py
├── npu.py              # GtxNpu class
├── ops/                # MM, VEC, ACT, DMA submodules
│   ├── __init__.py
│   ├── mm.py
│   ├── vec.py
│   ├── act.py
│   └── dma.py
├── disasm.py           # GTX disasm table
├── verify.py           # ported from gtx/verify.py
├── hexio.py            # hex parsing
└── data/
    ├── firmware/
    │   ├── ABS.elf
    │   ├── ADD.elf
    │   └── ...
    └── golden/
        ├── ABS.golden.hex
        └── ...
```

### `pyproject.toml` change

```toml
[tool.setuptools.package-data]
riscv = [
  "data/bin/spike",
  "data/include/**/*.h",
  "data/lib/libdisasm.a",
  "data/lib/libfesvr.a",
  "data/lib/libriscv.so",
  "data/lib/pkgconfig/*.pc",
  "gtx/data/firmware/*.elf",       # NEW
  "gtx/data/golden/*.hex",         # NEW
  "gtx/data/golden/*.golden.hex",  # NEW
]
```

### `MANIFEST.in` change

```
recursive-include src/main/python/riscv/gtx/data *.elf *.hex
```

This is needed because `MANIFEST.in` currently has `recursive-exclude src/main/python/riscv/data *` (line 14) — that's about the spike build artifacts, but be explicit about including the new GTX data tree.

### Resource access (3.8-compatible)

```python
# riscv/gtx/__init__.py
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files as _files
else:
    from importlib_resources import files as _files   # backport

def firmware_path(name: str):
    """Return a Traversable for a bundled .elf firmware."""
    return _files(__package__).joinpath("data", "firmware", name)

def golden_bytes(name: str) -> bytes:
    return _files(__package__).joinpath("data", "golden", name).read_bytes()
```

### Add `importlib_resources` to install requires for cp38 only

```toml
# pyproject.toml
[project]
dependencies = [
  "numpy>=1.20,<2.0",
  'importlib_resources>=5.0; python_version < "3.9"',
]
```

The PEP 508 environment marker keeps the backport off cp39+ (where it's stdlib).

### Anti-recommendations

| Avoid | Why |
|---|---|
| `pkg_resources` (setuptools) | Deprecated; slow startup (imports half of setuptools); does not work cleanly with zip-imported wheels. |
| `pathlib.Path(__file__).parent / "data"` | Fails when the package is run from a zip-import or frozen executable. `files()` is the canonical API. |
| Shipping `.elf` files outside the package (e.g. as a separate `spike-gtx-firmware` package) | Adds a release coordination burden; v1 should be one wheel, one install. |
| Re-generating golden `.hex` files in CI | Goldens are produced by the C++ reference; check them in as data, don't regenerate at test time. |
| Storing in `tests/data/` only | Then the wheel doesn't include them — users who `pip install spike` and write their own NPU tests can't reuse the firmware. Mirror to `riscv/gtx/data/` and have tests reference the package data. |

**Confidence: HIGH**

---

## What NOT to Add

| Library / Tool | Why Not |
|---|---|
| **NumPy 2.x** | Drops cp38 support; conflicts with pyspike's wheel matrix. |
| **scipy** | Only used in `verify_ref.py` for `scipy.special.erf` (one op: `op_gelu_erf`). Reimplement `erf` via NumPy series approximation or skip GELU_ERF in v1 — adding scipy adds 30 MB to wheel deps. |
| **ml_dtypes** | NumPy 1.26 has correct FP16; ml_dtypes is for bf16/fp8/fp4 which the GTX HW doesn't need (FP8 is already implemented in `gtx_fp8_to_32` in C++ with a custom encoding). |
| **numba** | LLVM dep; v1 perf budget achievable with NumPy. |
| **cython** | Adds C compile step; "pure Python" decision (`PROJECT.md`). |
| **jax / torch / tensorflow** | All bring CUDA/XLA assumptions; FP16 fusion changes rounding; massive wheel bloat. |
| **capstone** | spike's disassembler is canonical; capstone has no GTX. |
| **hypothesis-numpy** | Standard `hypothesis.strategies.floats(width=16)` covers our needs. |
| **pytest-cases** | Plain `pytest_generate_tests` is simpler for the .elf-discovery pattern. |
| **`struct.pack('<e', ...)` for hot paths** | Slower than `np.frombuffer` for any N>1; fine for one-off scalars. |

---

## Version Compatibility Matrix

| Component | Version | Compatible With | Notes |
|---|---|---|---|
| Python | 3.8–3.12 | All NumPy 1.20–1.26 | Matches pyspike's existing cibuildwheel matrix. |
| NumPy | 1.26.4 | Python 3.8–3.12, manylinux2014 | Last 1.x; install on PyPI for all our targets. |
| NumPy | 1.20.0 (floor) | Python 3.8–3.10 | `np.frombuffer` count/offset semantics stable; minimum for `dtype='<f2'` working with `np.float16`. |
| pybind11 | >3 (already) | All Pythons in matrix | Existing constraint. |
| pytest | latest (already) | All | Existing. |
| pytest-xdist | latest | pytest | NEW — add to `[dev]`. |
| hypothesis | >=6.0 | Python 3.8+ | NEW — add to `[dev]`. |
| importlib_resources | >=5.0 | cp38 only via marker | NEW — add to runtime deps. |

---

## Installation (additions to existing `pyproject.toml`)

```toml
[project]
dependencies = [
  "numpy>=1.20,<2.0",
  'importlib_resources>=5.0; python_version < "3.9"',
]

[project.optional-dependencies]
dev = [
  # ... existing entries ...
  "hypothesis>=6.0",
  "pytest-xdist",
]
```

```toml
[tool.setuptools.package-data]
riscv = [
  # ... existing spike data ...
  "gtx/data/firmware/*.elf",
  "gtx/data/golden/*.hex",
  "gtx/data/golden/*.golden.hex",
]
```

```
# MANIFEST.in addition
recursive-include src/main/python/riscv/gtx/data *.elf *.hex
recursive-include src/main/python/riscv/gtx *.py *.pyi
```

---

## Sources

- **NumPy 2.0 release notes** — Python 3.8 dropped, manylinux2014 retained: <https://numpy.org/devdocs/release/2.0.0-notes.html> (HIGH confidence)
- **NumPy 2.3.0 release notes** — manylinux upgraded to manylinux_2_28; Python 3.11–3.13 only: <https://numpy.org/devdocs/release/2.3.0-notes.html> (HIGH)
- **NumPy 1.26.4 wheels on PyPI** — confirms cp38–cp312 manylinux2014 wheels: <https://pypi.org/project/numpy/1.26.4/> (HIGH)
- **NumPy byte-swapping & dtype.byteorder docs** — `'<f2'`/`'<u2'` semantics: <https://numpy.org/doc/stable/user/byteswapping.html>, <https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html> (HIGH)
- **NumPy `numpy.finfo`** — float16 binary16 conformance & subnormal handling: <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html> (HIGH)
- **Python `importlib.resources.files()`** — added 3.9; backport via `importlib_resources` PyPI: <https://docs.python.org/3/library/importlib.resources.html>, <https://pypi.org/project/importlib-resources/> (HIGH)
- **pytest parametrize / pytest_generate_tests** — discovery pattern: <https://docs.pytest.org/en/stable/example/parametrize.html> (HIGH)
- **GTX C++ reference** — internal: `gtx_npu.h:89-151` (FP16 conv), `gtx/CLAUDE.md:104-138` (LE byte order, VSUM precision, reset), `gtx_params.h` (memory sizes), `verify.py:217-244` (DDR hex BE pairs) (HIGH — direct source read)
- **pyspike codebase mapping** — `.planning/codebase/STACK.md`, `.planning/codebase/INTEGRATIONS.md`, commit `c9cf7c4` (HIGH — internal authoritative)

---

*Stack research for: Python RoCC NPU functional model on pyspike*
*Researched: 2026-05-04*
