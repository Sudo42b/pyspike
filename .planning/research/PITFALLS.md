# Pitfalls Research

**Domain:** C++ FP16 RoCC NPU functional model → Pure-Python (NumPy) bit-exact port
**Researched:** 2026-05-04
**Confidence:** HIGH (all critical pitfalls cross-verified against `~/NIGHTLY/gtx_spike/gtx/` source; speculative items marked LOW)

**Legend:**
- `[from gtx CLAUDE.md]` — explicitly documented in upstream gtx repo
- `[verified in source]` — confirmed by reading the C++ implementation
- `[speculative]` — likely failure mode based on porting experience, not yet observed

---

## Critical Pitfalls

### Pitfall 1: `verify.py` Big-Endian vs L1/L0 Little-Endian Discrepancy

**What goes wrong:**
The hex compare uses **big-endian** FP16 packing (`r_raw = (data[i*2] << 8) | data[i*2+1]`, `verify.py:235`), while ALL L1/L0/DDR writes in the C++ code emit **little-endian** bytes (`spu.l1[off] = fp16 & 0xFF; spu.l1[off+1] = (fp16 >> 8) & 0xFF`, e.g. `gtx_npu_mm.cc:135-136`, `gtx_npu_vec.cc:56-57`, `gtx_npu_act.cc:55-56`). The comparison "works" only because `verify.py` interprets a swapped 16-bit pair, and a Python port that follows the C++ byte layout but accidentally pre-swaps inside NumPy will produce a phantom mismatch on every word.

**Why it happens:**
A Python port author may discover the byte mismatch and "fix" `verify.py` to little-endian, or use `arr.view(np.uint16)` which yields native-endian (LE on x86) — and instantly desynchronize from the SystemC golden hex format expectation. Conversely, an author may follow `verify.py` literally and write big-endian to L1, breaking SystemC TLM compatibility.

**How to avoid:**
- DO NOT touch `verify.py` byte order. Treat it as a black box: bytes go in, PASS/FAIL comes out.
- Write L1/L0 in **little-endian** matching C++ (every byte pair: `lo = raw & 0xFF; hi = (raw >> 8) & 0xFF`).
- For NumPy view-based access, use `arr.view('<u2')` (explicit little-endian) — never `arr.view(np.uint16)` without endianness suffix.
- Add a unit test: write a known FP16 pattern (e.g. `0x3C00` = 1.0) at L1[0], dump DDR around it, run `verify.py --fp16` against a hex line that reads `003C....` (BE-interpreted) — confirms golden round-trip.

**Warning signs:**
- Every other byte differs in `verify.py` mismatch report
- `--ulp 1` reports astronomical ULP distances (sign bit got swapped into mantissa)
- Single-element results show as `0x003C` vs `0x3C00`

**Phase to address:**
GTX-MEM-01 (memory layer foundation). Must land **before** GTX-MM-01, since MM is the first op that exercises the FP16 round-trip.

---

### Pitfall 2: VSUM/MM_O Per-Element FP16 Cast Breaks Bit-Exactness

**What goes wrong:**
`gtx_npu_vec.cc:102-113` (VSUM) and `gtx_npu_mm.cc:200-205` (MM_O row-sum) accumulate in **FP32 across the entire reduction**, then cast to FP16 **once at the end**. A naive Python port that uses `np.float16` arrays and writes:

```python
sum = np.float16(0)
for x in arr_f16:
    sum += x       # cast back to FP16 every step — WRONG
```

produces ULP-distinct results because each addition truncates to FP16 mantissa. Same trap for DOT, MM_V, ESUM, SOFTMAX (which uses FP32 internal `tmp[]` array, `gtx_npu_act.cc:84`).

**Why it happens:**
- `numpy.add.reduce(arr_f16)` with `dtype` unspecified preserves FP16 dtype → per-step truncation
- "Pythonic" `sum(arr)` hands FP16 to Python's `int+float` machinery → also lossy
- Explicit `arr_f16.astype(np.float32).sum()` works but author may not realize the regimen is mandatory

**How to avoid:**
- **Mandatory regimen:** load FP16 → upcast to FP32 → accumulate in FP32 → single FP16 cast at write-back. Use `arr_f16.astype(np.float32).sum()` or `np.add.reduce(arr_f16, dtype=np.float32)`.
- For row-wise GEMM partial sums: keep FP32 accumulator across rows, only cast at C[i,j] write to L1. `gemm_core` (`gtx_npu_mm.cc:60`) initializes `C` as `std::vector<float>` — Python equivalent: `np.zeros((M,N), dtype=np.float32)`.
- For MM_O (`gtx_npu_mm.cc:200`): `sum = float32(0); for k: sum += fp16_to_fp32(spu.l1[off..])`. The sum stored in `mxe_accum` is FP32 — preserve dtype.
- Add a regression: `np.float16([1.0, 1e-4]*1000).sum()` returns inf in FP16 vs ~0.1 in FP32 — guards against accidental FP16 reduction.

**Warning signs:**
- Tail bits of reductions disagree by 1-3 ULP (subtle)
- Long vectors fail `--ulp 1` while short ones pass
- Catastrophic loss when summing values of mixed magnitude (e.g. softmax denominator)
- `mxe_accum` chains (MM_O → MMC_O → MMC_V) accumulate error linearly

**Phase to address:**
GTX-MM-01 (mxe_accum) AND GTX-VEC-01 (VSUM). Both should add an explicit `dtype=np.float32` regimen test as acceptance criterion.

---

### Pitfall 3: `mxe_accum` Continuity Across MM_O / MM_V Chains

**What goes wrong:**
`gtx_npu_mm.cc:209-212`:
```cpp
if (has_bias) sum += mxe_accum[nest_id][spu_id];   // MMC_O reads prior accum
mxe_accum[nest_id][spu_id] = sum;                   // both MM_O and MMC_O write
```

The pattern is: **MM_O initializes** (no read, write), **MMC_O continues** (read+add+write), **MMC_V continues** (same). A Python port that re-zeros `mxe_accum` per call, or persists it as `np.float16` (loses precision across long chains), or fails to scope it per `(nest_id, spu_id)` tuple, will silently produce wrong GEMM bias values when firmware uses the chain `mm.s → mmc.s → mmc` pattern.

**Why it happens:**
- It's tempting to make `mxe_accum` a method-local, since "it's just a temp"
- The C++ struct has `mxe_accum[GTX_NUM_NESTS][GTX_SPUS_PER_NEST]` — easy to flatten incorrectly
- The first call must zero, subsequent calls accumulate — but funct3 (`is_accumulate` bool) drives this, not call count. Python port may misroute the funct3=1 (mmc.o) vs funct3=0 (mm.o) decision.

**How to avoid:**
- Allocate `self.mxe_accum = np.zeros((GTX_NUM_NESTS, GTX_SPUS_PER_NEST), dtype=np.float32)` at **construction**, never re-create. Only `reset()` zeros it.
- Honor `is_accumulate` exactly as `gtx_npu_mm.cc:333` decides it (driven from `firmware_mm_op` `is_accumulate` parameter, which is `funct7==0x01` for MMC variants).
- Add a regression: emulate firmware `mm.s → mmc.s → mmc` with known A/B; expected final FP16 result should differ from a single fused `mm` call by exactly the bias chain.

**Warning signs:**
- GEMM results match for first row but drift for subsequent rows
- Disabling `mxe_accum` reset between unrelated tests "fixes" some tests, breaks others
- Per-NEST or per-SPU offset-by-one — wrong tuple indexing

**Phase to address:**
GTX-MM-01. Acceptance: a 3-stage chain regression that exercises the read-modify-write order on `mxe_accum`.

---

### Pitfall 4: xs1=0 Quirk — Spike Passes -1, C++ Reads `XPR[insn.rs1]` Directly

**What goes wrong:** [from gtx CLAUDE.md, verified in `gtx_npu_mm.cc:335`]
When `xs1=0` (no rs1 register flag), Spike sets `insn.rs1 = -1` (actually `0xFFFFFFFF` truncated to 5 bits = `0x1F` = `x31`). The C++ code bypasses the trampoline-passed `xs1` parameter and reads `p->get_state()->XPR[insn.rs1]` directly. A Python `custom0(self, proc, insn, xs1, xs2)` implementation that **uses the `xs1` argument** rather than reading from `proc.get_state().XPR[insn.rs1]` will receive garbage (sign-extended -1 = `0xFFFFFFFFFFFFFFFF`).

**Why it happens:**
- pybind11 marshals all 4 args (`proc, insn, xs1, xs2`) — author assumes `xs1` is always usable
- `bits[14:12]` look like funct3, but they're flags. Author may not realize `xs1=0` means "Spike won't pre-fetch the register, but you can still read it manually"
- Some firmware encodings (gem5 simplified DISPATCH_*: `funct7=0x04..0x07`) use `xs1=0,xs2=0,xd=0` — operands come from GSPR instead of registers. Other encodings (e.g. `firmware_mm_op` with funct3=2) actually need the rs1 register.

**How to avoid:**
- **Convention:** in every `customN` Python override, read register operands as `proc.get_state().XPR[insn.rs1]` and `proc.get_state().XPR[insn.rs2]` directly. Treat `xs1`/`xs2` parameters as "the trampoline's pre-marshalled values" but trust `insn.rs1`/`insn.rs2` indices.
- For ops that genuinely have no rs1 (DISPATCH_MM/VEC/ACT/DMA), don't access `XPR[rs1]` at all — just dispatch on funct7.
- Document this in `examples/rocc/` as a non-trivial counter example.
- Verify pyspike's `rocc_insn_t` binding exposes `.rs1`/`.rs2` (not just `.xs1`/`.xs2`); per `.planning/codebase/CONCERNS.md` lines 162-164, both are exposed read-only.

**Warning signs:**
- Random opcode triggers operations with row=65536 / col=65536 (`dim16(0xFFFF) → 0x10000` quirk)
- Crashes on first firmware instruction that has `xs1=0`
- Tests pass when running through ISS-full encoding but fail through gem5-simplified encoding (or vice versa)

**Phase to address:**
GTX-CORE-02 (custom0/custom1 entry). Add a unit test that constructs a synthetic `rocc_insn_t` with `xs1=0` and verifies the dispatcher reads from `XPR[insn.rs1]` rather than the marshalled `xs1` param.

---

### Pitfall 5: Gem5 Simplified vs ISS Full Encoding Collision

**What goes wrong:** [from gtx CLAUDE.md, verified in `gtx_npu.h:264-353`]
Two encoding regimes coexist:
- **gem5 simplified**: `funct7 ∈ {0x04, 0x05, 0x06, 0x07}` → DISPATCH MM/VEC/ACT/DMA, operands staged in GSPR
- **ISS full**: `funct7 ∈ {0x00..0x7F}` per-op (e.g. `0x00=MM`, `0x01=MMC`, `0x40=DMA_LOAD`, `0x4A=OPSET`)

These ranges **overlap**: `funct7=0x00` is both "WRSPR firmware" (gem5) and "MM ISS-full" (ISS). The C++ disambiguates via `insn.rs1 != 0` heuristic in `firmware_mm_op` (`gtx_npu_mm.cc:319-321` comment notes: "collides with gem5 WRSPR/RDSPR encodings; disambiguated by insn.rs1!=0"). A Python port that picks "the obvious dispatch table" (one funct7 → one op) will silently route half the firmware to the wrong handler.

**Why it happens:**
- Looks like a clean switch table; author writes `if funct7 == 0x00: handle_wrspr()` and never sees `firmware_mm_op` get called
- Disambiguation logic is buried in `firmware_mm_op` rather than the top-level dispatch
- Tests may exercise only one encoding (e.g. only ISS-full from `verify_ref.py`) and miss the gem5-simplified firmware path

**How to avoid:**
- Mirror the C++ dispatch order exactly: top-level `custom0` switch on `funct7`, then secondary funct3 check + `insn.rs1` heuristic
- Build a coverage matrix table `(funct7, funct3, has_rs1)` → handler, populated from C++ source; use it as a Python `dict` lookup
- Two regression suites required:
  - One firmware suite using gem5 simplified (`run_tests_n1s16.sh` style)
  - One using ISS-full (`run_llext_tests.sh` style)

**Warning signs:**
- Half the regression tests fail; the other half pass perfectly
- Specific firmware ops never get traced even though firmware should call them
- WRSPR appears to do GEMM (or vice versa) — the dispatch routed to the wrong handler

**Phase to address:**
GTX-DISP-01. Acceptance: explicit funct7×funct3 coverage matrix as test parametrize.

---

### Pitfall 6: WJOIN `exit(0)` Replication

**What goes wrong:** [from gtx CLAUDE.md]
`gtx_npu.h` and the firmware regime expect `WJOIN` (funct7=0x03) to terminate the simulation via `exit(0)` unless `GTX_NO_EXIT` is set. Firmware ends with `WJOIN` in an infinite loop pattern — the simulator must exit, otherwise pytest hangs. A Python port that just returns 0 from `custom0(..funct7=0x03..)` will spin forever.

**Why it happens:**
- "Killing the test runner" feels wrong from Python — author may resist
- pybind11 GIL: calling `os._exit(0)` or `sys.exit()` mid-RoCC is non-obvious and may interact with C++ destructors badly
- pytest doesn't catch raw `os._exit` — fixtures don't tear down

**How to avoid:**
- Honor the `GTX_NO_EXIT` env var. If set, return 0 and let firmware loop until host kills.
- If unset, raise `SystemExit(0)` from the Python `custom0` handler (cleaner than `os._exit`; pybind11 will propagate; pytest will catch and re-raise)
- For unit tests, **always** set `GTX_NO_EXIT=1` and detect WJOIN via a callback/event log instead

**Warning signs:**
- Test runs forever
- pytest worker timeout
- Firmware "completes" but never returns control

**Phase to address:**
GTX-RST-01 (reset and termination). Document the env-var contract clearly in module docstring.

---

### Pitfall 7: `reset()` Stack Pointer Initialization

**What goes wrong:** [from gtx CLAUDE.md]
On `processor_t::reset`, `sp = 0`. The first firmware instruction (`addi sp,sp,-16`) traps because `0 - 16 = 0xFFFFFFFFFFFFFFF0` is not in any valid memory region. The C++ extension hooks `reset()` to set `sp = 0x80100000` (`gtx_npu_core.cc`-style). A Python port that overrides `reset()` but forgets to set sp will see a trap on instruction 1.

**Why it happens:**
- Default `extension_t::reset()` does nothing meaningful from Python's POV
- The trap manifests as "unexpected memory access" deep in spike, not "sp wasn't initialized"
- Author may not realize `reset()` is even called

**How to avoid:**
```python
class GtxNpu(riscv.isa.ROCC):
    def reset(self):
        super().reset()
        # gtx CLAUDE.md: avoid trap on first addi sp,sp,-16
        self.processor.get_state().XPR.write(2, 0x80100000)  # x2 = sp
        self.mxe_accum[:] = 0  # clear FP32 accumulators
```

Verify pyspike binding exposes `XPR.write(idx, value)` — per `.planning/codebase/CONCERNS.md` line 105-110, `extension_t::reset()` IS exposed (line 420 of py_module.cc). Confirm `XPR` is mutable from Python (read-only would be a binding bug — flag for fix).

**Warning signs:**
- "Trap from reserved address 0xFFFFFFFFFFFFFFF0" or similar
- Trap on PC = first firmware instruction
- Test fails immediately, not partway through

**Phase to address:**
GTX-RST-01. Already in PROJECT.md Active list. Acceptance: minimum smoke test ("hello world" .elf with single `addi sp,sp,-16` survives reset → executes one MEXEC NOP cycle → exits via WJOIN").

---

### Pitfall 8: NumPy `float16` Operation Upcasting Behavior

**What goes wrong:** [speculative, partial source verification]
NumPy's `np.float16` has surprising upcast rules:
- `np.float16(a) * np.float16(b)` → result is FP16 (good)
- `np.float16(a) + 1.0` (Python float) → result is FP64 (silently upcasts)
- `np.float16(a) + np.float32(b)` → result is FP32 (potentially desired but easy to miss)
- `arr_f16[i] += scalar_python_float` → in-place, but the RHS arithmetic happens in FP64 first, then casts back to FP16 on store. This may produce different rounding than `arr_f16[i] = np.float16(arr_f16[i] + scalar)`.
- Subnormal flush behavior depends on NumPy version + compiler flags. The C++ `gtx_fp32_to_16` has explicit subnormal logic (`gtx_npu.h:122-135`) — `np.float16(small_value)` may or may not match.

**Why it happens:**
- NumPy's "value-based casting" deprecation history (NEP 50) shifted behavior across versions
- Subnormals are FTZ on some HW under `-ffast-math`; the C++ uses explicit bit manipulation, NumPy may use FPU rounding modes that differ
- `+=` looks like in-place but has hidden temp creation when types disagree

**How to avoid:**
- **Never use `np.float16` arithmetic for the actual computation.** Treat FP16 as a storage format only. Always: load → `astype(np.float32)` → compute → custom `fp32_to_fp16` (use the `verify.py:fp32_to_fp16` reference, it matches `gtx_npu.h:117`).
- Implement `fp16_to_fp32` and `fp32_to_fp16` as **pure-Python bit-manipulation** functions (port from `gtx_npu.h:89-151`), do not delegate to `np.float16(x)` or `x.astype(np.float16)`. This guarantees parity with C++ across NumPy versions.
- Vectorize via `np.frompyfunc(fp32_to_fp16, 1, 1)` or pre-build a 65536-entry `uint16 → fp32` LUT (FP16 has only 65536 values — full LUT is 256KB).
- Add a parametrized test: every FP16 representable value (`range(0x10000)`) round-trips through `fp32_to_16(fp16_to_32(x)) == x` (modulo NaN payloads).

**Warning signs:**
- Subnormal results disagree (very small numbers, exponent=0)
- NaN sign or payload differs
- Tests pass on NumPy 1.20 but fail on NumPy 2.x (or vice versa)
- `--atol 0.001` passes but `--ulp 1` fails on small magnitudes

**Phase to address:**
GTX-CORE-01 (memory layer + FP16 helpers). Build the `fp16_to_fp32` / `fp32_to_fp16` pair **first** with full round-trip test coverage. Everything else depends on it.

---

### Pitfall 9: Activation Direction Asymmetry (ADDRA/ADDRR Reversal)

**What goes wrong:** [from gtx CLAUDE.md, verified in `gtx_npu_act.cc:35-42`]
- **Forward direction** (RELU, SOFTMAX, ESUM): read from `LSPR_SPM_ADDRA`, write to `LSPR_SPM_ADDRR`
- **Reversed direction** (PRELU, GELU, TANH, SIGM): read from `LSPR_SPM_ADDRR`, write to `LSPR_SPM_ADDRA`

A Python port that uses a single `(rd_addr, wr_addr) = (addr_a, addr_r)` setup for all activations will overwrite input data with output for half the activation set — silently corrupting downstream ops because firmware staged input at ADDRR expecting it to remain readable until the output reads from ADDRA.

**Why it happens:**
- Looks arbitrary; not documented in any obvious "Activation API" spec
- Easy to copy-paste from RELU implementation when adding GELU
- Bug only shows up if ADDRA and ADDRR are different L1 regions (which firmware sets up; unit tests with same address won't reveal it)

**How to avoid:**
- Hard-code the asymmetry table in a constant:
  ```python
  REVERSED_ACTIVATIONS = {GTX_ACT_PRELU, GTX_ACT_GELU, GTX_ACT_TANH, GTX_ACT_SIGMOID}
  rd_addr, wr_addr = (addr_r, addr_a) if op in REVERSED_ACTIVATIONS else (addr_a, addr_r)
  ```
- Add a unit test per activation that uses **distinct** ADDRA and ADDRR values, with different known patterns at each, and asserts the correct buffer was overwritten.

**Warning signs:**
- RELU tests pass, GELU/TANH/SIGM/PRELU tests fail
- Activation result appears correct but downstream MM reads wrong operand
- Bug disappears when test happens to use ADDRA == ADDRR

**Phase to address:**
GTX-ACT-01. Acceptance: parametrized test with `(op, expected_rd_addr, expected_wr_addr)` covering all 7 activations.

---

### Pitfall 10: DDR `GTX_DDR_REVERSED` Mode Asymmetry

**What goes wrong:** [from gtx CLAUDE.md, verified in `gtx_npu_dma.cc:480-599`]
- **Default**: hex line is read left-to-right, byte 0 of line → `mem[offset+0]` (`gtx_npu_dma.cc:533-538`)
- **`GTX_DDR_REVERSED=1`**: hex line is read right-to-left per 32-byte bus word, last byte of line → `mem[offset+0]` (`gtx_npu_dma.cc:526-531`). Required to match SystemC HW sim hex format.

The `dump_to_file` symmetric path also has both modes (`gtx_npu_dma.cc:575-583`). A Python port must implement BOTH modes — golden hex files for HW-sim regressions are in reversed format; objcopy-derived test fixtures are in default LTR.

**Why it happens:**
- Author may implement only one mode and "test it works" with consistent fixtures
- The reversal is **per 32-byte line**, not whole-file — easy to miss the boundary
- Mode is selected by env var, not API parameter — runtime-only

**How to avoid:**
- Implement both `ddr_init_from_file(path, reversed=False)` and `ddr_dump_to_file(path, addr, size, reversed=False)` as explicit kwargs.
- Honor `GTX_DDR_REVERSED` env var in module init: `self.ddr_reversed = os.environ.get("GTX_DDR_REVERSED") == "1"`
- Round-trip test: `ddr.fill(pattern); dump(reversed=R); load(reversed=R); assert ddr == pattern` for both R=True and R=False.
- For each .elf regression in `run_tests_n1s16.sh`, propagate `GTX_DDR_REVERSED` exactly as the shell script sets it.

**Warning signs:**
- Result hex file has "swapped within 32-byte chunks" pattern
- Off-by-line failures (rather than off-by-byte)
- HW-sim golden fails but synthetic golden passes

**Phase to address:**
GTX-DMA-01. Acceptance: explicit reverse-mode round-trip + at least one HW-sim derived golden in regression set.

---

### Pitfall 11: P/S/T Loop State Machine + 4-Mode Dispatch

**What goes wrong:** [from gtx CLAUDE.md, partial source verification]
The 4-mode routing in `dispatch()`:
1. No loop → entire NEST × SPU broadcast (DDR-wide ops)
2. P only → NEST internal L2 broadcast
3. P + S → DMA on NEST
4. P + T → compute on NEST + SPU

is selected by which loops are active (`is_ploop`, `is_sloop`, `is_tloop` flags from `startp/endp/starts/ends/startt/endt` in `gtx_npu_loop.cc`). `firmware_mm_op` (`gtx_npu_mm.cc:338-339`) uses `nest = is_ploop ? tmu_id : 0; spu = is_tloop ? curr_id : 0` — the wrong active set silently routes ops to NEST=0, SPU=0 in all modes, corrupting hidden state across other SPUs.

**Why it happens:**
- Loop bookkeeping (`tmu_id`, `curr_id`, flags) lives in instance state spread across `custom1` start/end handlers — easy to leave in inconsistent state
- "It works for single SPU" — most simple tests run with only Mode 4 (P+T) and never exercise the others
- `is_ploop`, `is_tloop` are independent flags — 4 combinations, not 4 mutually exclusive modes

**How to avoid:**
- Make loop state explicit and never auto-zero: `self.is_ploop=False; self.is_sloop=False; self.is_tloop=False; self.tmu_id=0; self.curr_id=0`. Modify only in `startp/endp/...` handlers.
- Add a debug invariant: at every `custom0` entry, log `(is_ploop, is_sloop, is_tloop, tmu_id, curr_id)`; cross-check against the C++ trace for the same firmware.
- Explicit unit test for each of the 4 modes (synthesize a custom1 sequence that enters/exits each combination).

**Warning signs:**
- Single-SPU tests pass; multi-SPU tests give all-zero results outside SPU 0
- NEST 1-3 always look like NEST 0
- `mxe_accum` corruption between SPUs (because ops route to wrong SPU)

**Phase to address:**
GTX-CORE-02. Must validate against gtx C++ trace before GTX-MM-01.

---

### Pitfall 12: pybind11 GIL + RoCC Hot Loop Performance

**What goes wrong:** [from .planning/codebase/CONCERNS.md lines 119-136]
Every `custom0` invocation crosses `PYBIND11_OVERRIDE(reg_t, rocc_t, custom0, ...)` which acquires the GIL. Inside Python, accessing `proc.get_state().XPR[i]` from a tight loop performs N pybind11 type marshals per call. For a firmware that issues 10⁶+ RoCC instructions, this is prohibitively slow.

Additionally, `py::cast(processor_t*)` returns a wrapper whose Python-side validity is tied to the GIL frame — caching `proc_state = proc.get_state()` between RoCC calls is unsafe if the underlying processor is mutated by spike.

**Why it happens:**
- Author sees `custom0(self, proc, insn, ...)` and treats it like a fast Python function
- NumPy makes per-element ops fast; **per-call dispatch overhead dominates**
- GIL serialization is invisible until profiling

**How to avoid:**
- **Stage operands before the inner loop, not inside it.** Read all needed registers once at the top of `custom0`; do not call `proc.get_state().XPR[i]` again inside any vectorizable inner work.
- **Vectorize at the op level**: a single `custom0` call should do as much NumPy work as possible (one whole MM, one whole VEC across `length` elements). Avoid Python loops inside the op.
- **Profile first regression**: run a short firmware with `cProfile`. If `custom0` Python overhead > 50% of wall time, restructure ops to amortize.
- **Don't cache pybind11 wrapper objects across RoCC calls.** Re-acquire `proc.get_state()` each call.
- For `mxe_accum` and similar persistent state: keep in `self.*` Python attributes, NOT in spike-side state.

**Warning signs:**
- Regression takes 100x longer than C++ libgtx_npu.so
- `cProfile` shows >50% time in pybind11 marshalling
- Tests that pass time out under CI

**Phase to address:**
GTX-CORE-02 (initial perf budget) + spot-check after GTX-MM-01 lands.

---

### Pitfall 13: NumPy Fancy Indexing Breaks In-Place Memory Updates

**What goes wrong:** [speculative]
The C++ code mutates L1 in place: `spu.l1[off] = lo; spu.l1[off+1] = hi`. Naive Python equivalent `self.l1[nest_id][spu_id][off:off+2] = bytes_view` works only when `self.l1` is structured as a contiguous `(NEST, SPU, L1_SIZE)` `np.uint8` ndarray. If the author uses fancy indexing — e.g. `self.l1[mask] = new_bytes` — NumPy creates a copy, the write doesn't propagate, and subsequent reads see stale data.

Modular addressing `(addr_r + i*2) % GTX_L1_SIZE` requires `np.put` or explicit slice — `arr[idx % size] = ...` works for scalars but NOT for arrays of indices that wrap around (last element overwrites earlier ones in unspecified order).

**Why it happens:**
- "NumPy is fast" → write vectorized fancy indexing
- Modular arithmetic on indices is not associative under NumPy's order-of-write rules
- L1 is only 384KB — feels small enough that copies don't matter (until they happen 10⁶ times)

**How to avoid:**
- Allocate `self.nest_l1 = np.zeros((NUM_NESTS, SPUS_PER_NEST, L1_SIZE), dtype=np.uint8)` as one contiguous buffer.
- Always slice: `self.nest_l1[n, s, off:off+2]` — slices are views, not copies.
- For modular addressing across L1 boundary, split into two slices manually: if `off + 2 > L1_SIZE`, write `[off:L1_SIZE]` and `[0:wrap_remainder]` separately.
- Use `arr.flags.writeable` and `arr.base is None` (None means it owns data) to assert no-copy in tests.

**Warning signs:**
- Writes "succeed" but reads show old data
- Tests pass standalone but fail in sequence (state didn't persist)
- `arr.flags.owndata` is `True` after a slice (indicates copy)

**Phase to address:**
GTX-CORE-01. Add a memory-layout invariant test: assert all L1 slices are views (`base is not None`).

---

### Pitfall 14: Wheel Bundling — .elf Firmware Asset Bloat

**What goes wrong:** [speculative]
GTX-FW-01 requires bundling firmware .elf regression assets. Each .elf is ~100KB-1MB; `run_tests_n1s16.sh` and `run_llext_tests.sh` sum to many tens. Naively including all in `package_data` produces a wheel >100MB, which:
- Slows `pip install` significantly
- Breaks PyPI 100MB hard limit on a single project version (without wheel splitting)
- Forces every user to download regression data they may never run

**Why it happens:**
- "Just bundle the tests" — easiest path
- `MANIFEST.in` glob `*.elf` sweeps everything in
- Test fixtures and runtime assets get conflated

**How to avoid:**
- Split: production wheel `spike` (no .elf) + optional `spike[test]` extra that fetches assets via post-install hook or separate `spike-tests` package.
- Use `git lfs` or external blob storage for .elf, fetched lazily by a `python -m riscv.gtx.tests.fetch` command.
- For CI: gitignore .elf files, generate them as part of the test pipeline from compiled firmware sources (if reproducible).
- Wheel size budget: track via `du -sh dist/*.whl` in CI; alert if >50MB.

**Warning signs:**
- `pip install spike` takes >30 seconds on fast network
- Wheel size > 50MB
- PyPI rejects upload

**Phase to address:**
GTX-PKG-01. Decide bundling strategy at package skeleton time, before adding any .elf to the repo.

---

### Pitfall 15: Spike Upstream API Drift

**What goes wrong:** [from .planning/codebase/CONCERNS.md lines 41-56]
Recent commits show frequent upstream Spike bumps (`5d4348e bumped upstream spike to 20feb9c2`, `42c1ebb bumped upstream spike to 591cff16`). Each bump has touched `riscv_extension.cc` / `py_module.cc` (up to 16 lines per commit). `rocc_insn_t` layout, `extension_t` virtual list, or `processor_t::get_state()` signature could change without notice.

A Python port that hard-codes assumptions about `insn.rs1` field semantics, `XPR` access, or relies on internal Spike state may break silently when upstream is bumped.

**Why it happens:**
- Spike has no public stable API contract
- Bindings layer hides the drift; Python users don't notice until something subtly breaks
- pyspike has no version-pinning at runtime (per concerns doc, not yet implemented)

**How to avoid:**
- Pin the GTX Python port to a specific spike commit; declare in `vendor/SPIKE_VERSION` file.
- At module import, assert spike's `__version__` (commit hash exposed via setup.py local_scheme) matches expected. Loud failure beats silent corruption.
- Maintain `vendor/gtx_cpp_reference/` snapshot (per PROJECT.md GTX-REF-01) as ground truth for "what the C++ extension looked like when port was validated."
- Build matrix: CI runs against current spike + N-1 commit at minimum to catch drift early.

**Warning signs:**
- Wheel builds succeed but tests fail after spike bump
- `rocc_insn_t.rs1` is suddenly missing or renamed
- New abstract virtual on `extension_t` that pybind11 trampoline doesn't override

**Phase to address:**
GTX-PKG-01. Add version assertion in `riscv/gtx/__init__.py`.

---

### Pitfall 16: Python 3.8 NumPy ABI Floor

**What goes wrong:** [speculative]
Wheel must support Python 3.8 (per PYS-EXT-06 in PROJECT.md). NumPy 2.0+ requires Python 3.9+. Many features (e.g. `np.exceptions.VisibleDeprecationWarning`, `dtype` semantics) shifted in NumPy 1.20→1.25→2.0. Bit-exact code that uses NumPy-2-only behavior breaks Python 3.8 wheels.

Specific traps:
- `np.float16(x).item()` returns a Python float — but the rounding may differ between NumPy versions
- `dtype='f2'` vs `np.float16` are not always equivalent in `astype`
- `np.add.reduce(arr, dtype=np.float32)` introduced behavior changes in 1.22

**Why it happens:**
- Author tests on local NumPy 1.26+ then ships
- cibuildwheel may pick a NumPy that the user's environment can't satisfy
- ABI matrix is non-trivial; "works on my Python 3.12" doesn't mean "works on 3.8"

**How to avoid:**
- Pin NumPy lower bound in `pyproject.toml`: `numpy>=1.20,<3.0` (verify lower bound that has stable FP16 behavior).
- Use `oldest-supported-numpy` build dependency in cibuildwheel config.
- Run CI matrix: Python 3.8 / 3.10 / 3.12 × NumPy oldest / current.
- Avoid NumPy-2-only API; review with `numpy --feature-detect` style audit.
- Implement FP16 conversion in pure-Python bit manipulation (Pitfall 8 prevention) — sidesteps NumPy-version-dependent rounding entirely.

**Warning signs:**
- Tests pass on Python 3.12 fail on 3.8
- Subnormal handling differs across NumPy versions
- `pip install spike` succeeds but `import` fails with NumPy ABI mismatch

**Phase to address:**
GTX-PKG-01 + GTX-VERIFY-01.

---

### Pitfall 17: Custom0 Reentrancy via CSR Write Callback

**What goes wrong:** [speculative]
If the Python NPU writes to a CSR that has a side-effect callback (or to MMIO that triggers another RoCC dispatch via SystemC-equivalent route), the chain `Spike → custom0 (Python) → write CSR → another extension → custom1 (Python)` can re-enter Python under the same GIL frame. pybind11's `PYBIND11_OVERRIDE` does NOT inherently re-acquire — recursive calls work, but `mxe_accum` or `is_ploop` mutations interleave in non-obvious ways.

For pyspike, the GTX_NPU does NOT use MMIO (PCIe/vfio paths are out of scope per PROJECT.md), so this is currently low risk. But if v2 introduces MMIO devices (PYS-EXT-07 supports them), reentrancy must be considered.

**Why it happens:**
- Threading-style mental model doesn't apply (single-threaded Python under GIL)
- "Reentrancy" feels like a C/C++ concern; Python authors don't expect it
- Side effects in CSR writes are invisible at the call site

**How to avoid:**
- Document NPU as **non-reentrant**: a single `custom0` invocation must complete before another fires. If firmware violates, raise.
- Add a guard flag `self._in_custom0 = False`; assert at entry, set/clear around body.
- Defer NPU-internal CSR writes to end of `custom0` rather than mid-op.
- For v1 (no MMIO), this is a documentation pitfall only.

**Warning signs:**
- Inconsistent state after specific firmware sequences
- `mxe_accum` corruption that disappears with single-SPU tests
- Stack overflow on deeply recursive firmware

**Phase to address:**
GTX-CORE-02 (document constraint), revisit in v2 when MMIO devices added.

---

### Pitfall 18: Negative Zero and NaN Comparison in Verification

**What goes wrong:** [verified in `verify.py:142-147`]
`verify.py` treats NaN as max distance (0xFFFF) and uses signed-magnitude ULP comparison (`verify.py:150-158`). A Python port that emits a different NaN bit pattern than C++ (`gtx_fp32_to_16` truncates NaN mantissa to upper 10 bits, `gtx_npu.h:71-72`) will fail `--ulp 1` even though both are "a NaN."

Negative zero: C++ `gtx_fp16_to_32(0x8000) = -0.0`; comparing `-0.0 == 0.0` is True in IEEE 754, but the bit patterns differ. The signed-magnitude `to_signed_mag` (verify.py:150) maps `0x8000 → 0`, so `-0.0` and `+0.0` have ULP distance 0 (correct). But a port that emits `+0.0` where C++ emits `-0.0` (or vice versa) passes ULP but fails exact byte compare.

**Why it happens:**
- `np.float32(-0.0) + 0.0 == 0.0` (positive)
- `np.where(arr == 0, 0, arr)` collapses negative zeros
- NaN payload propagation differs between numpy.exp(np.nan) and std::exp(NaN)

**How to avoid:**
- Implement custom `fp32_to_fp16` that explicitly preserves NaN payloads matching C++ (`gtx_npu.h:71`)
- Test: `fp32_to_16(np.float32('-0.0')) == 0x8000` and `fp32_to_16(np.float32('nan')) == 0x7E00` (or whatever C++ emits — verify against actual binary)
- Don't normalize zeros mid-pipeline (`(arr + 0)` or `arr * 1.0` may flip sign of -0.0)

**Warning signs:**
- `--ulp 1` passes but exact byte compare fails on a small fraction of words
- Sign bit differs but otherwise zero
- NaN propagation regression after a math library change

**Phase to address:**
GTX-CORE-01 (FP16 conversion helpers) — explicit -0.0 and NaN test cases.

---

### Pitfall 19: SPR Write Ordering vs Dispatch Fire

**What goes wrong:** [speculative, partial verification]
Gem5-simplified DISPATCH operations (funct7=0x04..0x07) read operands from GSPR. The firmware sequence is:
```
WRSPR GSPR_GTX_OPERAND1, value1
WRSPR GSPR_GTX_OPERAND2, value2
WRSPR GSPR_GTX_OPCODE, op
DISPATCH_VEC      # reads GSPR
```

If the Python port executes any of WRSPR steps lazily (deferred until DISPATCH), or batches GSPR writes through a write-combining scheme, the DISPATCH will see stale operands. The C++ does not appear to defer (per `gtx_npu_spr.cc` direct write pattern), but a "smart" Python port might.

**Why it happens:**
- "Optimization": batch CSR writes
- Race conditions don't exist in single-threaded sim, but author may add caching
- The dispatch handler is the only consumer — easy to assume writes are eager

**How to avoid:**
- WRSPR must be eager. No deferred queue. Each WRSPR finalizes before returning.
- DISPATCH_* must read GSPR fresh on entry, no cached operand snapshot.
- Add a regression: WRSPR(OP1=A) → DISPATCH(reads OP1) → WRSPR(OP1=B) → DISPATCH(reads OP1) — both dispatches must see their respective values.

**Warning signs:**
- Tests pass when DISPATCH fires immediately after single WRSPR
- Multi-WRSPR sequences produce wrong results
- "Reorderable" looking firmware fails

**Phase to address:**
GTX-SPR-01 + GTX-DISP-01.

---

### Pitfall 20: ULP Pass + Atol Fail Reporting Confusion

**What goes wrong:** [verified in `verify.py:251-254`]
`verify.py` treats a value as "within tolerance" if **either** `ulp_dist <= ulp_tol` **OR** `abs_diff <= atol`. So `--ulp 1 --atol 0.001` passes if either constraint holds. A test reporter that says "PASS" doesn't distinguish between:
- All values bit-exact match (the ideal)
- All within 1 ULP (acceptable rounding diff)
- All within 0.001 absolute (looser — for example, 0x0001 vs 0x0002 ULP=1, but for tiny subnormals abs_diff < 0.001 covers many ULPs)

A regression report claiming "bit-exact" when it actually passed via `atol` for several values is misleading; degradation can creep in over time.

**Why it happens:**
- `verify.py` reports "PASS" without surfacing how many values were exact vs tolerance-saved
- Actually, it does surface this (`stats['exact_matches']` vs `stats['within_tolerance']`, lines 322-325) but a CI summary parser may collapse to just PASS/FAIL.
- Author trusts "PASS" implicitly

**How to avoid:**
- For acceptance gates, require `mismatches == 0 AND within_tolerance == 0` (i.e. **all** values exact-match). Surface this as a separate "strict" mode.
- Track regression metric over time: ratio `exact_matches / total_fp16`; alert if it drops between commits.
- For each .elf regression, baseline the strict-pass status; investigate any regression to "tolerance-pass."

**Warning signs:**
- "PASS" with non-zero `within_tolerance` count
- Subtle drift across CI runs in the exact-match ratio
- Verification continues to pass after a known-incorrect change

**Phase to address:**
GTX-VERIFY-01. Wrap `verify.py` in a stricter assertion for CI.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use `np.float16` arithmetic directly (skip explicit FP16↔FP32 helpers) | 10× faster development | Bit-exactness diverges per NumPy version; subnormal/NaN edge cases break randomly | Never for production; OK for prototype-only spike |
| Single-mode DDR I/O (only LTR or only reversed) | Half the code | Half the regression suite breaks; must rewrite | Never — both modes are required for HW-sim compatibility |
| Cache `proc.get_state()` Python wrapper across calls | Saves marshalling | Use-after-GIL-frame, stale state | Never |
| Skip `mxe_accum` continuity (fresh zero every call) | Simpler implementation | All MMC chains break | Never |
| Bundle all .elf in main wheel | One-step install | 100MB+ wheel, PyPI rejection | Only for internal/dev wheel; never PyPI |
| `print()` debug instead of structured trace | Quick to add | Output diverges from C++ trace, hard to diff | Acceptable for early development; remove before GTX-VERIFY-01 |
| Pure Python loop over each FP16 element | Easy to write | 100× slower than vectorized | Acceptable for op correctness, refactor before perf gates |
| Skip `xs1=0` quirk handling | Code looks cleaner | Half of firmware misroutes | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| pybind11 `processor_t*` access | Cache `proc.get_state()` between RoCC calls | Re-acquire per call; staging is the only cache |
| `rocc_insn_t.rs1`/`.rs2` | Treat as register *value* | These are **register indices**; access value via `proc.get_state().XPR[insn.rs1]` |
| pybind11 `bits[14:12]` | Use as `funct3` discriminator | These are `{xd, xs1, xs2}` flags; reconstruct funct3 as `(xd<<2)\|(xs1<<1)\|xs2` only inside firmware MM/VEC handlers per gtx convention |
| Spike upstream bumps | Assume API stability | Pin to commit; assert at module init |
| C++ libgtx_npu.so reference | Use as runtime fallback | Python re-implementation IS the implementation; C++ is golden, not fallback |
| pyspike `riscv.isa.ROCC` base class | Override `__init__` without calling super | Always `super().__init__()` to register with extension |
| `verify.py` byte order | "Fix" the BE interpretation | Treat as black box — DDR dump format is implicitly BE-paired despite LE in-memory |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-element Python loop in `custom0` | Wall-time 100× C++ | Vectorize with NumPy, one big op per `custom0` call | Any vector length > 100 |
| pybind11 marshalling per register access | Profile shows 50%+ in pybind | Stage all operands once at top of `custom0` | Firmware with >10⁵ RoCC ops |
| FP32 promotion of FP16 array on every reduction | Memory bandwidth doubles | Keep FP32 accumulator persistent; only promote source views | Long vectors, reductions |
| Fancy indexing creating copies | OOM or slow on large L1 | Use slices, assert `base is not None` | L1 fully loaded (384KB × 4×16 = 24MB total per simulator) |
| Reset re-allocates all NumPy buffers | Cold-start 1+ second | `arr[:] = 0` instead of `arr = np.zeros(...)` | Tests that reset frequently |
| GIL-serialized RoCC + multi-processor sim | No parallelism | Document; consider `gil_scoped_release` if v2 needs it | Multi-core simulator (out of v1 scope) |

---

## Security Mistakes

(Domain-specific; security exposure is minimal — this is a deterministic functional simulator with no network surface.)

| Mistake | Risk | Prevention |
|---------|------|------------|
| `os.system()` for shelling out to C++ libgtx_npu.so for verification | Command injection if test paths are user-controlled | Use `subprocess.run([...], check=True, shell=False)` |
| Loading untrusted .elf via firmware regression | Spike emulates RV — minimal host risk, but pickled fixtures are unsafe | Never `pickle.load()` test fixtures; use hex/JSON only |
| `eval()` parsing of user-provided opcode encodings | Code injection in test harness | Use `int(s, 0)` and explicit opcode tables |

---

## "Looks Done But Isn't" Checklist

- [ ] **FP16 helpers**: Often missing **NaN payload preservation** — verify `fp32_to_fp16(np.float32('nan'))` matches `gtx_fp32_to_16` byte-for-byte
- [ ] **FP16 helpers**: Often missing **negative zero** — verify `fp16_to_fp32(0x8000)` is `-0.0` (not `+0.0`); test sign-bit propagation
- [ ] **FP16 helpers**: Often missing **subnormal correctness** — verify all 1024 subnormal values round-trip
- [ ] **L1/L0 access**: Often missing **byte order test** — explicit assertion that `0x3C00` writes as `[0x00, 0x3C]` little-endian
- [ ] **mxe_accum**: Often missing **chain regression** — `mm.s → mmc.s → mmc` end-to-end, not just isolated unit tests
- [ ] **Activation**: Often missing **direction assertion** — distinct ADDRA/ADDRR with different patterns, verify the right buffer is overwritten
- [ ] **DDR I/O**: Often missing **GTX_DDR_REVERSED round-trip** — both modes tested
- [ ] **Dispatch**: Often missing **xs1=0 quirk** — synthetic instruction with xs1=0 routes correctly
- [ ] **Dispatch**: Often missing **funct7 collision coverage** — both gem5-simplified (0x04-0x07) and ISS-full (0x00) tested side-by-side
- [ ] **Reset**: Often missing **sp init** — first-instruction `addi sp,sp,-16` survives without trap
- [ ] **WJOIN**: Often missing **GTX_NO_EXIT honoring** — env-var-driven termination
- [ ] **Verify**: Often missing **strict mode** — non-zero `within_tolerance` should raise in CI
- [ ] **Wheel**: Often missing **size budget** — `du -sh dist/*.whl` enforced in CI
- [ ] **Wheel**: Often missing **NumPy version matrix** — Python 3.8 × NumPy oldest tested
- [ ] **Spike pin**: Often missing **runtime assertion** — module init should compare expected vs actual spike commit hash

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Byte order wrong (Pitfall 1) | LOW | Single helper rewrite; re-run regression |
| Per-step FP16 cast (Pitfall 2) | MEDIUM | Refactor every reduction; touches VSUM/DOT/MM_O/MM_V/SOFTMAX/ESUM |
| `mxe_accum` continuity (Pitfall 3) | MEDIUM | State machine refactor; require chain regressions to verify |
| xs1=0 (Pitfall 4) | LOW | Replace `xs1` param with `XPR[insn.rs1]` reads in dispatchers |
| Encoding collision (Pitfall 5) | HIGH | Re-architect dispatch table; rebuild coverage matrix |
| WJOIN missing exit (Pitfall 6) | LOW | Add `SystemExit(0)` raise |
| Reset sp (Pitfall 7) | LOW | One line in `reset()` |
| NumPy float16 traps (Pitfall 8) | MEDIUM | Replace `np.float16` arithmetic with explicit fp16↔fp32 helpers throughout |
| Activation direction (Pitfall 9) | LOW | Add asymmetry table, fix per-op |
| DDR reversed mode (Pitfall 10) | MEDIUM | Implement both modes + round-trip test |
| Loop state machine (Pitfall 11) | HIGH | Trace-driven debug against C++; re-validate all 4 modes |
| GIL hot loop (Pitfall 12) | MEDIUM | Restructure `custom0` to amortize per-call cost |
| NumPy fancy indexing (Pitfall 13) | LOW | Slice instead; assert views |
| Wheel bloat (Pitfall 14) | MEDIUM | Split package, post-install fetch |
| Spike API drift (Pitfall 15) | HIGH | Re-bind whatever changed; re-validate full regression |
| Python 3.8 ABI (Pitfall 16) | MEDIUM | NumPy version pinning; CI matrix |
| Reentrancy (Pitfall 17) | LOW | Guard flag; doc constraint |
| -0.0 / NaN compare (Pitfall 18) | LOW | Custom fp16 helpers preserve sign and NaN |
| SPR write ordering (Pitfall 19) | LOW | Eager writes only |
| ULP/atol report confusion (Pitfall 20) | LOW | Strict mode wrapper |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. verify.py byte order | GTX-MEM-01 | Round-trip test: write 0x3C00 LE, dump, verify.py PASS |
| 2. FP16 cast regimen | GTX-MM-01, GTX-VEC-01 | `np.add.reduce(arr, dtype=np.float32)` invariant test |
| 3. mxe_accum continuity | GTX-MM-01 | mm.s→mmc.s→mmc chain regression |
| 4. xs1=0 quirk | GTX-CORE-02 | Synthetic insn with xs1=0 dispatches correctly |
| 5. Encoding collision | GTX-DISP-01 | Funct7×funct3 coverage matrix passes |
| 6. WJOIN exit | GTX-RST-01 | GTX_NO_EXIT=1 honored; default raises SystemExit |
| 7. Reset sp | GTX-RST-01 | `addi sp,sp,-16` survives reset |
| 8. NumPy float16 traps | GTX-CORE-01 | Round-trip: every fp16 value `f`: `f32_to_16(f16_to_32(f)) == f` |
| 9. Activation direction | GTX-ACT-01 | Distinct ADDRA/ADDRR per op; correct buffer overwritten |
| 10. DDR reversed mode | GTX-DMA-01 | Round-trip in both modes; HW-sim golden passes |
| 11. Loop state machine | GTX-CORE-02 | All 4 modes covered with synthesized custom1 sequences |
| 12. GIL hot loop | GTX-CORE-02 | cProfile of representative regression < 50% pybind overhead |
| 13. NumPy fancy indexing | GTX-CORE-01 | Memory-layout invariant: all L1 slices have `base is not None` |
| 14. Wheel size | GTX-PKG-01 | `du -sh dist/*.whl` < 50MB CI gate |
| 15. Spike API drift | GTX-PKG-01 | Module-init commit-hash assertion |
| 16. Python 3.8 ABI | GTX-PKG-01, GTX-VERIFY-01 | CI matrix Python 3.8 × oldest NumPy |
| 17. Reentrancy | GTX-CORE-02 | Guard flag asserts at custom0 entry |
| 18. -0.0 / NaN | GTX-CORE-01 | Explicit -0.0 and NaN payload tests |
| 19. SPR write ordering | GTX-SPR-01 | WRSPR(A)→DISPATCH→WRSPR(B)→DISPATCH sequence test |
| 20. ULP/atol confusion | GTX-VERIFY-01 | Strict-mode CI gate (`exact_matches == total`) |

---

## Sources

- `~/NIGHTLY/gtx_spike/gtx/CLAUDE.md` — explicit "Implementation Notes" section (FP16 byte order, DDR reversed mode, activation direction, VSUM precision, sp init, WJOIN exit)
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu.h` — RoCC encoding semantics, FP16↔FP32 helpers, ISS funct7 table
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_mm.cc` — gemm_core, mxe_accum continuity, firmware_mm_op funct3 reconstruction, xs1=0 reference
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_vec.cc` — VSUM/DOT FP32 accumulation pattern (lines 102-113)
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_act.cc` — Activation direction asymmetry (lines 35-42)
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_dma.cc` — DDR LTR/reversed mode (lines 480-599)
- `~/NIGHTLY/gtx_spike/gtx/verify.py` — ULP/atol semantics; big-endian FP16 interpretation (line 235)
- `/mnt/e/14_NIGHTLY/pyspike/.planning/codebase/CONCERNS.md` — pyspike binding-layer risks (GIL, lifetime, Spike API drift)
- `/mnt/e/14_NIGHTLY/pyspike/.planning/PROJECT.md` — Active requirements, scope, decisions

---
*Pitfalls research for: C++ FP16 RoCC NPU → Python (NumPy) port*
*Researched: 2026-05-04*
