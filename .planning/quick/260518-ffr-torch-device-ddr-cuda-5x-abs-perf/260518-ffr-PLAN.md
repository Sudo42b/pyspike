---
phase: quick-260518-ffr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/memory.py
autonomous: true
requirements:
  - QFFR-01  # DDR device unified with config_params.DEVICE (transfer elimination)
  - QFFR-02  # ABS strict byte-exact preserved post-fix
  - QFFR-03  # ABS walltime recovers to <=150s (target near P8 Plan 04 baseline 95s)

must_haves:
  truths:
    - "DDR backing tensor lives on the same torch device as L0/L1/L2 scratchpads"
    - "ABS .elf strict regression PASSes byte-exact across all 96 tiles (196609 hex lines)"
    - "ABS walltime drops from regressed ~458s to <=150s on the cuda-available environment"
    - "GELU strict PASS preserved (no collateral regression in adjacent ops)"
    - "DDR↔scratchpad transfers no longer cross PCIe per access (single device residency)"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/memory.py"
      provides: "DDR scratchpad allocation aligned to config_params.DEVICE"
      contains: "_DDR_DEVICE = DEVICE"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/memory.py:_DDR_DEVICE"
      to: "src/main/python/riscv/gtx/config_params.py:DEVICE"
      via: "from ..config_params import DEVICE; _DDR_DEVICE = DEVICE"
      pattern: "_DDR_DEVICE\\s*=\\s*DEVICE"
    - from: "DDR.read/write byte slices"
      to: "L0/L1/L2 scratchpad tensors"
      via: "same device — no implicit .cpu()/.cuda() round-trip on hot path"
      pattern: "device check in DDR_MEMORY.write (line ~165) becomes a no-op when both sides live on DEVICE"
---

<objective>
DDR(_DDR_DEVICE)이 강제 CPU에 묶여 L0/L1/L2(=cuda) ↔ DDR(=cpu) 사이에서 매 DMA마다
PCIe round-trip을 발생시켜 ABS 회귀가 95s → 458s (5x)로 무너졌다.
`unit/memory.py:79`의 `_DDR_DEVICE = torch.device("cpu")` 한 줄을 `config_params.DEVICE`와
일치시켜 transfer를 제거한다.

Purpose: 정합성(byte-exact)을 깨지 않고 5x perf 회귀를 차단한다. CLAUDE.md "회귀 1개라도
깨지면 출하 보류" 원칙 하에 perf는 부수적이지만, 458s는 회귀 sweep 운용에 실질적 차단이라
사용자가 명시적으로 "5x 회귀 차단"을 fix 목표로 못박았다.

Output: `src/main/python/riscv/gtx/unit/memory.py` 1-2줄 수정 + ABS 재측정 통과 로그.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@src/main/python/riscv/gtx/config_params.py
@src/main/python/riscv/gtx/unit/memory.py

<interfaces>
<!-- Single-source-of-truth device contract — DO NOT redefine, import the symbol. -->

From src/main/python/riscv/gtx/config_params.py (lines 10-12):
```python
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Current (BROKEN) DDR allocation in src/main/python/riscv/gtx/unit/memory.py:79:
```python
_DDR_DEVICE = torch.device("cpu")   # <-- hard-coded cpu, the regression root
```

DDR allocation call sites (already use _DDR_DEVICE — no further edits needed there):
- memory.py:101  (DDR_MEMORY.__init__: initial torch.zeros(size, dtype=uint8, device=_DDR_DEVICE))
- memory.py:145  (DDR_MEMORY.ensure: grow path torch.zeros(new_size, dtype=uint8, device=_DDR_DEVICE))

Cross-device safety net already in place (no behavior change needed):
- memory.py:162-167 DDR_MEMORY.write — auto `data.to(self._bytes.device)` if mismatched
- memory.py:298-299 ddr_load_from_hex — auto `src.to(ddr_buf.device)` for frombuffer (always CPU) → DDR
- memory.py:318 ddr_save_to_hex — `.detach().cpu().contiguous().numpy()` for hex dump (DEVICE-agnostic)
- npu.py:354 `l2_src = self.mem.l2_byte(req.nest).cpu()` — explicit CPU snapshot for DMA inbound; still correct when L2 already on DEVICE (cpu() is no-op on CPU, cheap host snapshot on cuda)

Vendor input loader (DDR-side load):
- ddr_load_from_hex already cross-device safe — frombuffer(CPU) → .to(DDR.device) → write
- Pattern works whether DDR is cpu or cuda; the existing `if src.device != ddr_buf.device` guard handles both
</interfaces>

<docstring_note>
memory.py:36-37 docstring claims "DDR keeps its own grow-on-demand path on CPU — DDR is
the RISC-V system DRAM, not a scratchpad, and the CPU residence keeps host↔device traffic
confined to the DMA boundary."

memory.py:82-97 DDR_MEMORY class docstring elaborates the "DDR on CPU, scratchpads on
DEVICE" contract.

After this fix the comments become stale (DDR no longer on CPU when cuda is available).
Update them to reflect "DDR shares DEVICE with scratchpads (cuda when available);
host↔device boundary now lives only at hex-dump I/O via ddr_save_to_hex's explicit
.cpu() at memory.py:318." Do NOT delete the rationale — preserve the "DDR is system DRAM
not scratchpad" semantic note; just correct the device-placement claim.
</docstring_note>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Unify DDR device with config_params.DEVICE + scan for hidden cpu pinning</name>
  <files>src/main/python/riscv/gtx/unit/memory.py</files>
  <action>
**Surgical change (Option 1 per task spec — semantic clarity preserved):**

1. Import `DEVICE` (already imported at memory.py:10-17 in the multi-line config import — verify it is in the import list; add if missing).

2. Edit memory.py:79 from:
```python
_DDR_DEVICE = torch.device("cpu")
```
to:
```python
_DDR_DEVICE = DEVICE  # was hard-coded cpu; matched to config_params.DEVICE to
                      # eliminate DDR↔scratchpad PCIe round-trips (260518-ffr).
```

3. Update the two stale docstring claims to reflect the new device contract WITHOUT
   deleting the surrounding rationale:
   - memory.py:36-37 (module docstring): replace "DDR keeps its own grow-on-demand
     path on CPU — DDR is the RISC-V system DRAM, not a scratchpad, and the CPU
     residence keeps host↔device traffic confined to the DMA boundary." with a
     statement that DDR now shares `DEVICE` with scratchpads; host↔device boundary
     remains confined to hex-dump I/O via `ddr_save_to_hex`'s explicit `.cpu()`.
   - memory.py:82-97 (DDR_MEMORY class docstring): rewrite the "Device contract"
     block to say DDR lives on DEVICE alongside L0/L1/L2; the cross-device guards
     in `read`/`write`/`ensure` remain as defensive no-ops when both sides are on
     DEVICE, and are still active for any CPU-origin tensor (e.g., `torch.frombuffer`
     output in `ddr_load_from_hex`). Preserve the "DDR is system DRAM, not a
     scratchpad" semantic note as a separate sentence (still true).

4. **Hidden-cpu-pinning sweep** (must-do to avoid Whack-A-Mole):
```bash
# Any other torch.device("cpu") / device='cpu' inside gtx tree?
grep -rn 'torch\.device(["'\'']cpu["'\''])\|device\s*=\s*["'\'']cpu["'\'']' \
    src/main/python/riscv/gtx/ 2>/dev/null

# Any torch.zeros / torch.empty / torch.tensor in gtx that omit device= entirely?
# (These default to CPU and silently re-introduce the regression.)
grep -rEn 'torch\.(zeros|empty|tensor|frombuffer)\(' \
    src/main/python/riscv/gtx/ 2>/dev/null \
    | grep -v 'device=' \
    | grep -v -E '(test|_verify\.py|frombuffer\(bytearray)'
```

   For each hit on the hot path (npu.py:98/101/104 `_mxe_accum`/`_credit_ld`/`_credit_st`,
   ops/act.py constant-table builders, ops/mm.py `_mxe_accum` writes, ops/vec.py packed
   constant builders): decide whether the tensor flows into a DDR/scratchpad slice
   assignment or arithmetic with a DEVICE tensor. If yes → add `device=DEVICE`. If no
   (e.g., one-shot lookup table built at import time and used only via `.numpy()` /
   `.item()` extraction), leave it and add a one-line comment noting CPU residence is
   intentional.

   **Allowed exclusions (do NOT modify):**
   - `_verify.py:45-46` `torch.frombuffer(...)` — single FP16 element decode for ULP
     comparison; never touches DDR.
   - `ddr_load_from_hex` line 294 `torch.frombuffer(bytearray(chunk), dtype=torch.uint8)`
     — already followed by `.to(ddr_buf.device)` at line 299.
   - `tests/` tree — out of scope (this fix is production-only).

   **Hot-path hits that DO need `device=DEVICE` injection** (verify by reading each site
   and tracing the tensor consumer):
   - `npu.py:98` `_mxe_accum = torch.zeros((NEST, SPU, ...), dtype=torch.float32)` →
     MXE accumulation reads/writes go through `npu._mxe_accum[nest, spu] = torch.tensor(...)`
     in ops/mm.py:248/277. If the right-hand `torch.tensor(...)` is on CPU and `_mxe_accum`
     is on DEVICE → device-mismatch slowdown. Add `device=DEVICE` to BOTH the
     accumulator allocation AND the `torch.tensor` builder calls (mm.py:248, mm.py:277).
   - `npu.py:101/104` `_credit_ld`/`_credit_st` `torch.zeros` — small counter tensors,
     probably scalar-indexed; add `device=DEVICE` for consistency but low impact.
   - `ops/act.py:60/62/67` activation lookup tables — built once at import. Consumed by
     `.index_select` or direct indexing into DEVICE tensors? If yes → `device=DEVICE`.
     If pure host-side build with `.to(target.device)` at use site → leave as CPU
     (preserves import-time cost).
   - `ops/vec.py:111/117` packed-constant builders — read each call site. If the
     output is XOR/AND'd with a DEVICE tensor, add `device=DEVICE`.

   **Surgical rule:** Each `device=DEVICE` injection must be justified by a downstream
   arithmetic-or-assignment with another DEVICE tensor. Do NOT add `device=DEVICE`
   prophylactically to every torch.zeros call — that breaks the CPU-only fast paths
   (e.g., hex-dump byte buffers).

5. Run `python -c "from riscv.gtx.unit.memory import _DDR_DEVICE, DEVICE; print(_DDR_DEVICE, DEVICE); assert _DDR_DEVICE == DEVICE"` from within `uv run` venv to smoke-verify import + alignment.

Constraint reminders (CLAUDE.md):
- C++ 추가 코드 금지 — Python-only edit
- 디버그 print 자동 제거 금지 — do not remove any existing diagnostic prints
- 회귀 1개라도 깨지면 출하 보류 — Task 2 strict gate is BLOCKING
  </action>
  <verify>
<automated>uv run python -c "import sys; sys.path.insert(0, 'src/main/python'); from riscv.gtx.unit.memory import _DDR_DEVICE; from riscv.gtx.config_params import DEVICE; assert _DDR_DEVICE == DEVICE, f'DDR device {_DDR_DEVICE} != config DEVICE {DEVICE}'; print(f'PASS _DDR_DEVICE={_DDR_DEVICE} DEVICE={DEVICE}')"</automated>
  </verify>
  <done>
- memory.py:79 reads `_DDR_DEVICE = DEVICE` (with explanatory comment)
- `DEVICE` is in the `from ..config_params import (...)` block at memory.py:10-17
- Module + DDR_MEMORY class docstrings updated to reflect new device contract
- grep sweep documented in commit message (which sites were touched, which were intentionally skipped)
- Smoke import test passes (the verify command above returns PASS)
- No diff in any file outside `src/main/python/riscv/gtx/` (production-only fix)
  </done>
</task>

<task type="auto">
  <name>Task 2: ABS strict re-measurement + GELU regression guard</name>
  <files>(no source edits; measurement + log capture only)</files>
  <action>
Re-run the gated regressions and capture walltime/byte-exact status. This task is the
quantitative gate for the user's "5x 회귀 차단" success criterion.

**Step 1 — ABS strict byte-exact (PRIMARY gate, MUST PASS):**

```bash
# Time-and-log the strict ABS sweep. timeout=900 per task spec.
time uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' \
    --no-cov -v --timeout=900 2>&1 | tee /tmp/260518-ffr-abs-strict.log

# Extract walltime from pytest summary + `time` real value. Both should agree
# within ~5s. Record the `real` line.
```

**Pass criteria:**
- `1 passed` in pytest summary (strict byte-exact across 96 tiles × 196609 hex lines)
- `real` walltime <= 150s (success target; user-stated 5x-block threshold)
- `real` walltime <= 95s would be ideal (P8 Plan 04 baseline)

**If walltime > 150s:**
- DO NOT accept the fix as complete.
- Re-run the hidden-cpu-pinning grep from Task 1 — likely a missed `torch.zeros`/
  `torch.tensor` on the hot path is still defaulting to CPU.
- Add cProfile to narrow down: `uv run python -m cProfile -o /tmp/abs.prof -s cumulative -m pytest ... 2>&1 | head -60` and look for `.cpu()`, `.cuda()`, `_to_copy`, `cudaMemcpy` cumulative time.
- Surgical add `device=DEVICE` to the offending site, re-run from Step 1.

**If pytest FAIL (byte-exact broken):**
- This is the CLAUDE.md "회귀 1개라도 깨지면 출하 보류" gate. STOP.
- The likely culprit: a producer that previously emitted CPU tensors is now emitting
  DEVICE tensors, and a consumer was silently relying on `.cpu()` semantics (numpy
  comparison, scalar `.item()` extract, hex dump). Inspect first-mismatch line/byte
  from the test output, trace the writer.
- Revert Task 1 device alignment locally; investigate before committing.

**Step 2 — GELU regression guard (SECONDARY gate, MUST PASS):**

```bash
time uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' \
    --no-cov -v --timeout=900 2>&1 | tee /tmp/260518-ffr-gelu-strict.log
```

**Pass criteria:** `1 passed` (GELU was strict-PASS pre-fix per STATE.md line 40; must
remain PASS post-fix to confirm no collateral damage). Walltime is informational only —
no hard threshold for GELU.

**Step 3 — Record measurements in commit body:**

Commit message body must include a table:

```
Walltime measurements (260518-ffr):
| op   | pre-fix (regressed) | post-fix | target | status |
|------|---------------------|----------|--------|--------|
| ABS  | 458s                | XXs      | <=150s | PASS/FAIL |
| GELU | (informational)     | XXs      | —      | PASS/FAIL |

ABS strict byte-exact: PASS (96 tiles × 196609 hex lines)
GELU strict byte-exact: PASS
```

Out-of-scope reminder (do NOT chase in this task):
- numba `_jit` restoration
- cuda-bindings 12.9.6 removal from uv.lock
- 12개 `#!TODO` markers
  </action>
  <verify>
<automated>uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' --no-cov -v --timeout=900</automated>
  </verify>
  <done>
- ABS strict regression: 1 passed, walltime <= 150s recorded
- GELU strict regression: 1 passed (no collateral regression)
- /tmp/260518-ffr-abs-strict.log and /tmp/260518-ffr-gelu-strict.log captured
- Walltime delta (pre-fix 458s → post-fix XXs) quantified in commit body
- If walltime target missed: hidden-cpu-pinning sweep re-run + offending site fixed before declaring done
  </done>
</task>

</tasks>

<verification>
**Combined gate (both must hold):**

1. `_DDR_DEVICE == DEVICE` at runtime (Task 1 smoke import).
2. ABS strict byte-exact PASS at walltime <= 150s on the cuda-available environment
   (Task 2 primary gate).
3. GELU strict byte-exact PASS preserved (Task 2 secondary gate).
4. Diff scoped to `src/main/python/riscv/gtx/` only (no test edits, no config edits,
   no docs outside the affected files' docstrings).
5. No new runtime dependencies added (CLAUDE.md "NumPy 외부 추가 런타임 의존성 신규 도입 금지").

**Atomic commit format (quick mode):**

```
fix(gtx): unify _DDR_DEVICE with config_params.DEVICE (260518-ffr)

DDR was hard-coded to torch.device("cpu") at unit/memory.py:79, while
L0/L1/L2/gspr live on config_params.DEVICE (=cuda when available). After
cuda-bindings 12.9.6 appeared in venv (2026-05-18) and cuda.is_available()
flipped to True, every DDR↔scratchpad transfer crossed PCIe, regressing ABS
strict 95s → 458s (5x).

Fix: _DDR_DEVICE = DEVICE. Cross-device guards in DDR_MEMORY.{read,write,
ensure} remain as defensive no-ops for both same-device and CPU-origin
producers (torch.frombuffer in ddr_load_from_hex, hex-dump .cpu() in
ddr_save_to_hex).

Hidden-cpu-pinning sweep results: <list grep hits, which were patched (with
device=DEVICE injection at <files:lines>), which were intentionally left as
CPU (with one-line rationale comment)>.

Walltime: <ABS pre/post/target table from Task 2 Step 3>.

ABS strict byte-exact: PASS (96 tiles × 196609 hex lines).
GELU strict byte-exact: PASS.
```
</verification>

<success_criteria>
- ABS strict regression PASS at walltime <= 150s (user's "5x 회귀 차단" met)
- GELU strict regression PASS (no collateral)
- Single atomic commit in `src/main/python/riscv/gtx/` scope
- STATE.md updated separately (quick-mode state tracking) with 260518-ffr resolution
  noting: root cause (device split), fix (1-line + docstring sync + N-site grep sweep),
  measurement (pre/post walltime), and what remains out-of-scope (numba restore,
  cuda-bindings uv.lock cleanup, 12 #!TODO markers — all separate followups)
</success_criteria>

<output>
After completion, create `.planning/quick/260518-ffr-torch-device-ddr-cuda-5x-abs-perf/260518-ffr-SUMMARY.md`
capturing:
- Final diff (memory.py + any hidden-cpu-pinning sites patched)
- ABS pre/post walltime table
- GELU PASS confirmation
- Grep-sweep audit log (which sites checked, which patched, which intentionally left CPU + why)
- Followups list (numba, cuda-bindings, #!TODO markers)
</output>
