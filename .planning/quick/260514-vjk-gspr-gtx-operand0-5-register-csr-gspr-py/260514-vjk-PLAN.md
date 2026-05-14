---
phase: 260514-vjk-quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/csr/gspr.py
  - src/main/python/riscv/gtx/unit/ins/encoding.py
  - src/main/python/riscv/gtx/npu.py
  - src/main/python/riscv/gtx/tloop_buffer.py
  - src/main/python/riscv/gtx/unit/ins/ops/act.py
  - src/main/python/riscv/gtx/unit/ins/ops/vec.py
  - tests/gtx/test_custom_dispatch_chain.py
autonomous: true
requirements:
  - VJK-01  # GSPR_GTX_OPERAND0..5 @csr registration in csr/gspr.py (source of truth)
  - VJK-02  # encoding.py re-export of bare-int address constants
  - VJK-03  # 5 callsite import realignment (npu/tloop_buffer/act/vec) + mm.py verify
  - VJK-04  # T-loop fast-path NameError regression test

must_haves:
  truths:
    - "csr/gspr.py 'GSPR' registry contains GSPR_GTX_OPERAND0..5 with addresses 0x000..0x005"
    - "encoding.GSPR_GTX_OPERAND0..5 are int constants derived from csr/gspr.py (single source of truth)"
    - "npu.py T-loop fast-path (line 238/240/264/265/269/270) resolves _GSPR_OP3 / _GSPR_OP5 without NameError"
    - "tloop_buffer.py imports GSPR_GTX_OPERAND3 / GSPR_GTX_OPERAND5 (no NameError on snapshot path)"
    - "act.py imports GSPR_GTX_OPERAND1/2/3 (no NameError on PReLU/GeLU/Tanh/Softmax/Pool kernels)"
    - "vec.py line 250 (npu.gspr[GSPR_GTX_OPERAND2] = rs2) resolves GSPR_GTX_OPERAND2 as int"
    - "ops/mm.py CSR_GSPR['GSPR_GTX_OPERAND3'] dict lookup returns the new Register at 0x003"
    - "existing 23 smoke+chain tests still pass; 1 new T-loop fast-path test passes -> 24 total"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/csr/gspr.py"
      provides: "6 @csr-decorated declarations for GSPR_GTX_OPERAND0..5 (addresses 0x000..0x005, width=64, RW, PIPE)"
      contains: "GSPR_GTX_OPERAND0"
    - path: "src/main/python/riscv/gtx/unit/ins/encoding.py"
      provides: "Re-exported int address constants GSPR_GTX_OPERAND0..5 derived from csr/gspr.py GSPR registry"
      contains: "GSPR_GTX_OPERAND0: int"
    - path: "src/main/python/riscv/gtx/npu.py"
      provides: "Import block (line 27-29) extended with GSPR_GTX_OPERAND3 as _GSPR_OP3, GSPR_GTX_OPERAND5 as _GSPR_OP5"
      contains: "_GSPR_OP3"
    - path: "src/main/python/riscv/gtx/tloop_buffer.py"
      provides: "Uncommented import of GSPR_GTX_OPERAND3 / GSPR_GTX_OPERAND5"
    - path: "src/main/python/riscv/gtx/unit/ins/ops/act.py"
      provides: "Uncommented import of GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3 from ..encoding"
    - path: "src/main/python/riscv/gtx/unit/ins/ops/vec.py"
      provides: "Import of GSPR_GTX_OPERAND2 (and GSPR_GTX_OPERAND3 if needed by line 223) from ..encoding"
    - path: "tests/gtx/test_custom_dispatch_chain.py"
      provides: "New test_tloop_fast_path_opset_no_nameerror — asserts custom0 T-loop fast-path OPSET path writes 0x003 without NameError"
      contains: "test_tloop_fast_path_opset_no_nameerror"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/ins/encoding.py"
      to: "src/main/python/riscv/gtx/unit/csr/gspr.py"
      via: "from ..csr.gspr import GSPR as _GSPR_REGS; addresses sourced from GSPR['GSPR_GTX_OPERAND*'].address"
      pattern: "from ..csr.gspr import GSPR as _GSPR_REGS"
    - from: "src/main/python/riscv/gtx/npu.py"
      to: "src/main/python/riscv/gtx/unit/ins/encoding.py"
      via: "from .unit.ins.encoding import GSPR_GTX_OPERAND3 as _GSPR_OP3, GSPR_GTX_OPERAND5 as _GSPR_OP5"
      pattern: "GSPR_GTX_OPERAND3 as _GSPR_OP3"
    - from: "src/main/python/riscv/gtx/unit/ins/ops/mm.py"
      to: "src/main/python/riscv/gtx/unit/csr/gspr.py"
      via: "CSR_GSPR['GSPR_GTX_OPERAND3'] dict lookup — auto-resolves once @csr decorator registers the entry"
      pattern: "CSR_GSPR\\['GSPR_GTX_OPERAND3'\\]"
---

<objective>
Restore the 6 GSPR_GTX_OPERAND0..5 registers that d6f73f9 "Architecture Refactoring"
silently dropped from `encoding.py`. Establish `csr/gspr.py` as the single source
of truth via `@csr` decoration, re-export the bare-int address aliases from
`encoding.py`, and realign 5 callsites whose imports were left commented or
unresolved.

Purpose: Eliminate the latent NameError / KeyError landmines at
`npu.py:238/240/264/265/269/270`, `tloop_buffer.py:35`, `act.py:329-461`,
`vec.py:250`, and `mm.py:251/280`. Today the bugs are dormant because the 23
existing tests do not enter the T-loop fast-path / activation handler paths;
production firmware will trigger them on the first OPSET-bearing tile.

Output:
  - 6 new @csr declarations in `csr/gspr.py` (PIPE 0x000..0x005, width=64, RW)
  - 10-line re-export block at the end of `encoding.py`
  - Updated imports in `npu.py`, `tloop_buffer.py`, `act.py`, `vec.py`
  - 1 new regression test asserting the T-loop fast-path OPSET path no
    longer raises NameError on `_GSPR_OP3` / `_GSPR_OP5`
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/quick/260514-vjk-gspr-gtx-operand0-5-register-csr-gspr-py/260514-vjk-CONTEXT.md

<!-- Locked decision summary (full text in CONTEXT.md):
  - csr/gspr.py is THE source of truth (insert 6 @csr blocks at top of "64-bit PIPE Registers" section, BEFORE @csr STACK_INFO@0x010).
  - Each block has a single `value = bits(0, 63)` field (only purpose: satisfy `@csr no-empty-fields` invariant; callsites use raw int).
  - encoding.py re-exports as module-level int constants derived from `GSPR['GSPR_GTX_OPERAND*'].address` — NOT hard-coded ints (Python is the single source of truth, vendor's gtx_params.h is informational).
  - mm.py uses dict lookup `CSR_GSPR['GSPR_GTX_OPERAND3']` — needs ZERO source change once csr/gspr.py registers the key (csr/__init__.py:CSR_GSPR is the PIPE-only view auto-built from GSPR registry).
  - vec.py mixes string-key and int-key access — leave the style mismatch alone (out of scope).
  - GSPR_GTX_OPCODE NOT in this task (no callsite uses it; deferred).
  - OverflowError at register_file.py:188 is a separate task. -->

<interfaces>
<!-- Key contracts the executor needs. Extracted from codebase. -->

From src/main/python/riscv/gtx/unit/csr/register.py:
```python
def make_csr(registry: Dict[str, "Register"]) -> Callable[..., Callable[[type], "Register"]]
def bits(start: int, end: Optional[int] = None, value: Optional[int] = None) -> _Bits

class Register:
    # @property name, address, width, rw_type, bus_type, fields
    # __index__ -> int returns address (so Register can be used as int key into tensors)
```

The @csr decorator (in csr/gspr.py as `csr = make_csr(GSPR)`) signature:
```python
@csr(name="GSPR_GTX_OPERAND0", address=0x000, width=64, rw_type="RW")  # bus_type defaults to BusType.PIPE
class GSPR_GTX_OPERAND0:
    value = bits(0, 63)   # MUST have at least one bits() field or @csr raises ValueError
```
Decorator raises `ValueError("@csr {name}: no bits() fields declared")` if no
`bits()` attrs found, and raises `ValueError("duplicate registry key ...")` if
`name` already in registry. Adding `value = bits(0, 63)` is the minimal way to
satisfy "at least one field".

From src/main/python/riscv/gtx/unit/csr/__init__.py:
```python
from .gspr import GSPR             # raw registry (PIPE + APB)
CSR_GSPR = {name: reg for name, reg in GSPR.items() if reg.bus_type is BusType.PIPE}
```
So registering with default `bus_type=BusType.PIPE` ALREADY makes
`CSR_GSPR['GSPR_GTX_OPERAND3']` resolve — that is why mm.py needs no edit.

From src/main/python/riscv/gtx/unit/ins/encoding.py:
```python
# Today: NO GSPR_GTX_OPERAND* symbols at module level (deleted by d6f73f9).
# After this plan: 6 int constants re-exported from csr/gspr.py at end of file.
```
</interfaces>

@src/main/python/riscv/gtx/unit/csr/gspr.py
@src/main/python/riscv/gtx/unit/csr/register.py
@src/main/python/riscv/gtx/unit/ins/encoding.py
@src/main/python/riscv/gtx/npu.py
@src/main/python/riscv/gtx/tloop_buffer.py
@src/main/python/riscv/gtx/unit/ins/ops/act.py
@src/main/python/riscv/gtx/unit/ins/ops/vec.py
@tests/gtx/test_custom_dispatch_chain.py
@tests/gtx/conftest.py
@tests/gtx/_mocks.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Register GSPR_GTX_OPERAND0..5 in csr/gspr.py and re-export from encoding.py</name>
  <files>
    src/main/python/riscv/gtx/unit/csr/gspr.py
    src/main/python/riscv/gtx/unit/ins/encoding.py
  </files>
  <action>
**Step 1 — csr/gspr.py: insert 6 @csr declarations.**

Location: directly under the `# 64-bit PIPE Registers` section header
(currently line 26), BEFORE the existing `@csr(name="STACK_INFO", address=0x010, ...)`
block at line 28. Keep one blank line between each declaration; precede the
6-block group with a one-line comment header:

```python
# Operand staging slots (OPSET writes 0x003 on slot=0, 0x005 on slot=1;
# see ops/spr.py:opset). Single bits(0, 63) field satisfies @csr's
# no-empty-fields invariant — callsites use raw int access.
@csr(name="GSPR_GTX_OPERAND0", address=0x000, width=64, rw_type="RW")
class GSPR_GTX_OPERAND0:
    value = bits(0, 63)


@csr(name="GSPR_GTX_OPERAND1", address=0x001, width=64, rw_type="RW")
class GSPR_GTX_OPERAND1:
    value = bits(0, 63)


@csr(name="GSPR_GTX_OPERAND2", address=0x002, width=64, rw_type="RW")
class GSPR_GTX_OPERAND2:
    value = bits(0, 63)


@csr(name="GSPR_GTX_OPERAND3", address=0x003, width=64, rw_type="RW")
class GSPR_GTX_OPERAND3:
    value = bits(0, 63)


@csr(name="GSPR_GTX_OPERAND4", address=0x004, width=64, rw_type="RW")
class GSPR_GTX_OPERAND4:
    value = bits(0, 63)


@csr(name="GSPR_GTX_OPERAND5", address=0x005, width=64, rw_type="RW")
class GSPR_GTX_OPERAND5:
    value = bits(0, 63)


```

Defaults (bus_type=BusType.PIPE) are correct — DO NOT pass an explicit
`bus_type` argument. Verbatim per CONTEXT.md D-01 (locked).

**Step 2 — encoding.py: append re-export block at end of file.**

Append (after the last existing constant, currently `WARP_F3_JOIN: int = 0b101`
at line 295) with two leading blank lines:

```python


# ============================================================================
# GSPR address constants — re-exported from csr/gspr.py for raw int access
# (bare-int access patterns: npu.py T-loop fast-path, tloop_buffer.py,
# act.py, vec.py). MM uses CSR_GSPR['GSPR_GTX_OPERAND3'] dict lookup directly.
# ============================================================================
from ..csr.gspr import GSPR as _GSPR_REGS
GSPR_GTX_OPERAND0: int = _GSPR_REGS['GSPR_GTX_OPERAND0'].address
GSPR_GTX_OPERAND1: int = _GSPR_REGS['GSPR_GTX_OPERAND1'].address
GSPR_GTX_OPERAND2: int = _GSPR_REGS['GSPR_GTX_OPERAND2'].address
GSPR_GTX_OPERAND3: int = _GSPR_REGS['GSPR_GTX_OPERAND3'].address
GSPR_GTX_OPERAND4: int = _GSPR_REGS['GSPR_GTX_OPERAND4'].address
GSPR_GTX_OPERAND5: int = _GSPR_REGS['GSPR_GTX_OPERAND5'].address
```

No `__all__` update needed (encoding.py has no `__all__`; module-level public
symbols are auto-exported). Relative import `..csr.gspr` is correct because
encoding.py lives at `riscv/gtx/unit/ins/encoding.py` and gspr.py at
`riscv/gtx/unit/csr/gspr.py`.

**Implementation order matters**: csr/gspr.py FIRST (so its @csr decorators
populate `GSPR` at import time), THEN encoding.py (which imports `GSPR`).

**Avoid:**
  - Do NOT hard-code `GSPR_GTX_OPERAND0: int = 0x000` — that violates the
    single-source-of-truth decision (CONTEXT D-02).
  - Do NOT add `GSPR_GTX_OPCODE` — out of scope (CONTEXT D-OutOfScope).
  - Do NOT add explicit `bus_type=BusType.PIPE` — the default already is PIPE,
    matches surrounding STACK_INFO/STACK_SAVE style.
  - Do NOT change widths from 64 or rw_type from "RW" — vendor `gtx_params.h:36-44`
    uses 64-bit RW slots, and OPSET handler in ops/spr.py expects 64-bit storage.

**Commit (Task 1 atomic):**
```
fix(gtx): restore GSPR_GTX_OPERAND0..5 register declarations

Re-registers the 6 operand staging slots (0x000..0x005) in csr/gspr.py
that d6f73f9 silently dropped. encoding.py re-exports the bare-int
address aliases for callsites that index gspr.tensor by raw int.
csr/gspr.py is the single source of truth; vendor gtx_params.h:36-44
is informational. CSR_GSPR['GSPR_GTX_OPERAND3'] (used by mm.py) now
auto-resolves through the PIPE-filter view in csr/__init__.py.
```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; uv run python -c "from riscv.gtx.unit.csr.gspr import GSPR; from riscv.gtx.unit.csr import CSR_GSPR; from riscv.gtx.unit.ins.encoding import GSPR_GTX_OPERAND0, GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND4, GSPR_GTX_OPERAND5; assert GSPR_GTX_OPERAND0 == 0x000 and GSPR_GTX_OPERAND5 == 0x005; assert CSR_GSPR['GSPR_GTX_OPERAND3'].address == 0x003 and CSR_GSPR['GSPR_GTX_OPERAND3'].rw_type == 'RW' and CSR_GSPR['GSPR_GTX_OPERAND3'].width == 64; print('OK: 6 registers registered and re-exported, dict lookup works')" &amp;&amp; uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v --tb=short 2&gt;&amp;1 | tail -30</automated>
  </verify>
  <done>
    1. `csr/gspr.py` contains 6 @csr-decorated classes for GSPR_GTX_OPERAND0..5,
       each at addresses 0x000..0x005, width=64, rw_type="RW", with the comment
       header explaining OPSET slot semantics.
    2. `encoding.py` ends with the re-export block; `GSPR_GTX_OPERAND0..5` are
       module-level int attributes derived from `GSPR[...].address`.
    3. The verify python one-liner prints `OK: 6 registers registered and
       re-exported, dict lookup works`.
    4. The 23 existing tests in test_fsm_smoke + test_custom0_smoke +
       test_csr_registry_chain + test_custom_dispatch_chain ALL still pass.
       (Task 1 does not unblock the T-loop NameError yet — that requires Task 2 —
       but it MUST NOT regress existing passes. The current 23 tests don't enter
       the T-loop fast-path, so the encoding-only re-export commits cleanly.)
    5. Task 1 atomic commit landed with message `fix(gtx): restore GSPR_GTX_OPERAND0..5
       register declarations`.
  </done>
</task>

<task type="auto">
  <name>Task 2: Realign 5 callsite imports + append T-loop fast-path NameError regression test</name>
  <files>
    src/main/python/riscv/gtx/npu.py
    src/main/python/riscv/gtx/tloop_buffer.py
    src/main/python/riscv/gtx/unit/ins/ops/act.py
    src/main/python/riscv/gtx/unit/ins/ops/vec.py
    tests/gtx/test_custom_dispatch_chain.py
  </files>
  <action>
**Step 1 — npu.py (line 27-29): extend the existing import block.**

Current state:
```python
from .unit.ins.encoding import (
    GTX_ISS_F7_OPSET,
)
```

Replace with:
```python
from .unit.ins.encoding import (
    GTX_ISS_F7_OPSET,
    GSPR_GTX_OPERAND3 as _GSPR_OP3,
    GSPR_GTX_OPERAND5 as _GSPR_OP5,
)
```

This resolves the `_GSPR_OP3` / `_GSPR_OP5` references at npu.py
lines 238, 240, 264, 265, 269, 270 (T-loop fast-path OPSET and bufferable
branches).

**Step 2 — tloop_buffer.py (line 35): uncomment.**

Current state:
```python
# from .unit.ins.encoding import GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND5
```

Change to:
```python
from .unit.ins.encoding import GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND5
```

(Just strip the leading `# ` — preserve trailing whitespace / blank lines.)

**Step 3 — act.py (line 29-41 import block): uncomment lines 30-32 and re-export 1.**

The current state has 3 commented lines:
```python
from ..encoding import (
    # ACT_OPS_REVERSED,
    # GSPR_GTX_OPCODE,
    # GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    GTX_ACT_ESUM, GTX_ACT_GELU, GTX_ACT_PRELU,
    ...
)
```

Uncomment ONLY the `GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,`
line (line 32). Leave `# ACT_OPS_REVERSED,` and `# GSPR_GTX_OPCODE,` commented
(out of scope per CONTEXT D-OutOfScope). Resulting block:

```python
from ..encoding import (
    # ACT_OPS_REVERSED,
    # GSPR_GTX_OPCODE,
    GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    GTX_ACT_ESUM, GTX_ACT_GELU, GTX_ACT_PRELU,
    ...
)
```

This unblocks the 9 usage sites at act.py lines 329, 334, 339, 356, 363, 383,
388, 425, 428, 461.

**Step 4 — vec.py (line 24-28 import block): add `GSPR_GTX_OPERAND2`.**

Current state:
```python
from ..encoding import (
    GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP, GTX_F7_VEC_DOT_SUM,
    GTX_F7_VEC_MATH, GTX_F7_VEC_ROUND, GTX_F7_VEC_SASMD, GTX_F7_VEC_SIGN,
    GTX_VEC_ADD, GTX_VEC_DIV, GTX_VEC_MUL, GTX_VEC_SUB,
)
```

vec.py uses string-key `npu.gspr.get("GSPR_GTX_OPERAND3", ...)` at lines
198/212 (works via @csr registration — no import change needed for those), but
ALSO uses raw-int `npu.gspr[GSPR_GTX_OPERAND2] = rs2` at line 250 AND
`npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF)` at line 223. Add both int-key
symbols to the import. Replace block with:

```python
from ..encoding import (
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP, GTX_F7_VEC_DOT_SUM,
    GTX_F7_VEC_MATH, GTX_F7_VEC_ROUND, GTX_F7_VEC_SASMD, GTX_F7_VEC_SIGN,
    GTX_VEC_ADD, GTX_VEC_DIV, GTX_VEC_MUL, GTX_VEC_SUB,
)
```

NOTE on style: CONTEXT explicitly says "string-vs-int key in vec.py stays as-is"
— do NOT convert lines 198/212 string-key access to int-key. Only fix the missing
int-key imports (line 223 and 250 need GSPR_GTX_OPERAND3 and GSPR_GTX_OPERAND2
respectively as Python names).

**Step 5 — mm.py: verify, no edit.**

Read mm.py around lines 251 and 280 and confirm both sites use the dict-lookup
pattern `CSR_GSPR['GSPR_GTX_OPERAND3']` (already imports `CSR_GSPR` at line 35).
DO NOT modify the file. The @csr registration in Task 1 makes this lookup work
automatically.

**Step 6 — tests/gtx/test_custom_dispatch_chain.py: append new test.**

Append at end of file (after the existing `test_end_to_end_custom0_and_custom1_return_int`
at line 156, with one blank line before):

```python


def test_tloop_fast_path_opset_no_nameerror(gtx_npu, mock_proc, dummy_insn):
    """custom0 T-loop fast-path inline OPSET no longer raises NameError on
    _GSPR_OP3 / _GSPR_OP5 (260514-vjk: GSPR_GTX_OPERAND restored).

    Drives the npu.py:236-241 fast-path: warp.is_tloop=True + _tloop_buf=[]
    + funct=GTX_ISS_F7_OPSET + xs1[insn.rs1] LSB=0 routes to gspr_tensor[_GSPR_OP3]
    write. Pre-d6f73f9 this raised NameError because _GSPR_OP3 was unbound.
    """
    gtx_npu.warp.is_tloop = True
    gtx_npu._tloop_buf = []
    mock_proc.state.XPR.write(1, 0)        # rs1 LSB=0 -> slot=0 -> OPERAND3 (0x003)
    mock_proc.state.XPR.write(2, 0xCAFE)   # rs2 -> value to stage
    dummy_insn.funct = GTX_ISS_F7_OPSET
    dummy_insn.rs1, dummy_insn.rs2 = 1, 2

    rc = gtx_npu.custom0(mock_proc, dummy_insn, 0, 0)

    assert rc == 0
    assert int(gtx_npu.gspr.tensor[0x003]) == 0xCAFE
```

This test:
  - Enters the npu.py T-loop fast-path (lines 226-241) — would have raised
    NameError on `_GSPR_OP3` before Task 2 fixes the import.
  - Verifies the staging slot (raw int 0x003 == GSPR_GTX_OPERAND3 address) holds
    the expected value via `gtx_npu.gspr.tensor[0x003]`.
  - Does NOT hit the bufferable-handler branch at lines 256-271 (handler is None
    for the OPSET fast-path return at line 241), so the test stays minimal.

**Avoid:**
  - Do NOT refactor vec.py string-key access at lines 198/212 to int-key
    (CONTEXT D-OutOfScope, "코딩 스타일 정합은 별도 task").
  - Do NOT uncomment `# ACT_OPS_REVERSED` or `# GSPR_GTX_OPCODE` in act.py.
  - Do NOT modify mm.py.
  - Do NOT change the test fixture (`conftest.py` / `_mocks.py`).
  - Do NOT reorder existing imports in any of the 4 source files — only add lines
    (npu/vec) or strip `# ` (tloop_buffer/act).

**Commit (Task 2 atomic):**
```
fix(gtx): realign 5 callsite imports for restored GSPR_GTX_OPERAND0..5

- npu.py:27-29: import GSPR_GTX_OPERAND3/5 as _GSPR_OP3/_GSPR_OP5
  (used in T-loop fast-path lines 238/240/264/265/269/270)
- tloop_buffer.py:35: uncomment OPERAND3/5 import
- act.py:32: uncomment OPERAND1/2/3 import (used in PReLU/GeLU/Tanh/Pool)
- vec.py:24: add OPERAND2/3 to existing import (line 223, 250 int-key access)
- mm.py: no edit (CSR_GSPR['GSPR_GTX_OPERAND3'] auto-resolves)

Adds test_tloop_fast_path_opset_no_nameerror to test_custom_dispatch_chain.py
asserting the T-loop fast-path OPSET branch writes gspr.tensor[0x003] = rs2
without raising NameError. 23 → 24 tests in the smoke+chain gate.
```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; uv run python -c "import importlib; m = importlib.import_module('riscv.gtx.npu'); assert m._GSPR_OP3 == 0x003 and m._GSPR_OP5 == 0x005; from riscv.gtx import tloop_buffer; assert tloop_buffer.GSPR_GTX_OPERAND3 == 0x003; from riscv.gtx.unit.ins.ops import act, vec; assert act.GSPR_GTX_OPERAND2 == 0x002 and vec.GSPR_GTX_OPERAND2 == 0x002; print('OK: all 5 callsites resolve GSPR_GTX_OPERAND symbols')" &amp;&amp; uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v --tb=short 2&gt;&amp;1 | tail -40 &amp;&amp; uv run pytest tests/gtx/test_custom_dispatch_chain.py::test_tloop_fast_path_opset_no_nameerror -v --tb=long 2&gt;&amp;1 | tail -15</automated>
  </verify>
  <done>
    1. `npu.py` line 27-29 import block extended with `GSPR_GTX_OPERAND3 as _GSPR_OP3, GSPR_GTX_OPERAND5 as _GSPR_OP5`.
    2. `tloop_buffer.py` line 35 uncommented (`from .unit.ins.encoding import GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND5`).
    3. `act.py` import block uncommented for `GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,` (other 2 commented lines preserved).
    4. `vec.py` import block extended with `GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,` on a new first line inside the parens.
    5. `mm.py` UNCHANGED (verified via `git diff src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns empty).
    6. `tests/gtx/test_custom_dispatch_chain.py` contains the new
       `test_tloop_fast_path_opset_no_nameerror` function at end of file.
    7. The verify python one-liner prints `OK: all 5 callsites resolve GSPR_GTX_OPERAND symbols`.
    8. Final gate: `uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v` reports **24 passed, 0 failed** (23 existing + 1 new T-loop fast-path test).
    9. Task 2 atomic commit landed with message `fix(gtx): realign 5 callsite imports for restored GSPR_GTX_OPERAND0..5`.
  </done>
</task>

</tasks>

<verification>
**Combined gate (after Task 2 commit):**

```bash
cd /mnt/e/14_NIGHTLY/pyspike && \
  uv run pytest tests/gtx/test_fsm_smoke.py \
                tests/gtx/test_custom0_smoke.py \
                tests/gtx/test_csr_registry_chain.py \
                tests/gtx/test_custom_dispatch_chain.py -v
```

Must report **24 passed, 0 failed** (= 23 existing + 1 new
`test_tloop_fast_path_opset_no_nameerror`).

**Manual spot checks (no `git diff` size budget violation):**

```bash
# csr/gspr.py +30 lines (6 @csr blocks * ~5 lines each)
git diff --stat src/main/python/riscv/gtx/unit/csr/gspr.py

# encoding.py +10 lines (re-export block)
git diff --stat src/main/python/riscv/gtx/unit/ins/encoding.py

# Total budget ≤ 60 lines (CONTEXT.md LOC budget)
git diff --stat src/main/python/riscv/gtx/ tests/gtx/test_custom_dispatch_chain.py | tail -1
```

LOC accounting (CONTEXT.md target):
  - csr/gspr.py: +30
  - encoding.py: +10
  - npu.py: +2
  - tloop_buffer.py: +0 net (1 line uncomment = 0 line delta)
  - act.py: +0 net (1 line uncomment)
  - vec.py: +1 (one new line in import block)
  - test_custom_dispatch_chain.py: +18 (one new test ~15 + 2 blanks + helpers)
  - **Total: ~60 ± 5 lines** — within budget.

**Negative invariants (regression guards):**
  - `git diff src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns empty.
  - `git diff src/main/python/riscv/gtx/unit/ins/ops/spr.py` returns empty.
  - `git diff src/main/python/riscv/gtx/unit/register_file.py` returns empty.
  - No new occurrences of `GSPR_GTX_OPCODE` anywhere (out of scope).
</verification>

<success_criteria>
1. `csr/gspr.py:GSPR` contains all 6 OPERAND register entries (addresses 0x000..0x005, width=64, rw_type="RW", bus_type=BusType.PIPE). Verified via:
   ```python
   from riscv.gtx.unit.csr.gspr import GSPR
   assert all(f'GSPR_GTX_OPERAND{i}' in GSPR for i in range(6))
   assert GSPR['GSPR_GTX_OPERAND3'].address == 0x003
   ```

2. `encoding.py` re-exports `GSPR_GTX_OPERAND0..5` as int attributes derived (not hard-coded) from csr/gspr.py. Verified via:
   ```python
   from riscv.gtx.unit.ins import encoding
   assert encoding.GSPR_GTX_OPERAND3 == 0x003
   ```

3. `CSR_GSPR['GSPR_GTX_OPERAND3']` dict lookup resolves to a Register at address 0x003. Verified via:
   ```python
   from riscv.gtx.unit.csr import CSR_GSPR
   assert CSR_GSPR['GSPR_GTX_OPERAND3'].address == 0x003
   ```

4. All 5 callsites import the needed symbols (no NameError on import):
   - `riscv.gtx.npu._GSPR_OP3 == 0x003` and `._GSPR_OP5 == 0x005`
   - `riscv.gtx.tloop_buffer.GSPR_GTX_OPERAND3 == 0x003`
   - `riscv.gtx.unit.ins.ops.act.GSPR_GTX_OPERAND2 == 0x002`
   - `riscv.gtx.unit.ins.ops.vec.GSPR_GTX_OPERAND2 == 0x002`

5. T-loop fast-path NameError regression closed: new test
   `test_tloop_fast_path_opset_no_nameerror` passes.

6. Existing 23-test green gate preserved → now 24/24 passing on:
   `tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py`.

7. Two atomic commits landed (one per task) with the exact subject lines specified
   in each task's `<action>` block.

8. Total LOC delta ≤ 60 across all 7 modified files (CONTEXT.md budget).

9. Out-of-scope items NOT touched:
   - No edits to `mm.py`, `spr.py`, `register_file.py`.
   - No `GSPR_GTX_OPCODE` re-export.
   - No vec.py string-key/int-key style unification.
   - No fix for `OverflowError @ register_file.py:188`.
</success_criteria>

<output>
After completion, write `.planning/quick/260514-vjk-gspr-gtx-operand0-5-register-csr-gspr-py/260514-vjk-SUMMARY.md`
following the standard quick SUMMARY template:

```markdown
# 260514-vjk SUMMARY — GSPR_GTX_OPERAND0..5 register 복원

**Phase:** quick / 260514-vjk
**Plans executed:** 1/1 (2 tasks atomic)
**Status:** [DONE | DEFERRED | BLOCKED]

## What Changed
- csr/gspr.py: +6 @csr blocks (GSPR_GTX_OPERAND0..5 @ 0x000..0x005, RW, 64-bit, PIPE)
- encoding.py: +10-line re-export block (bare-int aliases derived from csr/gspr.py)
- npu.py: import block extended (GSPR_GTX_OPERAND3/5 as _GSPR_OP3/_GSPR_OP5)
- tloop_buffer.py: line 35 uncommented
- act.py: line 32 uncommented (OPERAND1/2/3)
- vec.py: import block extended (OPERAND2/3)
- test_custom_dispatch_chain.py: +1 regression test (T-loop fast-path NameError guard)

## Test Gate
**Before:** 23/23 PASS (T-loop fast-path / activation / MM kernel paths latent NameError/KeyError)
**After:**  24/24 PASS (T-loop fast-path no-NameError invariant locked)

## Open Notes
- `GSPR_GTX_OPCODE` left out (no callsite needs it as of 2026-05-14)
- `OverflowError @ register_file.py:188` → separate quick task
- vec.py string-key vs int-key style mismatch → separate code-style task
- mm.py uses `CSR_GSPR['GSPR_GTX_OPERAND3']` dict pattern; act.py and vec.py mix int + string. No unification this task.

## Canonical Refs (final)
- csr/gspr.py:30-65 (new 6 @csr blocks)
- encoding.py:297-307 (new re-export block)
- npu.py:226-276 (T-loop fast-path, now NameError-free)
- tests/gtx/test_custom_dispatch_chain.py:160+ (new regression test)
```
</output>
