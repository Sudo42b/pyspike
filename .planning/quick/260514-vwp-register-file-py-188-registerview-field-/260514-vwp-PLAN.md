---
phase: 260514-vwp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/register_file.py
  - tests/gtx/test_csr_registry_chain.py
autonomous: true
requirements:
  - VWP-01  # 64-bit RegisterView field setter no longer OverflowErrors
  - VWP-02  # Regression guard tests pinned in test_csr_registry_chain.py

must_haves:
  truths:
    - "`lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF` executes without OverflowError"
    - "After the 64-bit broadcast write, every (NEST, SPU) slot at address 0x800 holds 0xCAFEBABEDEADBEEF (read back through signed-int64 reinterpretation)"
    - "Partial-field writes (e.g. THREAD_MASK.mask = 0xABCD) still preserve non-field bits — the existing 24-test green bar holds"
    - "Existing 24 tests across test_fsm_smoke / test_custom0_smoke / test_csr_registry_chain / test_custom_dispatch_chain remain PASS; total becomes 26"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/register_file.py"
      provides: "RegisterView.__setattr__ field branch with int64-safe shifted_mask"
      contains: "shifted_mask"
    - path: "tests/gtx/test_csr_registry_chain.py"
      provides: "Two regression tests guarding the 64-bit field setter fix"
      contains: "test_register_view_64bit_field_broadcast_write_no_overflow"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/register_file.py:180"
      to: "torch.Tensor.copy_ on self._tensor (no OverflowError)"
      via: "Python-side signed-int64 wrap of (mask << shift) before `~`"
      pattern: "shifted_mask = u64 - \\(1 << 64\\) if u64 >> 63 else u64"
    - from: "tests/gtx/test_csr_registry_chain.py (new test)"
      to: "src/main/python/riscv/gtx/unit/register_file.py:RegisterView.__setattr__"
      via: "gtx_npu.lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF"
      pattern: "lspr\\.SGPR0\\.gpr\\s*="
---

<objective>
Fix the `OverflowError: can't convert negative int to unsigned` raised by
`RegisterView.__setattr__` (register_file.py:188) whenever a field whose
width is 64 bits (e.g. `LSPR.SGPR0.gpr = bits(0, 63)`, every `GSPR_GTX_OPERANDn.value`)
is written through the attribute-style setter. Cause: `~(mask << shift)`
produces a Python arbitrary-precision negative integer (`-0x10000000000000000`)
that torch cannot cast back into int64 in the subsequent `self._tensor & ...`
expression. The fix is a Python-side reinterpretation of `(mask << shift)`
as a signed int64 so `~` stays in range. Append two regression tests to
lock the new behavior (64-bit broadcast write + partial-field preservation).

Purpose: unblocks every callsite that drives 64-bit CSR slots through the
RegisterView setter path (vec write-back, GSPR operand staging, SGPR scratch
writes) — currently each crashes the test harness on first use.
Output: a single atomic commit fixing the production setter and pinning
the regression with two new pytest cases (24 → 26 PASS in the smoke band).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/quick/260514-vwp-register-file-py-188-registerview-field-/260514-vwp-CONTEXT.md
@src/main/python/riscv/gtx/unit/register_file.py
@src/main/python/riscv/gtx/unit/csr/register.py
@src/main/python/riscv/gtx/unit/csr/lspr.py
@tests/gtx/test_csr_registry_chain.py
@tests/gtx/conftest.py

<interfaces>
<!-- Concrete signatures the executor needs verbatim. Do NOT explore. -->

From src/main/python/riscv/gtx/unit/csr/register.py (Field):
```python
class Field:
    __slots__ = ("name", "start", "end", "mask", "shift")
    # self.shift = start
    # self.mask  = (1 << (end - start + 1)) - 1
    # For SGPR0.gpr = bits(0, 63): shift=0, mask=0xFFFFFFFFFFFFFFFF.
```

From src/main/python/riscv/gtx/unit/csr/lspr.py (declared programmatically):
```python
SGPR0 = _declare_generated_csr(
    name="SGPR0", address=0x800, width=64, rw_type="RW",
    fields={"gpr": (0, 63)}, registry=LSPR,
)
# address & 0x3FF = 0x000  --- read back through gtx_npu.lspr.tensor[..., 0x000]
```

From src/main/python/riscv/gtx/unit/register_file.py:
```python
class RegisterView:
    __slots__ = ("_reg", "_tensor")
    def __setattr__(self, name: str, value: Any) -> None: ...
    # Returns None. The "field" branch is at lines 180-189 — the ONLY block
    # this plan changes. The "value" branch (176-178) and "_*" branch
    # (171-174) are OUT OF SCOPE.
```

From tests/gtx/conftest.py:
```python
@pytest.fixture(scope="function")
def gtx_npu(mock_proc):
    # Provides .gspr (1024,), .nspr (4,1024), .lspr (4,16,1024) RegisterFile views.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Fix RegisterView 64-bit field setter (signed-int64 wrap) + pin regression tests</name>
  <files>
    src/main/python/riscv/gtx/unit/register_file.py,
    tests/gtx/test_csr_registry_chain.py
  </files>
  <behavior>
    Test 1 — `test_register_view_64bit_field_broadcast_write_no_overflow`:
      - Use `gtx_npu` fixture.
      - Execute `gtx_npu.lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF`.
      - Must NOT raise `OverflowError` (currently raises before this plan).
      - SGPR0 lives at address 0x800; `0x800 & 0x3FF = 0x000`.
      - Read back via `gtx_npu.lspr.tensor[..., 0x000]` — shape `(4, 16)`.
      - The stored int64 is the signed reinterpretation of 0xCAFEBABEDEADBEEF:
        `signed = 0xCAFEBABEDEADBEEF - (1 << 64)` (top bit is set, so negative).
      - Assert every `(NEST, SPU)` slot equals that signed value.
      - Also assert that `int(gtx_npu.lspr.SGPR0.gpr & 0xFFFFFFFFFFFFFFFF) == 0xCAFEBABEDEADBEEF`
        on at least one `(nest, spu)` index (round-trip read through the
        `__getattr__` field path) — guards observable correctness, not just
        absence-of-exception.

    Test 2 — `test_register_view_partial_field_high_bits_preserves_low_bits`:
      - Use `gtx_npu` fixture.
      - THREAD_MASK is a 16-bit NSPR (`mask = bits(0, 15)`) seeded to 0xFFFF
        by vendor defaults (see existing `test_register_view_attribute_write_broadcasts_across_nests`).
      - Set `gtx_npu.nspr.THREAD_MASK.mask = 0xABCD`.
      - Assert `(gtx_npu.nspr.THREAD_MASK._tensor & 0xFFFF).tolist() == [0xABCD] * 4`.
      - This locks the existing partial-field path: the fix must NOT
        regress sub-64-bit fields.
  </behavior>
  <action>
    ## Step A — Apply the production fix (register_file.py)

    Edit `src/main/python/riscv/gtx/unit/register_file.py`, replacing the
    current field-branch body at lines 180-189 with the verbatim block from
    CONTEXT.md (locked decision — DO NOT paraphrase):

    Replace this exact block:
    ```python
            if name in self._reg.fields:
                field = self._reg.fields[name]
                # Bit manipulation via tensor ops
                mask = field.mask
                shift = field.shift

                # (tensor & ~(mask << shift)) | ((value & mask) << shift)
                new_val = torch.as_tensor(value, dtype=torch.int64) & mask
                self._tensor.copy_((self._tensor & ~(mask << shift)) | (new_val << shift))
                return
    ```

    With:
    ```python
            if name in self._reg.fields:
                field = self._reg.fields[name]
                mask = field.mask
                shift = field.shift

                # Reinterpret the shifted mask as a signed int64 to avoid
                # Python's arbitrary-precision negative result from
                # `~(mask << shift)` — torch cannot cast that back into
                # int64 (OverflowError). See CONTEXT.md root_cause.
                u64 = (mask << shift) & ((1 << 64) - 1)
                shifted_mask = u64 - (1 << 64) if u64 >> 63 else u64

                new_val = torch.as_tensor(value, dtype=torch.int64) & mask
                self._tensor.copy_((self._tensor & ~shifted_mask) | (new_val << shift))
                return
    ```

    Constraints:
      - DO NOT touch the `_`-prefixed branch (171-174), the `value` branch
        (176-178), or the trailing `super().__setattr__` (191).
      - DO NOT add helper functions, type annotations, or comments beyond
        the 3-line rationale above.
      - DO NOT reformat unrelated whitespace.
      - Net change ≤ 10 lines in this file.

    ## Step B — Append two regression tests (test_csr_registry_chain.py)

    Append at end of `tests/gtx/test_csr_registry_chain.py` (after the
    existing `test_find_by_address_pipe_missing_raises_keyerror`). Stay
    within ≤ 30 added lines total:

    ```python


    def test_register_view_64bit_field_broadcast_write_no_overflow(gtx_npu):
        """64-bit field writes (e.g. SGPR0.gpr) must not OverflowError.

        Pins the register_file.py:188 fix: ~(mask << shift) was leaking a
        Python negative bigint into torch's int64 cast path.
        """
        val_u64 = 0xCAFEBABEDEADBEEF
        gtx_npu.lspr.SGPR0.gpr = val_u64  # must not raise
        addr = 0x800 & 0x3FF
        stored = gtx_npu.lspr.tensor[..., addr]
        signed = val_u64 - (1 << 64)  # int64 reinterpretation (top bit set)
        assert stored.shape == (4, 16)
        assert (stored == signed).all().item()
        # Round-trip read through the field getter path must restore the
        # unsigned value.
        readback = int(gtx_npu.lspr[0][0].SGPR0.gpr) & 0xFFFFFFFFFFFFFFFF
        assert readback == val_u64


    def test_register_view_partial_field_high_bits_preserves_low_bits(gtx_npu):
        """Partial-field write must not regress after the 64-bit fix."""
        gtx_npu.nspr.THREAD_MASK.mask = 0xABCD
        masked = (gtx_npu.nspr.THREAD_MASK._tensor & 0xFFFF).tolist()
        assert masked == [0xABCD] * 4
    ```

    Notes for the executor:
      - The existing `test_register_view_attribute_write_broadcasts_across_nests`
        docstring (lines 80-87) mentions the OverflowError as out of scope.
        DO NOT edit that docstring — Karpathy §3 (surgical). The new tests
        speak for themselves.
      - `gtx_npu.lspr[0][0]` narrows the (NEST, SPU) dims and returns a
        scalar-tensor RegisterFile whose `SGPR0.gpr` is an int (per the
        existing `__getitem__` int-coercion path at register_file.py:76-77 /
        line 167 RegisterView getter).

    ## Step C — Run the verification command (see <verify>) and confirm 26 PASS.

    ## Step D — Commit atomically

    Stage exactly:
      - src/main/python/riscv/gtx/unit/register_file.py
      - tests/gtx/test_csr_registry_chain.py

    Commit message:
    ```
    fix(gtx): RegisterView 64-bit field setter OverflowError + regression test

    register_file.py:188 raised OverflowError when broadcasting a 64-bit
    field (e.g. SGPR0.gpr = bits(0, 63)) because ~(mask << shift) leaked
    a Python negative bigint into torch's int64 cast path. Reinterpret
    (mask << shift) as signed int64 before `~` so the operand stays in
    range. Pin the behavior with two regression tests in
    test_csr_registry_chain.py: 64-bit broadcast write + partial-field
    preservation.

    Quick task: 260514-vwp.
    ```

    Use a single atomic commit covering both files — production fix and
    its regression guard belong together.
  </action>
  <verify>
    <automated>uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v</automated>
    <!--
      MUST report exactly 26 PASS (24 pre-existing + 2 newly added).
      No failures, no errors, no xfails newly introduced. Any deviation =
      task NOT done — debug before committing.
    -->
  </verify>
  <done>
    - register_file.py field-branch uses `shifted_mask` (signed-int64 wrap)
      and does NOT call `~` on `(mask << shift)` directly.
    - `RegisterView.__setattr__` non-field branches (`_`-prefix, `value`, fallthrough)
      are byte-identical to pre-fix.
    - tests/gtx/test_csr_registry_chain.py contains the two new test
      functions, names exactly as listed in <behavior>.
    - `uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v`
      reports 26 passed.
    - One atomic commit with the message above; git status clean for these
      two paths.
  </done>
</task>

</tasks>

<verification>
- After commit, `git diff HEAD~1 -- src/main/python/riscv/gtx/unit/register_file.py`
  shows ≤ 10 net lines changed, confined to the field branch of
  `RegisterView.__setattr__`.
- `git diff HEAD~1 -- tests/gtx/test_csr_registry_chain.py` shows only
  appended lines (≤ 30) — no edits to existing tests.
- `uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v`
  → `26 passed`.
- Spot check: `uv run python -c "from riscv.gtx.npu import GtxNpu; n=GtxNpu(); n.reset(__import__('tests.gtx._mocks', fromlist=['MockProcessor']).MockProcessor()); n.lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF; print('ok')"`
  prints `ok` (no OverflowError).
</verification>

<success_criteria>
- The smoke test band goes from 24/24 PASS to 26/26 PASS with zero new
  failures, zero new warnings, zero regressed tests.
- `OverflowError: can't convert negative int to unsigned` no longer
  reproduces for any 64-bit field writethrough RegisterView.
- The production fix and the regression test live in a single atomic
  commit titled `fix(gtx): RegisterView 64-bit field setter OverflowError + regression test`.
- No file outside the two listed in `files_modified` is touched.
- The "Open Notes for Successor #1" from quick task 260514-ti0 is closed.
</success_criteria>

<output>
After completion, create `.planning/quick/260514-vwp-register-file-py-188-registerview-field-/260514-vwp-SUMMARY.md`
capturing:
  - Before/after of the changed block in register_file.py.
  - The two new test names + their assertion lines.
  - Final pytest line ("26 passed in N.NNs").
  - Commit SHA.
  - Explicit note that quick task 260514-ti0's "Open Notes #1" is now closed.
</output>
