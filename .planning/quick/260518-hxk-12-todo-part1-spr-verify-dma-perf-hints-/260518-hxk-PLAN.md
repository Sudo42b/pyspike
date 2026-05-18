---
phase: quick-260518-hxk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/ins/ops/spr.py
  - src/main/python/riscv/gtx/unit/context/dma.py
autonomous: true
requirements:
  - TODO-PART1-B-OPSET
  - TODO-PART1-B-CPSVR
  - TODO-PART1-B-MVSVR
  - TODO-PART1-C-CREDIT-LD-DOC
  - TODO-PART1-C-CREDIT-LD-VEC
  - TODO-PART1-C-CREDIT-ST-VEC
  - TODO-PART1-C-CREDIT-ST-CHK-GUARD
  - TODO-PART1-C-CREDIT-ST-CHK-VEC

must_haves:
  truths:
    - "spr.py opset/cpsvr/mvsvr handlers verified line-by-line against vendor C++; #!TODO 'verify' markers replaced with explicit `verified against vendor file:line` parity comments"
    - "dma.py credit_ld/credit_st S-loop branches use single-row vector ops (`row[:] += 1` / `row[:] -= 1`) instead of per-SPU Python for-loop"
    - "dma.py credit_st_chk negative-decrement TODO marker removed; existing outer (`total > 0`) + inner (`row[s] > 0`) guard documented in a single multi-line comment"
    - "dma.py credit_st_chk vector-op TODO marker resolved with rationale comment explaining `first-non-zero` semantics are intentionally NOT vectorisable"
    - "ABS .elf strict byte-exact regression remains PASS post-edit (baseline 94.82s, budget ≤ 100s)"
    - "GELU .elf strict byte-exact regression remains PASS post-edit"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/ins/ops/spr.py"
      provides: "Three SPR handlers (opset 0x4A, cpsvr 0x4B, mvsvr 0x4C) with explicit vendor parity comments and no remaining #!TODO markers in these three docstrings"
      contains: "verified against"
    - path: "src/main/python/riscv/gtx/unit/context/dma.py"
      provides: "credit.ld / credit.st / credit.st.chk handlers with all five `#!TODO` markers (lines 306, 314, 337, 463, 464) resolved"
      contains: "verified against"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/ins/ops/spr.py:opset"
      to: "vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc OPSET case"
      via: "side-by-side diff against rs1 slot / rs2 value semantics"
      pattern: "verified against vendor"
    - from: "src/main/python/riscv/gtx/unit/ins/ops/spr.py:cpsvr"
      to: "vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc CPSVR case"
      via: "side-by-side diff against L0 SVR replicate (bsz 0..3)"
      pattern: "verified against vendor"
    - from: "src/main/python/riscv/gtx/unit/ins/ops/spr.py:mvsvr"
      to: "vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc MVSVR case"
      via: "side-by-side diff against 32B copy+clear semantics"
      pattern: "verified against vendor"
    - from: "src/main/python/riscv/gtx/unit/context/dma.py:_credit_ld (S-loop)"
      to: "vector slice `npu._credit_ld[nest_id, :] += 1`"
      via: "single-row in-place add"
      pattern: "_credit_ld\\[nest_id, :\\] \\+= 1"
    - from: "src/main/python/riscv/gtx/unit/context/dma.py:_credit_st (S-loop)"
      to: "vector slice `npu._credit_st[nest_id, :] -= 1`"
      via: "single-row in-place sub"
      pattern: "_credit_st\\[nest_id, :\\] -= 1"
---

<objective>
12 TODO marker cleanup — part 1 of 2. Resolve 8 `#!TODO` markers
across two files:

- **Category B (3 markers, spr.py):** "제대로 했는지 확인" verification of
  three ISS-full SPR handlers (opset 0x4A, cpsvr 0x4B, mvsvr 0x4C) by
  reading the vendor C++ reference, doing a line-by-line diff, and
  replacing the marker with an explicit `verified against vendor X:Y`
  parity comment. If a real divergence surfaces, fix surgically; if the
  fix would require a >5-line rewrite, ESCALATE (do not patch; document
  in STATE.md and bail out — the acceptance gate ABS regression is the
  guardrail).

- **Category C (5 markers, dma.py):** Micro-perf hint cleanup. Replace
  two per-SPU Python for-loops with single-row vector ops
  (`row[nest_id, :] +=/−= 1`); remove a stale negative-decrement TODO
  (already guarded by outer `total > 0` + inner `row[s] > 0`); and
  document why the first-non-zero credit decrement is intentionally
  NOT vectorisable.

Out-of-scope (deferred to part 2): Category A (4 mcast/copy.mem stubs,
needs vendor C++ port).

Purpose: Reduce `#!TODO` debt in two hot files (spr.py is on the SPR
write/read path; dma.py is on every credit handshake). Vendor parity
comments make future audits / cross-AI review faster and prevent
"TODO archaeology" sessions. Zero behavioural change is the bar for
both edits — the strict-mode ABS + GELU regressions are the gate.

Output: Two files edited, eight markers removed, atomic commit per
task, both `.elf` strict regressions green.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@src/main/python/riscv/gtx/unit/context/dma.py
@src/main/python/riscv/gtx/unit/ins/ops/spr.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc

<interfaces>
<!-- Key invariants the executor must preserve (extracted from CLAUDE.md
     + memory/MEMORY.md + the spec). The executor MUST NOT renegotiate
     these; they are constraints, not goals. -->

Bit-exact regression invariant (CLAUDE.md "Core Value"):
  - 회귀 1개라도 깨지면 출하 보류
  - ABS .elf strict byte-exact PASS (baseline 94.82s)
  - GELU .elf strict byte-exact PASS

Test runner (memory `reference_test_runner`):
  - MUST use `uv run pytest …`. System `pytest` / `python -m pytest` is
    broken (libcusparseLt missing in system torch).

Debug print policy (memory `feedback_debug_prints`):
  - Do NOT auto-remove debug prints. None of the edits below touch
    print statements, so this is a non-issue, but flagged for awareness.

dma.py:_credit_st_chk semantics (already documented in source,
   dma.py:454-468):
  - "first non-zero SPU slot" decrement is INTENTIONAL — mirrors
    producer-side `_credit_st` increment which targets
    `npu._credit_st[nest_id, curr_id]` in the T-loop branch (single
    SPU per call, not broadcast). Vector ops like `row[row > 0] -= 1`
    decrement ALL non-zero slots and are SEMANTICALLY WRONG. The TODO
    marker proposing this is stale / mistaken.

dma.py:_credit_ld (S-loop) semantics:
  - Vendor dispatch.cc:950-962 specifies "increment for all SPUs in
    NEST" when in S-loop context. `row[nest_id, :] += 1` is a faithful
    vector port of the per-SPU for-loop — no semantic change.

Vendor C++ reference locations (memory `reference_vendor_cpp`):
  - `vendor/gtx_cpp_reference/gtx/` — RoCC dispatch reference
    (use this first; it's the in-tree mirror)
  - `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/` — production SystemC
    reference (use as second opinion only if the in-tree mirror is
    ambiguous)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Verify + comment-cleanup three SPR handlers (Category B)</name>
  <files>src/main/python/riscv/gtx/unit/ins/ops/spr.py</files>
  <action>
Goal: Resolve three `#!TODO: 제대로 했는지 확인` markers in spr.py
(lines 149, 168, 206) by doing a line-by-line vendor diff and either
(a) replacing the marker with a parity comment, or (b) applying a
surgical fix if a real divergence is found.

Procedure (repeat per handler):

1. Open `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc` and locate
   the `case` corresponding to:
     - opset → search for `OPSET` / funct7 0x4A
     - cpsvr → search for `CPSVR` / funct7 0x4B
     - mvsvr → search for `MVSVR` / funct7 0x4C
   Capture the exact file:line range of the vendor `case` body.
   If the in-tree mirror is missing/ambiguous, fall back to
   `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc`.

2. Read the vendor C++ body and write down the semantics in 1-2
   bullets (do NOT add these to source — they go in your task
   summary). Specifically verify:

   **opset (spr.py:146-162):**
   - rs1 LSB selects slot {0,1}; rs2 is the value.
   - slot 0 → write GSPR[0x003]; slot 1 → write GSPR[0x005].
   - Cross-check the two GSPR addresses against the vendor case
     (these are the "operand3" / "operand_sel" stage registers).

   **cpsvr (spr.py:165-199):**
   - rs1[4:0] = SVR addr/index; rs2[1:0] = byte size encoding.
   - L0 layout: 4 words per SVR, 8B per word → 32B SVR window
     starting at `base = (rs1[4:0]/4) * 32`.
   - bsz encoding: 0→1B*8, 1→2B*4, 2→4B*2, 3→8B*1 — verify the
     vendor uses the same `byte_size[1:0]` encoding and the same
     replicate-to-32B fill pattern.
   - nest/spu selection (is_ploop / is_tloop fallback) must match
     vendor's gspr/lspr addressing.

   **mvsvr (spr.py:202-233):**
   - rs1[4:0]=src SVR index, rs2[4:0]=dst SVR index, 32B per SVR.
   - `src_idx == dst_idx` → no-op (current code returns 0).
   - `l0[dst:dst+32] = l0[src:src+32].clone()` then
     `l0[src:src+32].zero_()` — verify vendor does copy-THEN-clear
     in the same order (no overlapping-region hazards because we
     bailed on equality above).

3. **If pyspike matches vendor:** Replace the `#!TODO: 제대로 했는지 확인.`
   line with a single comment line of the form:

       # Verified against vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:NNN-MMM
       # (parity confirmed YYYY-MM-DD by 260518-hxk).

   Keep the rest of the docstring intact. Do NOT touch the
   `operand1:` / `operand2:` description lines unless they actually
   disagree with the vendor — if they do, fix the description (it's a
   doc-only change, no behavioural impact).

4. **If pyspike diverges from vendor:** Decide based on size:
   - ≤5-line surgical fix → apply it inline, comment the fix with
     `# Fixed to match vendor … (was: <one-line description of old
     behavior>)`.
   - >5-line refactor or unclear semantics → ESCALATE. Do NOT patch.
     Leave the `#!TODO` marker in place, append `# ESCALATED 260518-hxk:
     <one-line reason>`, and record the divergence in your task summary
     so the user can spawn a follow-up debug task.

5. After all three handlers are processed, run a quick grep sanity
   check to confirm no `#!TODO` markers remain in these three
   functions:

       grep -n '#!TODO' src/main/python/riscv/gtx/unit/ins/ops/spr.py

   Expected: empty output (the file should have zero `#!TODO`
   markers after this task). If markers remain, they're either in
   handlers OUT of scope (none expected) or escalated divergences
   (documented in summary).

Non-negotiable constraints:
- Surgical changes only (CLAUDE.md "Surgical Changes"). Do NOT
  reformat the file, reorder imports, or "improve" adjacent docstrings.
- No behavioural change unless a real vendor divergence is found.
- Atomic commit at end of task.

Per project decision (memory `reference_vendor_cpp`): vendor C++ is
the source of truth for parity.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; grep -c '#!TODO' src/main/python/riscv/gtx/unit/ins/ops/spr.py | grep -qx '0'</automated>
  </verify>
  <done>
- spr.py opset, cpsvr, mvsvr docstrings no longer contain `#!TODO: 제대로 했는지 확인.`
- Each handler has a `# Verified against vendor … :NNN-MMM` comment
  citing exact vendor file:line range OR an `# ESCALATED 260518-hxk:
  <reason>` annotation if a divergence was found but not safely fixable.
- `grep '#!TODO' src/main/python/riscv/gtx/unit/ins/ops/spr.py` → 0 hits.
- Atomic commit message: `refactor(gtx-spr): verify opset/cpsvr/mvsvr against vendor (260518-hxk)`.
- No behavioural diff unless a vendor divergence required a surgical fix.
  </done>
</task>

<task type="auto">
  <name>Task 2: Resolve 5 dma.py credit-handler TODOs (Category C)</name>
  <files>src/main/python/riscv/gtx/unit/context/dma.py</files>
  <action>
Goal: Resolve five `#!TODO` markers in dma.py (lines 306, 314, 337,
463, 464). All five are in the credit_ld / credit_st / credit_st_chk
handlers. Four are perf hints; one is a stale safety-guard hint
(already covered by existing code). Net behavioural change: ZERO.

Marker-by-marker procedure:

1. **dma.py:306 — `_credit_ld` docstring operand confirmation:**
   - Current TODO: `#!TODO: operand 맞는지 확인.`
   - Current docstring claims:
       operand1: *target_spu[63:0]
       operand2: *target_nest[63:0]
   - Cross-check against vendor `gtx_npu_custom0.cc` (search for
     `CREDIT_LD` or funct7 = whatever matches GTX_ISS_F7_CREDIT_LD).
     If unclear, also consult vendor `gtx_npu_dispatch.cc:950-962`
     (already referenced earlier in the same docstring).
   - **If operands match:** Replace the `#!TODO: operand 맞는지 확인.`
     line with `# Operand layout verified against
     vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:NNN (parity
     confirmed 260518-hxk).` Keep the `operand1:` / `operand2:` lines
     intact.
   - **If operands diverge:** Fix the docstring (doc-only change,
     no behavioural impact). Comment the fix.

2. **dma.py:313-316 — `_credit_ld` S-loop vector op:**
   - Current code:
       ```
       if warp.is_sloop:
           #!TODO: vector연산으로 바꿀 것.
           for s in range(GTX_SPU_NUM):
               npu._credit_ld[nest_id, s] += 1
       ```
   - Replace with:
       ```
       if warp.is_sloop:
           # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
           npu._credit_ld[nest_id, :] += 1
       ```
   - Semantic identity: `+= 1` on every column of row `nest_id` is
     exactly what the for-loop did. Tensor in-place op is allowed
     because `npu._credit_ld` is a regular torch tensor (per CLAUDE.md
     stack: NumPy/torch backend).

3. **dma.py:337-339 — `_credit_st` S-loop vector op:**
   - Current code:
       ```
       elif warp.is_sloop:
           #!TODO: vector연산으로 바꿀 것. npu._credit_st[nest_id, :] -= 1
           for s in range(GTX_SPU_NUM):
               npu._credit_st[nest_id, s] -= 1
       ```
   - The TODO marker has already spelled out the fix. Replace with:
       ```
       elif warp.is_sloop:
           # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
           npu._credit_st[nest_id, :] -= 1
       ```

4. **dma.py:463 — `_credit_st_chk` negative-decrement guard TODO:**
   - Current code (the relevant block, dma.py:458-468):
       ```
       if nest_id < GTX_NEST_NUM:
           row = npu._credit_st[nest_id]
           total = int(row.sum().item())
           if total > 0:
               # Decrement one credit (first non-zero SPU slot only).
               #!TODO 만약 0보다 작은데 감소시키면 오류 발생. 해야함.
               #!TODO vector연산으로 바꿀 것. if total > 0: row[row > 0] -= 1; break
               for s in range(GTX_SPU_NUM):
                   if int(row[s]) > 0:
                       row[s] = int(row[s]) - 1
                       break
       ```
   - The outer `if total > 0` AND inner `if int(row[s]) > 0` together
     already prevent any negative decrement. The TODO is stale.
   - Remove the `#!TODO 만약 0보다 작은데 감소시키면 …` line.
   - Insert a single rationale comment in its place:
       ```
       # Negative-decrement is impossible here: outer `total > 0`
       # gate (line above) + inner `row[s] > 0` guard below jointly
       # ensure we only touch positive slots. (260518-hxk verified.)
       ```

5. **dma.py:464 — `_credit_st_chk` "vectorise" TODO:**
   - The TODO proposes `row[row > 0] -= 1; break` which is
     SEMANTICALLY WRONG: it decrements ALL non-zero SPU slots in one
     shot. The current code decrements ONLY the first non-zero slot
     (matches producer-side `credit.st` T-loop branch which increments
     `[nest_id, curr_id]` — single SPU per call). See `<interfaces>`
     block above + existing docstring at dma.py:438-440.
   - Remove the `#!TODO vector연산으로 바꿀 것. …` line.
   - Insert a single rationale comment in its place:
       ```
       # NOT vectorisable: must decrement ONLY the first non-zero SPU
       # slot to mirror the producer-side single-SPU increment at
       # `_credit_st` T-loop branch (dma.py:334). `row[row > 0] -= 1`
       # would decrement every non-zero slot — semantic mismatch.
       # for-loop with break is intentional. (260518-hxk verified.)
       ```

6. After all five markers are processed, run a grep sanity check on
   the three credit handlers:

       grep -n '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py

   Expected: empty output (none of the five remaining; this file may
   still have other TODO markers in unrelated functions — verify
   none of those are at lines 306, 314, 337, 463, 464; if other
   markers exist, leave them, they are part-2 scope).

7. Run the strict-mode acceptance gate (THIS is the
   non-behavioural-change proof):

       cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; \
       uv run pytest \
         'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' \
         --no-cov -v --timeout=900

   Then:

       cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; \
       uv run pytest \
         'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' \
         --no-cov -v --timeout=180

   - Both MUST PASS, byte-exact.
   - ABS walltime SHOULD be ≤ 100s (baseline 94.82s). >150s → regression
     → revert the dma.py edits and ESCALATE.

   Note on the ABS pre-existing regression (memory
   `project_abs_pre_existing_regression`): there's a known
   pre-existing ABS line-1 fp16[0] divergence (ref=134.875 vs
   dump=0.0). That regression EXISTS BEFORE this task starts. The
   acceptance criterion is "no NEW failures introduced by this
   task," not "make the pre-existing failure go away." If ABS was
   strict-PASS at the start of this task, it MUST remain strict-PASS
   after. If ABS was already failing on a specific line, the
   failure pattern MUST be identical (same line, same delta) — no
   widening.

Non-negotiable constraints:
- Surgical changes only — do NOT touch unrelated code in dma.py
  (the file has heavy docstrings; resist the urge to reword them).
- No `print()` insertions (memory `feedback_debug_prints`).
- Atomic commit at end of task.

Per project decision: ABS strict-byte-exact is the regression
guardrail (CLAUDE.md Core Value).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' --no-cov -v --timeout=900</automated>
  </verify>
  <done>
- dma.py lines 306, 314, 337, 463, 464 no longer carry `#!TODO`
  markers; each is replaced by a parity / rationale comment as
  specified above.
- `_credit_ld` S-loop branch uses `npu._credit_ld[nest_id, :] += 1`
  (single-row tensor op, no for-loop).
- `_credit_st` S-loop branch uses `npu._credit_st[nest_id, :] -= 1`
  (single-row tensor op, no for-loop).
- `_credit_st_chk` for-loop preserved (correct first-non-zero
  semantics), surrounded by two new rationale comments.
- ABS strict-mode regression PASS (or unchanged-failure-pattern if
  the pre-existing line-1 divergence was already there at task
  start).
- GELU strict-mode regression PASS.
- ABS walltime ≤ 100s (or ≤ baseline+5%).
- Atomic commit message: `perf(gtx-dma): vectorise credit S-loop ops + clean stale TODOs (260518-hxk)`.
  </done>
</task>

</tasks>

<verification>
Post-task sanity checks:

1. Marker count regression:
   ```
   grep -c '#!TODO' src/main/python/riscv/gtx/unit/ins/ops/spr.py
   ```
   Expected: 0 (down from 3).

   ```
   grep -n '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py
   ```
   Expected: 0 hits at lines 306, 314, 337, 463, 464 (other markers
   in unrelated functions may remain — those are part-2 scope).

2. Vendor parity citations exist:
   ```
   grep -c 'verified against' src/main/python/riscv/gtx/unit/ins/ops/spr.py
   ```
   Expected: ≥ 3 (one per opset/cpsvr/mvsvr handler, unless any was
   escalated).

3. Strict-mode regressions green (the real gate):
   ```
   cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; \
   uv run pytest \
     'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' \
     'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' \
     --no-cov -v --timeout=900
   ```
   ABS + GELU both PASS, byte-exact. ABS walltime ≤ 100s
   (baseline 94.82s).

4. Git log shows two atomic commits (one per task) with
   `260518-hxk` in the commit subject.
</verification>

<success_criteria>
This quick task is complete when:

- spr.py: three `#!TODO: 제대로 했는지 확인` markers REMOVED. Each is
  replaced by a `# Verified against vendor …` comment with explicit
  file:line range (or `# ESCALATED 260518-hxk: <reason>` if a
  divergence was found that couldn't be safely fixed in ≤5 lines).
- dma.py: five `#!TODO` markers at lines 306, 314, 337, 463, 464
  REMOVED. Two for-loops vectorised. Two rationale comments added to
  `_credit_st_chk`. One operand-doc parity comment added to
  `_credit_ld`.
- ABS .elf strict byte-exact regression PASS (or
  unchanged-failure-pattern relative to task-start baseline).
- GELU .elf strict byte-exact regression PASS.
- ABS walltime ≤ 100s.
- Two atomic commits (one per task) tagged `260518-hxk`.
- Category A (4 mcast/copy.mem stubs) explicitly NOT TOUCHED —
  deferred to part 2.
</success_criteria>

<output>
After completion, create
`.planning/quick/260518-hxk-12-todo-part1-spr-verify-dma-perf-hints-/260518-hxk-SUMMARY.md`
documenting:

- Which markers were removed (8 total).
- Per spr.py handler: the exact vendor file:line range cited in the
  parity comment, and any divergences found (with "fixed" or
  "escalated" status).
- Per dma.py change: confirmation of byte-exact behaviour (ABS + GELU
  walltimes from the acceptance gate run).
- Any markers ESCALATED rather than resolved (with one-line reason).
- Pointer to follow-up: "Part 2 (Category A: 4 mcast/copy.mem stubs)
  remains to be planned — needs vendor C++ port effort."
</output>
