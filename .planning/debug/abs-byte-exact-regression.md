---
status: investigating
trigger: "abs-byte-exact-regression — ABS n1s16_abs.elf byte-exact regression broken during cleanup arc; bisect from 8660c89 (last PASS) to HEAD"
created: 2026-05-15T09:07:24Z
updated: 2026-05-15T10:00:00Z
---

## Current Focus

hypothesis: CONFIRMED via static analysis.
test: |
  Code inspection of `src/main/python/riscv/gtx/unit/context/dma.py` and git history
  via `git show d6f73f9 -- src/main/python/riscv/gtx/unit/context/dma.py`.
expecting: |
  Five @handler blocks commented out (load/store/copy/tpose/fill) — confirmed at HEAD.
  Commenting introduced by d6f73f9 — confirmed by git diff.
  No subsequent commit restores them — confirmed by git log -S on those handler signatures.
next_action: |
  Propose fix to user for approval, then implement.
  Goal mode is `find_root_cause_first_then_propose_fix`, so return diagnosis
  + fix proposal and wait for user direction before applying.

## Symptoms

expected: |
  ABS n1s16_abs.elf 96-tile sweep byte-exact PASS against test/ABS/n1s16/data/n1s16_abs_ref.nohdr.txt.
  First reference fp16 value: ref=0x5837 (134.875). All 96 tiles' worth of DDR dump match byte-for-byte.
  This was the gold-standard demo for Phase 8 Plan 04 completion (commit `8660c89`).
actual: |
  fp16 mismatch at line 1, first value:
    ref=0x5837 (134.875)
    dump=0x0000 (0.0)
    ulp=22583 (far beyond ULP=1 / atol=0.001 tolerance)
  First byte is wrong — dump=0.0 implies store path is broken (DDR region uninitialized or zero-filled at the dump address).
  97 tiles complete the run (~13min 39s, 8s/tile average), so dispatch loop finishes; only the OUTPUT bytes are wrong.
errors: |
  No AssertionError in stderr (verified via grep on full stderr log).
  No python exceptions or warnings beyond the standard `_version` warning.
  pytest exit code 0 for the subprocess (the byte-compare in pytest is what fails, not the spike subprocess).
reproduction: |
  PYTEST_ELF_REGRESSION=1 PYTEST_ELF_TIMEOUT=1200 uv run pytest tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=1200 -v
  Wall time: ~13.5 minutes for this single case. Use 1200s timeout.
  CRITICAL: `uv run pytest` only — system pytest has broken torch (`libcusparseLt`).
started: |
  Last known PASS: commit `8660c89` (Phase 8 Plan 04 completion, ~2026-05-10).
  Confirmed FAIL at HEAD `5ba1b86` after quick 260515-mie was fully reverted.
  Plan invariant commits (45d72f1, ed92898) are NOT the cause — revert measurement confirms identical failure.

## Eliminated

- hypothesis: plan invariant work (45d72f1 sloop/tloop sentinels, ed92898 start/end asserts) broke ABS
  evidence: quick 260515-mie revert (15a9d19 + 725b2aa) produced identical byte mismatch (line 1 fp16[0]: ref=0x5837 dump=0x0000 ulp=22583). Measured 2026-05-15.
  timestamp: 2026-05-15T09:07:24Z

## Evidence

- timestamp: 2026-05-15T09:07:24Z
  checked: git log between 8660c89 (last known PASS, Phase 8 Plan 04) and HEAD 5ba1b86
  found: |
    Cleanup arc commits (in chronological order, oldest first; reverse of git log):
      8660c89 — Phase 8 Plan 04 (last known PASS)
      [many commits between 8660c89 and 3d9988e — needs to be enumerated]
      3d9988e docs(quick-260514-sqv): land planner PLAN.md artifact
      f245b0d docs: simplify ORDER.md FSM definition (356 → 56 lines)
      aa742f5 fix(gtx): disasm precedence bug + rename CUSTOM0/1_OPCODE → CUSTOM0/1
      6bc2c3f test(gtx): reset test infra for ORDER.md FSM redesign
      10f6b74 docs(state): record dirty tree cleanup + sqv followup completion
      1fda59d test(quick-260514-ti0-01): add CSR registry + RegisterFile tensor chain coverage
      5c5bc67 test(quick-260514-ti0-02): add custom0/1 dispatch chain coverage
      8197364 docs(quick-260514-ti0): complete CSR registry + dispatch-chain test plan
      222e013 docs(state): record quick task 260514-ti0
      b228422 fix(gtx): restore GSPR_GTX_OPERAND0..5 register declarations
      b5700da fix(gtx): realign 5 callsite imports for restored GSPR_GTX_OPERAND0..5
      8bbfb1a docs(260514-vjk): complete GSPR_GTX_OPERAND0..5 restoration plan
      718d141 docs(state): record quick task 260514-vjk
      b5df4a0 fix(gtx): RegisterView 64-bit field setter — wrap shifted mask through signed int64
      c355314 docs(state): record quick task 260514-vwp
      1d6cd9a refactor(gtx): drop GSPR/NSPR/LSPR base-constant redefinition
      e8a5f25 fix(gtx): uncomment GSPR_GTX_OPERAND3 import in dma.py
      a3489d4 docs(state): record quick task 260515-0c4
      b464bb4 fix(gtx): single-source SPR addresses via csr — no aliases, inline `.address` ← TOP SUSPECT
      2949143 docs(state): record quick task 260515-0ro
      765d7fb refactor(gtx): silent-clamp → assert on NEST/SPU id select sites
      2ec3fab docs(state): record Step 4/4 + cleanup arc closure
      36f5cc5 test(gtx): drop stale test_ddr_modes.py
      cb49a1b docs(state): record quick task 260515-je2
      [45d72f1 / ed92898 — both REVERTED]
      725b2aa Revert: silent-overwrite → assert
      15a9d19 Revert: WarpState plan-lifetime sentinels
      db3cdd1 docs(quick): 260515-mie
      5ba1b86 docs(state): record quick task 260515-mie revert (HEAD)
  implication: |
    The breaking commit is between `8660c89` (PASS) and `cb49a1b` (FAIL, just before reverted invariant work).
    Window has many non-source commits (docs). Source-touching candidates in window (descending suspicion):
      - b464bb4 (top suspect per memory)
      - 1d6cd9a (drop SPR base-constant redefinition)
      - e8a5f25 (uncomment GSPR_GTX_OPERAND3 import)
      - 765d7fb (silent-clamp → assert)
      - b5df4a0 (RegisterView 64-bit field setter)
      - aa742f5 (disasm precedence + CUSTOM0/1 rename)
      - b228422 / b5700da (GSPR_GTX_OPERAND0..5 restoration)
      - 1fda59d / 5c5bc67 (test additions — unlikely to affect runtime)
      - 6bc2c3f (test infra reset — unlikely)
      - 36f5cc5 (drop stale test file — unlikely)

## Evidence (continued)

- timestamp: 2026-05-15T09:30:00Z
  checked: b464bb4 full diff inspection
  found: |
    REAL BUG INTRODUCED but NOT on ABS critical path:
    vec.py lines 199 and 213 — search-and-replace error:
      OLD:  npu.gspr.get("GSPR_GTX_OPERAND3", insn.rd)
      NEW:  npu.gspr.get("GSPR['GSPR_GTX_OPERAND3'].address", insn.rd)
    The replacement happened INSIDE THE STRING LITERAL (regex over-match).
    Result: lookup key is the malformed string `"GSPR['GSPR_GTX_OPERAND3'].address"`,
    which doesn't match any register name → AttributeError → default fallback returned.
    
    Pre-bug semantics: lookup OPERAND3 by name → return its value (or default).
    Post-bug semantics: lookup fails → ALWAYS return default (insn.rd).
  implication: |
    This bug exists but only affects L0 SASMD `_dispatch_sasmd` (line 199) and
    L0 II `_dispatch_arith_l0_ii` (line 213). Both only reached when funct3 & 4 is set.
    ABS uses funct7=0x1D (SIGN), goes through `_apply_unary` directly — bug NOT on path.
    
    Still, this is a real defect that should be fixed as a separate commit, regardless of ABS.

- timestamp: 2026-05-15T09:40:00Z
  checked: Full commit window 8660c89..HEAD source commits (`git log --oneline 8660c89..HEAD -- src/main/python/riscv/gtx/`)
  found: |
    38 source-touching commits in window. Key candidates ordered chronologically (older first):
    - 639ddb4 refactor: consolidate ext modules under unit/ + wjoin flush-before-dump
      *** Commit message EXPLICITLY claims: "test/ABS/n1s16/n1s16_abs.elf 96-tile regression
          with GTX_NO_EXIT=1 + timeout 1800s — DDR dump matches byte-for-byte." → ABS PASS here ***
    - 57f056c, 2020c71, 96fd117, f50a292, 5dc7e47 — all claim "ABS regression byte-exact PASS"
    - ee3116b refactor: drop dead WRITEBACK code  (claims "OPSET clear mirrored" — safety reasoned not tested)
    - d6f73f9 Architecture Refactoring  (31 files, +1750 -1511, NO test claim about ABS)
    - 10a8b143 fix: restore _extract_id 2-arg vendor semantics
       *** explicit: "d6f73f9 partial-refactored _extract_id 2-arg -> 1-arg ... causing TypeError on every custom1 dispatch" ***
    - 021cbd3 fix: unmask endp handler NameError (sqv followup)
    - aa742f5 fix: disasm precedence bug + rename
    - b228422 fix: restore GSPR_GTX_OPERAND0..5 register declarations
       *** explicit: "d6f73f9 silently dropped" ***
    - b5700da fix: realign 5 callsite imports for restored GSPR_GTX_OPERAND0..5
    - b5df4a0 fix: RegisterView 64-bit field setter
    - 1d6cd9a refactor: drop GSPR/NSPR/LSPR base-constant redefinition
    - e8a5f25 fix: uncomment GSPR_GTX_OPERAND3 import in dma.py
    - b464bb4 fix: single-source SPR addresses (introduces vec.py L199/L213 bug, but not on ABS path)
    - 765d7fb refactor: silent-clamp → assert
    - [revert pair for plan invariant work, no source effect]
  implication: |
    The break is almost certainly at d6f73f9 or one of the fix commits 10a8b143 / b228422 /
    b5700da / b5df4a0 (or a STILL-UNFIXED regression from d6f73f9 that the smoke tests don't catch).
    
    The fact that the last commit message claiming "ABS PASS" with an actual run is `5dc7e47`
    (and 639ddb4 explicitly), and d6f73f9 follows, makes d6f73f9 the prime candidate.
    
    Smoke gates from b228422, b5700da, b5df4a0, b464bb4, 765d7fb all claim only 26/26 PASS
    on smoke+chain tests — none of them re-ran ABS .elf regression (opt-in, ~14 min).

- timestamp: 2026-05-15T10:00:00Z
  checked: src/main/python/riscv/gtx/unit/context/dma.py @handler decorations at HEAD vs git history
  found: |
    ROOT CAUSE IDENTIFIED — STATIC ANALYSIS (no test run needed).
    
    In `src/main/python/riscv/gtx/unit/context/dma.py` at HEAD, FIVE @handler decorations
    for critical DMA mnemonics are COMMENTED OUT:
    
      Line  55: # @handler(... funct7=GTX_ISS_F7_DMA_LD_ST, funct3=0, mnemonic='load')
      Line  86: # @handler(... funct7=GTX_ISS_F7_DMA_LD_ST, funct3=1, mnemonic='store')
      Line 117: # @handler(... funct7=GTX_ISS_F7_DMA_LD_ST, funct3=2, mnemonic='copy')
      Line 204: # @handler(... funct7=GTX_ISS_F7_DMA_TPOSE,           mnemonic='tpose')
      Line 226: # @handler(... funct7=GTX_ISS_F7_DMA_FILL,            mnemonic='fill')
    
    The function bodies are ALSO commented out (entire `# def _firmware_dma_load(...):` blocks).
    
    Git blame: `git show d6f73f9 -- src/main/python/riscv/gtx/unit/context/dma.py` shows
    the diff that did this. It is part of the "Architecture Refactoring..." commit (d6f73f9,
    2026-05-13). The diff is line-by-line: every line of the 5 handler defs was prefixed with `#`.
    
    Followup commits never restored these. They appear to have been forgotten during the
    refactor — the diff shows the deletions are part of a much larger rewrite (31 files,
    +1750 -1511) and these specific blocks were converted to block comments without
    corresponding replacements being added elsewhere.
  implication: |
    The dispatch chain at HEAD has NO live handler for `funct7=GTX_ISS_F7_DMA_LD_ST`
    (the main firmware_dma instruction used for all L2↔L1, L2↔DDR, L1↔L1 transfers
    in firmware). `state_dispatch` (dispatch_state.py:46) looks up funct7=0x40 in
    `npu._custom0_resolved`, gets None, and `state_execute` (execute.py:26-27) returns
    `WRITEBACK` immediately with no handler call. The mnemonic is `None`, so
    `try_buffer` rejects it (mnemonic not in BUFFERABLE_MNEMONICS, even though the
    string 'load'/'store'/'copy' ARE in that set — but dispatch never resolves to
    that mnemonic).
    
    Failure chain on ABS:
    1. Firmware emits `load L2→L1` (funct7=0x40, funct3=0) → silent NOP. L1 stays zero.
    2. Firmware emits `abs.v` (funct7=0x1D) → buffered into T-loop buffer (handler exists).
    3. Firmware emits `store L1→L2` (funct7=0x40, funct3=1) → silent NOP. L2 stays zero.
    4. At end_t, tloop_buffer.flush is called → walks buffer looking for
       'load'/'vec_unary'/'store' frames. No 'load' frames present → no fusion → falls
       through to `_replay` which only replays the buffered vec_unary entries.
    5. Replayed vec_unary handler reads L1[addr_a] = zeros, computes abs(0)=0, writes
       L1[addr_r] = zeros via `_apply_unary` + `_l1_view_addr().copy_(result)`.
    6. S-loop emits `firmware_dma_sloop_store` (funct7=0x40 funct3=1, is_sloop=True)
       → silent NOP. `npu.deferred_ddr_stores` NEVER GETS APPENDED.
    7. WJOIN flushes empty queue → no L2→DDR transfer.
    8. atexit dump reads DDR = all zeros from reset → emit zeros to dump file.
    9. pytest byte-compare: dump=0x0000, ref=0x5837 → FAIL.
    
    This explains EVERY observed symptom:
    - dump=0x0000 at line 1 (DDR never written)
    - 97 tiles complete (WJOIN/progress markers still fire — they have live handlers)
    - No AssertionError in stderr (assertions in dma_engine functions never fire
      because the handlers that call those functions are all commented out)
    - ~13min 39s wall time (each tile still emits its full instruction stream;
      vec_unary still gets buffered/replayed, just on empty data)
    
    The hypothesis matches the symptom set EXACTLY. No further investigation needed.

## Resolution

root_cause: |
  Commit `d6f73f9 Architecture Refactoring...` (2026-05-13) commented out 5 critical
  @handler decorations + function bodies in `src/main/python/riscv/gtx/unit/context/dma.py`:
    - `_firmware_dma_load`   (funct7=0x40 GTX_ISS_F7_DMA_LD_ST, funct3=0, mnemonic='load')
    - `_firmware_dma_store`  (funct7=0x40 GTX_ISS_F7_DMA_LD_ST, funct3=1, mnemonic='store')
    - `_firmware_dma_copy`   (funct7=0x40 GTX_ISS_F7_DMA_LD_ST, funct3=2, mnemonic='copy')
    - `_tpose`               (funct7=0x38 GTX_ISS_F7_DMA_TPOSE,           mnemonic='tpose')
    - `_fill`                (funct7=0x39 GTX_ISS_F7_DMA_FILL,            mnemonic='fill')
  
  These are the main firmware_dma entry points for L2↔L1 (T-loop), L2↔DDR (S-loop deferred),
  and L1↔L1 (T-loop copy). With them gone, dispatch falls through to silent NOP (handler=None).
  ABS firmware's load/store stream becomes a no-op, so:
    - L2 never receives compute results from L1
    - `firmware_dma_sloop_store` is never invoked → `DeferredDdrStore` is never queued
    - WJOIN's `flush_deferred_ddr_stores` drains an empty queue
    - DDR stays at reset zero → atexit dump matches reset state → byte-compare fails on
      the very first fp16 (0x0000 vs ref 0x5837).
  
  Smoke tests pass because they don't exercise firmware_dma (they test isolated unit pieces
  via direct API calls). Only the ELF regression catches it, and that's opt-in (~14min).

fix: |
  Uncomment the 5 handler blocks in `src/main/python/riscv/gtx/unit/context/dma.py`.
  Specifically:
    - Remove leading `# ` from lines 55-83  (load handler)
    - Remove leading `# ` from lines 86-114 (store handler)
    - Remove leading `# ` from lines 117-138 (copy handler)
    - Remove leading `# ` from lines 204-223 (tpose handler)
    - Remove leading `# ` from lines 226-243 (fill handler)
  Also re-enable the imports at top of file:
    - Line 17 `# GSPR_GTX_OPERAND3` → `GSPR_GTX_OPERAND3` (NO — see note below)
    - Line 18 `# LSPR_SPM_ADDRA, LSPR_SPM_ADDRR` → restore if those constants exist
    - Line 19 `# GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL` → uncomment
  
  IMPORTANT NUANCE: The commented function bodies still reference the OLD API style
  (`GSPR_GTX_OPERAND3` as bare int, `LSPR_SPM_ADDRA` as bare int). Those constants no
  longer exist at HEAD (removed by b464bb4 "single-source SPR addresses"). Uncommenting
  verbatim will produce NameErrors. The handlers must be updated to use the post-b464bb4
  patterns:
    - `npu.gspr.get(GSPR_GTX_OPERAND3, 0)` → `npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)`
    - `npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)` → `npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)`
    - Similarly for LSPR_SPM_ADDRR, LSPR_SPM_ADDRB.
  And import `GSPR, LSPR` from `..csr` at top of file (vec.py and act.py already do this).
  
  Also fix the unrelated vec.py malformed-string bug introduced by b464bb4
  (lines 199 and 213): `"GSPR['GSPR_GTX_OPERAND3'].address"` should be
  `GSPR['GSPR_GTX_OPERAND3'].address` (drop the surrounding quotes) — but this is
  a separate fix; it is not blocking ABS.

verification: ""
files_changed:
  - src/main/python/riscv/gtx/unit/context/dma.py
