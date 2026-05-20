"""
credit.ld	4'b1010	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	tmu	5	N/A	*target_spu[63:0]	*target_nest[63:0]	N/A	N/A	N/A	N/A	load credit inc/dec
credit.st	4'b1010	3'b001	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	tmu	5	N/A	*target_spu[63:0]	N/A	N/A	N/A	N/A	N/A	store credit inc/dec
credit.chk	4'b1010	3'b011	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	tmu	4	N/A	*target_spu[63:0]	N/A	N/A	N/A	N/A	N/A	check load/store credit

"""
from ...inst_handler import inst_register
from ....config_params import NEST_NUM, SPU_NUM
# SPR
# ----- ISS funct7 (custom0) -- DMA section-----
GTX_ISS_F7_CREDIT_LD: int = 0x50      # credit.ld -- per-NEST/SPU counter inc/dec 0b1010000
GTX_ISS_F7_CREDIT_ST: int = 0x51      # credit.st -- per-NEST/SPU counter inc/dec 0b1010001
GTX_ISS_F7_CREDIT_LD_CHK: int = 0x52  # credit.ld.chk -- flush trigger when is_sloop 0b1010010
GTX_ISS_F7_CREDIT_ST_CHK: int = 0x53  # credit.st.chk -- flush trigger when is_sloop 0b1010011
GTX_ISS_F7_DMA_LD_ST: int = 0x40      # firmware DMA load/store/copy
GTX_ISS_F7_DMA_3D: int = 0x41         # SVR + 3D variants (load.svr/store.svr/load.3d/store.3d)


# ============================================================================
# credit_ld_chk / credit_st_chk -- mid-execution flush trigger for plan-style
# (WSPLIT/WJOIN) firmware. Vendor parity: gtx_npu_dispatch.cc:898-905 collapses
# both funct7=0x52 and 0x53 into the same `if (is_sloop) flush_deferred_ddr_stores()`
# behavior. P8 MTDMA-01: multi-tile vendor `.elf` (e.g. n1s16_abs.c) emits
# `credit.ld.chk` (0x52) -- NOT `credit.st.chk` (0x53) -- inside the shared
# block before pushing the next tile's deferred __store_cr; without 0x52 also
# flushing, the deferred queue accumulates 96+ entries that all read stale L2
# at exit-time atexit flush, scrambling tiles 0..N-1 with tile N's data.
# ============================================================================
# P8 NEG fix (2026-05-11): credit_ld (0x50) and credit_st (0x51) handlers were
# missing — vendor non-multi-tile firmware (n1s16_neg.c, n1s16_exp.c, etc.) emits
# both as part of the SPU thread credit-counter dance. In the functional model
# these are sequential-execution NOPs (per vendor gtx_npu_custom0.cc:857-882:
# they only inc/dec per-NEST/per-SPU counters that the *_chk variants never
# actually wait on — gtx_npu_custom0.cc:889-905 comment "always true in
# functional model — DMA is instantaneous"). However, without these handlers
# registered, pyspike's default dispatch fell through unexpectedly, causing
# single-tile vendor sweep ops (NEG/EXP/EXPM1/CUMSUM) to hang waiting on the
# subsequent flush trigger that never landed in the expected dispatch slot.
@inst_register.custom0(kind='custom0', funct7=GTX_ISS_F7_CREDIT_LD, mnemonic='credit.ld')
def _credit_ld(npu, proc, insn, xs1, xs2):
    """Port of vendor dispatch.cc:950-962 (credit.ld) — full counter logic.

    Loop-context-dependent per-NEST/per-SPU credit_ld counter update:
      S-loop (is_sloop): DMA load done → increment credit_ld[s] for all SPUs in NEST
      T-loop (is_tloop): SPU consumes load credit → decrement credit_ld[curr_id]

    pyspike's sequential model makes the *_chk variants pass unconditionally,
    so the counter state is currently unobserved by control flow. Tracked
    anyway for vendor 1:1 parity and to surface future check-path coupling.

    # Operand layout verified against
    # vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:646-661 +
    # gtx_npu_dispatch.cc:874-882 (parity confirmed 260518-hxk):
    # vendor consumes ONLY warp state (is_ploop / is_sloop / is_tloop +
    # tmu_id + curr_id) — no rs1/rs2 GPR reads, no GSPR operand staging.
    # The target_spu / target_nest bitmask docstring fields below describe
    # the ENCODING SLOTS reserved by the ISA but NOT used by the functional
    # model (mirrors vendor: both the dispatch.cc and custom0.cc paths
    # ignore them — they only matter for a future cycle-accurate path).
    operand1: *target_spu[63:0]
    operand2: *target_nest[63:0]
    """
    warp = npu.warp
    nest_id = warp.tmu_id if warp.is_ploop else 0
    if nest_id < NEST_NUM:
        if warp.is_sloop:
            # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
            # Equivalent to: for s in range(SPU_NUM): _credit_ld[nest_id, s] += 1
            npu._credit_ld[nest_id, :] += 1
        elif warp.is_tloop and warp.curr_id < SPU_NUM:
            npu._credit_ld[nest_id, warp.curr_id] -= 1
    return 0


@inst_register.custom0(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST, mnemonic='credit.st')
def _credit_st(npu, proc, insn, xs1, xs2):
    """Port of vendor dispatch.cc:963-974 (credit.st) — full counter logic.

    Loop-context-dependent per-NEST/per-SPU credit_st counter update:
      T-loop (is_tloop): SPU done computing → increment credit_st[curr_id]
      S-loop (is_sloop): DMA store consumes credit → decrement credit_st[s] for all SPUs
    operand1: *target_spu[63:0]
    """
    warp = npu.warp
    nest_id = warp.tmu_id if warp.is_ploop else 0
    if nest_id < NEST_NUM:
        if warp.is_tloop and warp.curr_id < SPU_NUM:
            npu._credit_st[nest_id, warp.curr_id] += 1
        elif warp.is_sloop:
            # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
            # Verified against vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:668-672
            # (S-loop decrements all SPUs in NEST). Equivalent to:
            # for s in range(SPU_NUM): _credit_st[nest_id, s] -= 1
            npu._credit_st[nest_id, :] -= 1
    return 0


@inst_register.custom0(kind='custom0', funct7=GTX_ISS_F7_CREDIT_LD_CHK,
         mnemonic='credit.ld.chk')
def _credit_ld_chk(npu, proc, insn, xs1, xs2):
    """Credit-gated TMU dequeue (260517-s9k) — runs in C3 (is_tloop) context.
    #!operand1: *target_spu[63:0]
    Vendor parity
    -------------
    Mirrors ``gtx_npu_dispatch.cc:41-61`` (use_spu_queue / scredit_flag
    push/pop infrastructure). Vendor C++ pushes opcodes onto per-SPU
    queues when ``scredit_flag[spu]`` is set, and pops them when credit
    becomes available. pyspike's functional model has no actual stall
    (DMA instantaneous), so this handler is effectively
    "consume one credit, dispatch one batch from the T-loop buffer."

    Spec rule 7 (260517-s9k task spec)
    ---------------------------------
    S-loop drains FIRST whenever both buffers have entries at a chk
    point. DDR<->L2 is the sole data path, so an SMU batch that's still
    queued can't be replayed AFTER a TMU compute batch that consumed its
    output — drain the producer (SMU) before the consumer (TMU) here.

    Double-decrement resolution: chose option (a) — CLAMP-at-0
    ---------------------------------------------------------
    The producer-side ``_credit_ld`` already decrements
    ``npu._credit_ld[nest, curr_id]`` in the T-loop branch
    (see :func:`_credit_ld` lines 325-334). If THIS handler ALSO
    decremented unconditionally, the counter would go negative on real
    firmware (e.g. ABS emits credit.ld once per SPU per tile inside the
    TMU thread, then credit.ld.chk consumes it).

    Chose (a) over (b) because:
      - Safer: clamp-at-0 is a no-op when the producer-side already
        decremented, so the existing eager-mode behavior is preserved
        bit-for-bit (verified: ABS .elf 96-tile strict byte-exact PASS).
      - Reversible: if a future cycle-accurate path needs (b) — remove
        the T-loop branch decrement at dma.py:325-334 and make this
        handler the sole consumer — the change is local to two functions.
      - (b) is "cleaner semantically" but riskier; it may surface
        regressions in non-multi-tile firmware that relied on the prior
        producer-side decrement pattern. Defer to a separate plan if
        actually needed.

    Non-regression invariant (CRITICAL)
    -----------------------------------
    This handler MUST NOT call :meth:`flush_deferred_ddr_stores`.
    Deferred-store visibility stays owned EXCLUSIVELY by:
      - ``control.py:_do_endp`` when ``!wsplit_seen``
      - ``control.py:wjoin_with_exit`` (custom1 funct3=0b101)
      - ``control.py:wjoin_custom0_no_exit`` (custom0 funct7=0x03)
    The earlier Plan 04 attempt to flush here broke ADD-style firmware
    whose shared block sandwiches ``__credit_chk`` BETWEEN successive
    ``__store`` calls (see the pre-260517-s9k NOP rationale preserved
    in git history). The buffer dequeue below is a different mechanism:
    it replays SMU-snapshotted DMA ops in firmware-emitted order; it
    does NOT commit the deferred-DDR queue.
    """
    warp = npu.warp
    nest_id = warp.tmu_id if warp.is_ploop else 0
    spu_id = warp.curr_id if warp.is_tloop else 0

    # Spec rule 7: S-loop drains first if both have content.
    if npu._sloop_buf:
        from .sloop_buffer import flush as _sloop_flush
        _sloop_flush(npu)

    # Credit-gated TMU dequeue. Counter may already be zero on the very
    # first tile when no producer has fired yet — clamp to 0 (option a).
    if nest_id < NEST_NUM and spu_id < SPU_NUM:
        cred = int(npu._credit_ld[nest_id, spu_id])
        if cred > 0:
            npu._credit_ld[nest_id, spu_id] = cred - 1

    # Drain T-loop buffer (existing fusion path preserved). The chk
    # boundary is the natural batch-end for the inner (load, vec, store)
    # cadence; ``tloop_buffer.flush`` re-arms ``_tloop_buf`` to ``[]``
    # afterward so subsequent bufferable ops keep accumulating until the
    # next chk or ``end_t``.
    if npu._tloop_buf:
        from .tloop_buffer import flush as _tloop_flush
        _tloop_flush(npu)

    return 0


@inst_register.custom0(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST_CHK,
         mnemonic='credit.st.chk')
def _credit_st_chk(npu, proc, insn, xs1, xs2):
    """Credit-gated SMU dequeue (260517-s9k) — runs in C2 (is_sloop) context.

    Mirror of :func:`_credit_ld_chk` for the SMU side: TMU publishes a
    store credit (via ``credit.st`` in T-loop branch, dma.py:347-349),
    SMU consumes it here and dequeues the next L2->DDR batch from
    ``_sloop_buf``.

    SMU is per-NEST (no curr_id meaning), but ``credit_st`` is tracked
    per-(NEST, SPU) by the producer (T-loop increment). Decrement the
    first non-zero SPU slot — one credit per chk invocation, mirroring
    the producer pattern at dma.py:351-352.

    Double-decrement resolution: chose option (a) — CLAMP-at-0
    ---------------------------------------------------------
    Same rationale as :func:`_credit_ld_chk` — the producer-side
    ``_credit_st`` S-loop branch already decrements
    ``npu._credit_st[nest, :]``. Clamp-at-0 here keeps the existing
    eager-mode behavior bit-for-bit if the producer already drained.

    Non-regression invariant
    ------------------------
    Does NOT call :meth:`flush_deferred_ddr_stores` — see
    :func:`_credit_ld_chk` docstring. The buffer dequeue below replays
    SMU-snapshotted DMA ops; it does NOT commit the deferred-DDR queue.
    """
    warp = npu.warp
    nest_id = warp.tmu_id if warp.is_ploop else 0

    if nest_id < NEST_NUM:
        row = npu._credit_st[nest_id]
        total = int(row.sum().item())
        if total > 0:
            # Decrement one credit (first non-zero SPU slot only).
            # Negative-decrement is impossible here: outer `total > 0`
            # gate (line above) + inner `row[s] > 0` guard below jointly
            # ensure we only touch positive slots. (260518-hxk verified.)
            # NOT vectorisable: must decrement ONLY the first non-zero SPU
            # slot to mirror the producer-side single-SPU increment at
            # `_credit_st` T-loop branch (this file, T-loop +=1 branch).
            # `row[row > 0] -= 1` would decrement every non-zero slot —
            # semantic mismatch with producer pattern. for-loop with break
            # is intentional. Vendor parity: gtx_npu_custom0.cc:662-676
            # functional model has no observable wait (DMA instantaneous);
            # the counter delta is the only side-effect that matters.
            # (260518-hxk verified.)
            for s in range(SPU_NUM):
                if int(row[s]) > 0:
                    row[s] = int(row[s]) - 1
                    break

    # Drain S-loop buffer (sequential replay, no fusion). Spec rule 7
    # ("S-loop drains first") is trivially satisfied here because
    # ``_tloop_buf`` is None in C2 context (TMU is not active inside an
    # SMU section — see warp_state.WarpState mutual exclusion).
    if npu._sloop_buf:
        from .sloop_buffer import flush as _sloop_flush
        _sloop_flush(npu)

    return 0
