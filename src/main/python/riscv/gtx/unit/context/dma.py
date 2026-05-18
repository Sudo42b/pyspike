"""DMA op @handler entry points -- spike-bound shim layer (CONTEXT D-01).

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc dispatch entry points.
The pure-function bodies live in dma_engine.py (Plan 01); this file ONLY reads
from proc/insn/npu and delegates.

Phase 3 plan 02 Task 2a: 9 active @handler entry points
  - firmware_dma load/store/copy (funct7=0x40, mask_funct3=True, funct3=0/1/2)
  - load_svr/store_svr           (funct7=0x41, mask_funct3=True, funct3=0/1)
  - load_svr_l1/store_svr_l1     (funct7=0x43/0x45, mask_funct3=False)
  - tpose                        (funct7=0x38, mask_funct3=False)
  - fill                         (funct7=0x39, mask_funct3=False)

Phase 3 plan 02 Task 2b: 5 disasm-only stubs + credit_st_chk stub.
"""
from ..._registry import handler
from . import dma_engine
from ..ins.encoding import (
    GTX_ISS_F7_TPOSE, GTX_ISS_F7_FILL,
    GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
    GTX_ISS_F7_DMA_LD_SVR_L1, GTX_ISS_F7_DMA_ST_SVR_L1,
    GTX_ISS_F7_MCAST_S2L, GTX_ISS_F7_MCAST_G2S,
    GTX_ISS_F7_CREDIT_LD, GTX_ISS_F7_CREDIT_ST,
    GTX_ISS_F7_CREDIT_LD_CHK, GTX_ISS_F7_CREDIT_ST_CHK,
)
from ..csr import GSPR, LSPR
from ...config_params import GTX_NEST_NUM, GTX_SPU_NUM


# ============================================================================
# Helpers
# ============================================================================
def _select_nest(npu) -> int:
    """Select NEST id per gtx_npu_dma.cc:289-291.

    is_ploop -> use warp.tmu_id; else default to 0. `warp.tmu_id` is bounded
    to ``[0, GTX_NEST_NUM)`` by the assert inside ``control._do_startp``,
    so we mirror that invariant here instead of silent-clamping.
    """
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    assert nest < GTX_NEST_NUM, f"NEST id {nest} >= GTX_NEST_NUM={GTX_NEST_NUM}"
    return nest


def _select_spu(npu) -> int:
    """Select SPU id from warp.curr_id. Bounded by ``control._do_startt``."""
    spu = npu.warp.curr_id
    assert spu < GTX_SPU_NUM, f"SPU id {spu} >= GTX_SPU_NUM={GTX_SPU_NUM}"
    return spu


# ============================================================================
# firmware_dma load/store/copy (funct7=0x40, mask_funct3=True)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=0,
         mnemonic='load', mask_funct3=True)
def _firmware_dma_load(npu, proc, insn, xs1, xs2):
    """firmware_dma LOAD (funct7=0x40 funct3=0).

    Pitfall 3 (CORE-04): xs1/xs2 args are unreliable when the encoding flag is 0
    (Spike marshals -1). Read XPR[insn.rs1] / XPR[insn.rs2] directly.
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)   # 0x003 per gtx_params.h:40
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        return dma_engine.firmware_dma_sloop_load(
            npu.mem, nest=nest,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_load_store(
            npu.mem, nest=nest, spu=spu, is_store=False,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=1,
         mnemonic='store', mask_funct3=True)
def _firmware_dma_store(npu, proc, insn, xs1, xs2):
    """firmware_dma STORE (funct7=0x40 funct3=1).

    is_sloop branch passes `npu` (not `npu.mem`) so firmware_dma_sloop_store can
    push DeferredDdrStore onto npu.deferred_ddr_stores (Plan 05 flushes).
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        return dma_engine.firmware_dma_sloop_store(
            npu, nest=nest,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_load_store(
            npu.mem, nest=nest, spu=spu, is_store=True,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=2,
         mnemonic='copy', mask_funct3=True)
def _firmware_dma_copy(npu, proc, insn, xs1, xs2):
    """firmware_dma COPY (funct7=0x40 funct3=2). T-loop L1->L1 only.

    Pitfall 1: COPY decodes addr_hi from rs1>>32 (32-bit dst), NOT (rs1>>27)&0x1F..
    addr_hi is the dst, addr_lo is the src.
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_copy(
            npu.mem, nest=nest, spu=spu,
            src_addr=args['addr_lo'], dst_addr=args['addr_hi'],
            length=args['length'], height=args['height'])
    return 0


# ============================================================================
# load_svr/store_svr (funct7=0x41, mask_funct3=True)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_3D, funct3=0,
         mnemonic='load.svr', mask_funct3=True)
def _load_svr(npu, proc, insn, xs1, xs2):
    """load.svr (funct7=0x41 funct3=0): L1 -> L0 SVR transfer (32 bytes)."""
    state = proc.state
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_load_svr(npu.mem, nest_id=nest, spu_id=spu,
                              l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_3D, funct3=1,
         mnemonic='store.svr', mask_funct3=True)
def _store_svr(npu, proc, insn, xs1, xs2):
    """store.svr (funct7=0x41 funct3=1): L0 -> L1 SVR transfer (32 bytes)."""
    state = proc.state
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_store_svr(npu.mem, nest_id=nest, spu_id=spu,
                               l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


# ============================================================================
# tpose / fill (funct7=0x38 / 0x39, mask_funct3=False)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_TPOSE, mnemonic='tpose')
def _tpose(npu, proc, insn, xs1, xs2):
    """tpose (funct7=0x38): matrix transpose in L1 (FP16, 2 bytes per elem).

    Source matrix base: LSPR['SPM_ADDRA'].address (0x900) -- gtx_params.h:64
    Result matrix base: LSPR['SPM_ADDRR'].address (0x903) -- gtx_params.h:67
    AUTHORITATIVE values; no magic numbers in handler body.
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rows = rs1 & 0xFFFF
    cols = rs2 & 0xFFFF
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0) & 0xFFFFFFFF
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0) & 0xFFFFFFFF
    return dma_engine.exec_transpose(
        npu.mem, nest_id=nest, spu_id=spu, rows=rows, cols=cols,
        addr_a=addr_a, addr_r=addr_r)


@handler(kind='custom0', funct7=GTX_ISS_F7_FILL, mnemonic='fill')
def _fill(npu, proc, insn, xs1, xs2):
    """fill (funct7=0x39): fill L1 region at addr_r with constant FP16 value.

    Result address: LSPR['SPM_ADDRR'].address (0x903) -- gtx_params.h:67. AUTHORITATIVE
    constant; no magic number in handler body (LSPR_SPM_ADDRB is NOT used here).
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    length = rs1 & 0xFFFF
    fill_val = (rs1 >> 16) & 0xFFFF
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0) & 0xFFFFFFFF
    return dma_engine.exec_fill(
        npu.mem, nest_id=nest, spu_id=spu,
        length=length, fill_val=fill_val, addr_r=addr_r)


# ============================================================================
# Disasm-only stubs (v2 deferral -- DMA-V2-01)
#
# Per 03-RESEARCH "P3 Scope vs v2 Deferral":
#   load_3d, store_3d, mcast_s2l, mcast_g2s, mcast_s2s, copy_mem
# are registered for disasm parity with C++ but body is NOP in P3.
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_MCAST_S2L, funct3=0,
         mnemonic='mcast.s2l')
def _mcast_s2l(npu, proc, insn, xs1, xs2):
    """firmware mcast.s2l (funct7=0x42): L2 → L1 broadcast to selected SPUs.

    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273.
    rs1 = (L2_src << 32) | L1_dst  (high=src/low=dst — NOT OPSET layout).
    rs2 = (height<<48) | (length<<32) | read_stride.
    rs3 = target_spu_bitmask (from GSPR_GTX_OPERAND3).
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    nest = _select_nest(npu)
    # Vendor: gtx_npu_custom0.cc:241-249 (decode), :253-269 (body).
    return dma_engine.firmware_mcast_s2l(
        npu.mem, nest=nest,
        l2_addr=(rs1 >> 32) & 0xFFFFFFFF,
        l1_addr=rs1 & 0xFFFFFFFF,
        height=(rs2 >> 48) & 0xFFFF,
        length=(rs2 >> 32) & 0xFFFF,
        rd_stride=rs2 & 0xFFFFFFFF,
        target_spu_mask=rs3 & 0xFFFF,
    )


@handler(kind='custom0', funct7=GTX_ISS_F7_MCAST_G2S, funct3=0,
         mnemonic='mcast.g2s', mask_funct3=True)
def _mcast_g2s(npu, proc, insn, xs1, xs2):
    """firmware mcast.g2s (funct7=0x44, f3=0): DDR → L2 broadcast to selected NESTs.

    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:545-583.
    rs1 = (DDR_src << 27) | L2_dst  (37-bit DDR / 27-bit L2).
    rs2 = (height<<48) | (length<<32) | read_stride.
    rs3 = target_nest_bitmask (from GSPR_GTX_OPERAND3).
    NOTE: NO zero-fill special case (vendor has none — earlier docstring was fiction).
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    # Vendor: gtx_npu_custom0.cc:552-562 (decode), :565-580 (body).
    return dma_engine.firmware_mcast_g2s(
        npu.mem,
        ddr_addr=(rs1 >> 27) & 0x1FFFFFFFFF,
        l2_addr=rs1 & 0x7FFFFFF,
        height=(rs2 >> 48) & 0xFFFF,
        length=(rs2 >> 32) & 0xFFFF,
        rd_stride=rs2 & 0xFFFFFFFF,
        target_nest_mask=rs3 & 0xFFFF,
    )


@handler(kind='custom0', funct7=GTX_ISS_F7_MCAST_G2S,
         funct3=2,
         mnemonic='mcast.s2s', mask_funct3=True)
def _mcast_s2s(npu, proc, insn, xs1, xs2):
    """firmware mcast.s2s (funct7=0x44, f3=2; reachable via OPSET sub_op=0x22):
    L2 → L2 across NESTs.

    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:732-762.
    op1[26:0]=src_addr, op1[53:27]=dst_addr, op1[61:56]=src_tmu.
    op2[31:0]=src_stride, op2[47:32]=length, op2[63:48]=height.
    op3[31:0]=dst_stride, op3[63:32]=target_nest_bitmask (FLAT — no self-broadcast
    guard, no select bit; earlier docstring was fiction per RESEARCH Pitfall 3).
    NOTE: funct3=2 firmware reachability uncertain — see RESEARCH Pitfall 4.
    """
    state = proc.state
    op1 = state.XPR[insn.rs1]
    op2 = state.XPR[insn.rs2]
    op3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    # Vendor: gtx_npu_dispatch.cc:740-748 (decode), :751-760 (body).
    return dma_engine.firmware_mcast_s2s(
        npu.mem,
        src_tmu=(op1 >> 56) & 0x3F,
        src_addr=op1 & 0x7FFFFFF,
        dst_addr=(op1 >> 27) & 0x7FFFFFF,
        src_stride=op2 & 0xFFFFFFFF,
        dst_stride=op3 & 0xFFFFFFFF,
        length=(op2 >> 32) & 0xFFFF,
        height=(op2 >> 48) & 0xFFFF,
        target_nest_mask=(op3 >> 32) & 0xFFFFFFFF,
    )


@handler(kind='custom0', funct7=GTX_ISS_F7_MCAST_G2S, funct3=3,
         mnemonic='copy.mem', mask_funct3=True)
def _copy_mem(npu, proc, insn, xs1, xs2):
    """firmware copy.mem (funct7=0x44, f3=3; OPSET sub_op=0x23):
    DDR↔DDR (and L2↔DDR, L2↔L2).

    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:509-543 (decode)
            + vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:763-846 (body).
    op1[36:0]=src_addr_raw, op3[36:0]=dst_addr_raw (37-bit; ≥ GTX_L2_SIZE_BYTES → DDR).
    op2[31:0]=src_stride, op2[47:32]=length, op2[63:48]=height.
    dst_stride = (op1[63:48] low 16) | (op3[63:48] << 16) — split layout.
    DDR-path MUST call npu.flush_deferred_ddr_stores() first (vendor dispatch.cc:784).
    """
    state = proc.state
    op1 = state.XPR[insn.rs1]
    op2 = state.XPR[insn.rs2]
    op3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    src_stride = op2 & 0xFFFFFFFF
    dst_stride = ((op1 >> 48) & 0xFFFF) | (((op3 >> 48) & 0xFFFF) << 16)
    nest = _select_nest(npu)
    # Vendor: gtx_npu_custom0.cc:518-528 (decode), gtx_npu_dispatch.cc:768-845 (body).
    return dma_engine.firmware_copy_mem(
        npu, nest_id=nest,
        src_addr_raw=op1 & 0x1FFFFFFFFF,
        dst_addr_raw=op3 & 0x1FFFFFFFFF,
        src_stride=src_stride,
        dst_stride=dst_stride,
        length=(op2 >> 32) & 0xFFFF,
        height=(op2 >> 48) & 0xFFFF,
    )


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
@handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_LD, mnemonic='credit.ld')
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
    if nest_id < GTX_NEST_NUM:
        if warp.is_sloop:
            # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
            # Equivalent to: for s in range(GTX_SPU_NUM): _credit_ld[nest_id, s] += 1
            npu._credit_ld[nest_id, :] += 1
        elif warp.is_tloop and warp.curr_id < GTX_SPU_NUM:
            npu._credit_ld[nest_id, warp.curr_id] -= 1
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST, mnemonic='credit.st')
def _credit_st(npu, proc, insn, xs1, xs2):
    """Port of vendor dispatch.cc:963-974 (credit.st) — full counter logic.

    Loop-context-dependent per-NEST/per-SPU credit_st counter update:
      T-loop (is_tloop): SPU done computing → increment credit_st[curr_id]
      S-loop (is_sloop): DMA store consumes credit → decrement credit_st[s] for all SPUs
    operand1: *target_spu[63:0]
    """
    warp = npu.warp
    nest_id = warp.tmu_id if warp.is_ploop else 0
    if nest_id < GTX_NEST_NUM:
        if warp.is_tloop and warp.curr_id < GTX_SPU_NUM:
            npu._credit_st[nest_id, warp.curr_id] += 1
        elif warp.is_sloop:
            # Vector port of per-SPU for-loop (260518-hxk perf cleanup).
            # Verified against vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:668-672
            # (S-loop decrements all SPUs in NEST). Equivalent to:
            # for s in range(GTX_SPU_NUM): _credit_st[nest_id, s] -= 1
            npu._credit_st[nest_id, :] -= 1
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_LD_CHK,
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
        from ...sloop_buffer import flush as _sloop_flush
        _sloop_flush(npu)

    # Credit-gated TMU dequeue. Counter may already be zero on the very
    # first tile when no producer has fired yet — clamp to 0 (option a).
    if nest_id < GTX_NEST_NUM and spu_id < GTX_SPU_NUM:
        cred = int(npu._credit_ld[nest_id, spu_id])
        if cred > 0:
            npu._credit_ld[nest_id, spu_id] = cred - 1

    # Drain T-loop buffer (existing fusion path preserved). The chk
    # boundary is the natural batch-end for the inner (load, vec, store)
    # cadence; ``tloop_buffer.flush`` re-arms ``_tloop_buf`` to ``[]``
    # afterward so subsequent bufferable ops keep accumulating until the
    # next chk or ``end_t``.
    if npu._tloop_buf:
        from ...tloop_buffer import flush as _tloop_flush
        _tloop_flush(npu)

    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST_CHK,
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

    if nest_id < GTX_NEST_NUM:
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
            for s in range(GTX_SPU_NUM):
                if int(row[s]) > 0:
                    row[s] = int(row[s]) - 1
                    break

    # Drain S-loop buffer (sequential replay, no fusion). Spec rule 7
    # ("S-loop drains first") is trivially satisfied here because
    # ``_tloop_buf`` is None in C2 context (TMU is not active inside an
    # SMU section — see warp_state.WarpState mutual exclusion).
    if npu._sloop_buf:
        from ...sloop_buffer import flush as _sloop_flush
        _sloop_flush(npu)

    return 0
