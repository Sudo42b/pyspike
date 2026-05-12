"""SPR ops -- WRSPR/RDSPR handlers (SPR-02).

Ports vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:56-113 verbatim.

Four handlers:
  funct7 = 0x49 (ISS-full WRSPR)        -- full encoding with ALL register reads via XPR
  funct7 = 0x48 (ISS-full RDSPR)        -- same, plus force-write to rd
  funct7 = 0x00 (gem5-simplified WRSPR) -- collision-aware: insn.rs1!=0 -> P4 MM stub
  funct7 = 0x01 (gem5-simplified RDSPR) -- collision-aware: insn.rs1!=0 -> P4 MMC stub

All four read register values directly via proc.state.XPR[insn.rs1] to bypass
Spike's xs1=0 -> -1 marshalling (CORE-04 / D-05).
"""
from ..._registry import handler
from ...spr_router import wr_spr, rd_spr


@handler(kind='custom0', funct7=0x49, mnemonic='wrspr')
def wrspr_iss(npu, proc, insn, xs1, xs2):
    """ISS-full WRSPR: addr = rs1, val = rs2 (port custom0.cc:107-113)."""
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = state.XPR[insn.rs2]
    wr_spr(npu, addr & 0xFFFF, val)
    return 0


# P8 RELU fix (2026-05-11): OPSET handler ported from vendor gtx_npu_custom0.cc:116-131.
# Firmware __clamp_min / __sasmd_imm / DMA load_cr etc. call __opset(slot, value)
# before the dispatch instruction. slot=0 stages rs3 into GSPR_GTX_OPERAND3 (0x003),
# slot=1 stages rs4 into GSPR 0x005. Vendor CLAMP_MIN does not actually consume
# OPERAND4 (only OPERAND2 = rs2 scalar), but the stale OPERAND3 leak from prior
# instructions caused multi-tile RELU sweeps to dispatch with wrong staging,
# resulting in 76% byte mismatch vs vendor _ref.txt. Registering OPSET also
# unblocks firmware DMA / DOT / MM variants that explicitly pre-stage operands.
@handler(kind='custom0', funct7=0x4A, mnemonic='opset')
def opset_iss(npu, proc, insn, xs1, xs2):
    """ISS-full OPSET: stage rs3/rs4 for the next instruction.

    Port of vendor gtx_npu_custom0.cc:116-131.
    rs1 = slot index (bit 0): 0 -> GSPR_GTX_OPERAND3 (0x003), 1 -> GSPR 0x005.
    rs2 = staging value (full reg width preserved).
    OPSET does NOT clear the other slot; both slots are consumed and cleared
    by the next dispatch (vendor return-without-clear contract).
    """
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    slot = rs1_val & 1
    if slot == 0:
        npu.gspr[0x003] = rs2_val  # GSPR_GTX_OPERAND3
    else:
        npu.gspr[0x005] = rs2_val  # GSPR_GTX_OPERAND4 (vendor uses literal 0x005)
    return 0


@handler(kind='custom0', funct7=0x48, mnemonic='rdspr')
def rdspr_iss(npu, proc, insn, xs1, xs2):
    """ISS-full RDSPR: addr = rs1, return rd_spr(addr); force-write to rd (port custom0.cc:96-105)."""
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = rd_spr(npu, addr & 0xFFFF)
    if insn.rd != 0:
        state.XPR.write(insn.rd, val)
    return val


@handler(kind='custom0', funct7=0x00, mnemonic='wrspr_gem5')
def wrspr_gem5(npu, proc, insn, xs1, xs2):
    """gem5 WRSPR with funct7=0x00 collision (D-02): insn.rs1!=0 -> P4 MM dispatch.

    Port of custom0.cc:56-72.

    Plan 04 wired the rs1!=0 branch to MM dispatch (was a P2 stub returning 0).
    Because npu.custom0 (npu.py:125-144) tries None-key first, this handler must
    re-dispatch to the funct3-keyed MM variant when rs1!=0; otherwise the 5 MM
    handlers registered at funct7=0x00 would be unreachable.

    Note: when insn.rs1==0, XPR[0]=0 always, so addr=0. This is the verbatim
    C++ behavior (writes GSPR_GTX_RUN). Open question 1 from research is
    documented but the port is verbatim.
    """
    if insn.rs1 != 0:
        # Plan 04: re-dispatch to per-funct3 MM handler (5 variants under funct7=0x00).
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        sub_table = npu._custom0.get(0x00, {})
        mm_handler = sub_table.get(funct3)
        if mm_handler is not None:
            return mm_handler(proc, insn, xs1, xs2)
        return 0  # unknown funct3 -> NOP
    state = proc.state
    val_rs1 = state.XPR[insn.rs1]   # always 0 (x0 hardwired)
    val_rs2 = state.XPR[insn.rs2]
    wr_spr(npu, val_rs1 & 0xFFFF, val_rs2)
    return 0


@handler(kind='custom0', funct7=0x01, mnemonic='rdspr_gem5')
def rdspr_gem5(npu, proc, insn, xs1, xs2):
    """gem5 RDSPR with funct7=0x01 collision: insn.rs1!=0 -> P4 MMC dispatch.

    Port of custom0.cc:74-80.

    Plan 04 wired the rs1!=0 branch to MMC dispatch (was a P2 stub returning 0).
    Same re-dispatch pattern as wrspr_gem5 (None-key precedence in npu.custom0).
    """
    if insn.rs1 != 0:
        # Plan 04: re-dispatch to per-funct3 MMC handler (5 variants under funct7=0x01).
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        sub_table = npu._custom0.get(0x01, {})
        mmc_handler = sub_table.get(funct3)
        if mmc_handler is not None:
            return mmc_handler(proc, insn, xs1, xs2)
        return 0  # unknown funct3 -> NOP
    state = proc.state
    val_rs1 = state.XPR[insn.rs1]   # always 0 (x0)
    return rd_spr(npu, val_rs1 & 0xFFFF)
