"""P4 funct7 routing matrix scaffolds (test_funct7_routing.py).

Covers MM-03 (routing matrix) + MM-05 (#5 Mode 4 dispatch).
Wave 1 plans (gemm_core / mm_engine / ops/mm) GREEN-fill these.
"""
import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_funct7_zero_collision_routing():
    """MM-03 / ROADMAP success #3: funct7=0x00 + insn.rs1!=0 -> MM,
    rs1==0 -> wrspr_gem5 (verbatim port; no MM mutation).

    Plan 04 wired wrspr_gem5/rdspr_gem5 rs1!=0 branch to re-dispatch into
    MM/MMC funct3-keyed handlers. The rs1==0 case still flows through
    wrspr_gem5's verbatim P2 port (writes addr=XPR[0]=0 to GSPR_GTX_RUN).
    Pitfall F NOP per-handler guard ensures no mxe_accum mutation if the
    funct3 handler is invoked directly with rs1==0.
    """
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True

    # Case A: funct7=0x00 + rs1==0. The dispatcher's None-key wins (wrspr_gem5),
    # which writes XPR[0]=0 -> GSPR[0x000]. mxe_accum is NOT mutated.
    accum_before = npu._mxe_accum.copy()
    insn = MockInsn(funct=0x00, rs1=0, xs1=0, xs2=1, xd=0)  # synthesized funct3=1
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    np.testing.assert_array_equal(npu._mxe_accum, accum_before,
        "funct7=0x00 + rs1==0 must NOT mutate mxe_accum (Pitfall F NOP-safety)")

    # Case B: funct7=0x00 + rs1!=0 -> wrspr_gem5 re-dispatches to mm_o (funct3=1).
    proc.get_state().XPR.write(1, (1 << 48) | (4 << 16) | 1)
    npu.lspr[0][0][0x900] = 0  # ADDRA
    npu.mem.l1_f16(0, 0)[0:4] = np.array([1.0] * 4, dtype=np.float16)
    npu.gspr[0x003] = 0  # GSPR_GTX_OPERAND3
    npu._mxe_accum.fill(0.0)
    insn = MockInsn(funct=0x00, rs1=1, xs1=0, xs2=1, xd=0)  # funct3=1 (mm_o)
    npu.custom0(proc, insn, 0, 0)
    # mm_o wrote sum(A) = 4.0 to mxe_accum[0,0]
    assert npu._mxe_accum[0, 0] == np.float32(4.0), \
        f"rs1!=0 should route to mm_o, mxe_accum[0,0]={npu._mxe_accum[0, 0]}"


def test_funct7_one_always_mmc():
    """MM-03: funct7=0x01 always routes to MMC regardless of rs1 (no collision)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.lspr[0][0][0x900] = 0  # ADDRA
    npu.gspr[0x003] = 0
    npu.mem.l1_f16(0, 0)[0:2] = np.array([3.0, 4.0], dtype=np.float16)

    # Pre-load mxe_accum to verify MMC reads prior
    npu._mxe_accum[0, 0] = np.float32(100.0)

    proc.get_state().XPR.write(1, (1 << 48) | (2 << 16) | 1)  # row=1, col=2
    insn = MockInsn(funct=0x01, rs1=1, xs1=0, xs2=1, xd=0)  # funct3=1 mmc_o
    npu.custom0(proc, insn, 0, 0)

    # mmc_o: prior=100 + sum([3,4])=7 -> 107
    assert npu._mxe_accum[0, 0] == np.float32(107.0), \
        f"mmc_o should read prior+add, got {npu._mxe_accum[0, 0]}"


def test_mode4_routes_to_tmu_curr():
    """MM-05 #5: Mode 4 (P+T) dispatch_4mode entry point -- DOCUMENTED NOP at P4 for funct7=GTX_OP_MM.

    Per RESEARCH finding #4: dispatch_iss_opcode and firmware_mm_op are SEPARATE
    paths. P4 deliberately does NOT extend the gem5-simplified DISPATCH_MM body
    in dispatch_iss_opcode; that promotion is P5 territory. This test pins the
    current NOP contract so the dispatch_4mode entry point keeps its shape.
    Companion test test_mode4_firmware_mm_op_routes_to_tmu_curr verifies the
    actual MM dispatch path used by Phase 4."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.dispatch_4mode import dispatch_4mode
    from riscv.gtx.encoding import GTX_OP_MM
    from tests.gtx._mocks import MockProcessor

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.is_sloop = False
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 5

    # dispatch_4mode for GTX_OP_MM in Mode 4 -- NOP at P4 (P5/P6 promotion).
    rc = dispatch_4mode(npu, opcode=GTX_OP_MM, op1=0, op2=0, op3=0, sub_op=0)
    assert rc == 0
    # All cells unchanged (NOP)
    assert (npu._mxe_accum == 0.0).all(), \
        "P4 dispatch_4mode is a documented NOP for funct7=GTX_OP_MM; will be promoted in P5"


def test_mode4_firmware_mm_op_routes_to_tmu_curr():
    """MM-05 #5 (companion): Mode 4 firmware_mm_op path mutates ONLY mxe_accum[tmu_id, curr_id].

    This is the path P4 actually exercises (the @handler shim in ops/mm.py
    delegates to mm_engine.firmware_mm which writes to mxe_accum[warp.tmu_id,
    warp.curr_id]). Use mm_o (which writes to mxe_accum) as the witness.
    Verifies (1) target cell DID mutate, (2) other 63 cells unchanged."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    # Mode 4: P+T loops active, no S-loop, route to (tmu_id=1, curr_id=5)
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.is_sloop = False
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 5

    # Stage A=[1,2,3,4] at L1[(1,5)][0:4] -> sum=10
    npu.lspr[1][5][LSPR_SPM_ADDRA] = 0
    npu.gspr[GSPR_GTX_OPERAND3] = 0
    npu.mem.l1_f16(1, 5)[0:4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)

    # Snapshot full mxe_accum BEFORE
    accum_before = npu._mxe_accum.copy()

    # Drive mm_o through firmware_mm_op path (custom0 funct7=0x00 funct3=1)
    proc.get_state().XPR.write(1, (1 << 48) | (4 << 16) | 1)
    insn = MockInsn(funct=0x00, rs1=1, xs1=0, xs2=1, xd=0)  # funct3=1 (mm_o)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0

    # Target cell mutated to FP32(sum) = 10.0
    assert npu._mxe_accum[1, 5] == np.float32(10.0), \
        f"firmware_mm_op should route mm_o to mxe_accum[1,5]; got {npu._mxe_accum[1, 5]}"
    # All other 63 cells unchanged
    idx_target = 1 * 16 + 5  # = 21
    flat_before = accum_before.flatten()
    flat_after = npu._mxe_accum.flatten()
    other_before = np.delete(flat_before, idx_target)
    other_after = np.delete(flat_after, idx_target)
    np.testing.assert_array_equal(other_after, other_before,
        "firmware_mm_op Mode 4 must touch ONLY mxe_accum[tmu_id, curr_id]; other 63 cells must remain unchanged")
    # Sanity: the unrelated (0,0) sentinel did not move
    assert npu._mxe_accum[0, 0] == 0.0, \
        "firmware_mm_op Mode 4 must NOT touch mxe_accum[0,0]"
