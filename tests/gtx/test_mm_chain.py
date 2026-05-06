"""P4 MM chain integration scaffolds (test_mm_chain.py).

Covers MM-04 (ADDRC + mxe_accum + isolation + dtype).
Wave 2 Plan 05 GREEN-fills these via the full Wave 1 stack
(gemm_core + mm_engine + ops/mm).
"""
import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_mm_addrc_chain_continuity():
    """MM-04 / ROADMAP success #2: mm.s -> mmc.s -> mmc chain via ADDRC FP32 bias.

    NOTE per RESEARCH Pitfall B: this is the ADDRC-bias chain, NOT an mxe_accum
    chain. mm.s/mmc.s/mmc variants do NOT touch mxe_accum -- they use
    LSPR_SPM_ADDRC=0x902."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required for GtxNpu")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import (
        LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
    )
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = nest
    npu.warp.curr_id = spu

    # Layout: A at L1[0:0x80], B at L1[0x100:0x180], ADDRC FP32 stage at L1[0x200:0x340],
    # final FP16 result at L1[0x400:0x420].
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.lspr[nest][spu][LSPR_SPM_ADDRB] = 0x100
    npu.lspr[nest][spu][LSPR_SPM_ADDRC] = 0x200
    npu.lspr[nest][spu][LSPR_SPM_ADDRR] = 0x400

    np.random.seed(7)
    # 4x4 matrices for the chain (small for fast 3-loop).
    A1 = np.random.randn(4, 4).astype(np.float16)
    B1 = np.random.randn(4, 4).astype(np.float16)
    A2 = np.random.randn(4, 4).astype(np.float16)
    B2 = np.random.randn(4, 4).astype(np.float16)
    A3 = np.random.randn(4, 4).astype(np.float16)
    B3 = np.random.randn(4, 4).astype(np.float16)

    # rs1 packed for 4x4x4 (row_A=4, col_A=4, col_B=4) -- per mm_engine.decode_firmware_mm_args.
    rs1_packed = (4 << 48) | (4 << 16) | 4
    proc.get_state().XPR.write(1, rs1_packed)

    # Snapshot mxe_accum before chain (will assert unchanged after -- Pitfall B).
    accum_before = npu._mxe_accum.copy()

    # Helper: stage a 4x4 FP16 matrix into L1 at the given byte offset, LE.
    def _stage_fp16_matrix(addr, mat):
        l1 = npu.mem.l1_byte(nest, spu)
        flat = mat.flatten()
        for idx, v in enumerate(flat):
            raw = int(v.view(np.uint16))
            off = addr + idx * 2
            l1[off] = raw & 0xFF
            l1[off + 1] = (raw >> 8) & 0xFF

    # Step 1: mm_s (funct7=0x00, funct3=0). xd=0, xs1=0, xs2=0 -> funct3 = 0.
    _stage_fp16_matrix(0, A1)
    _stage_fp16_matrix(0x100, B1)
    insn_mm_s = MockInsn(funct=0x00, rs1=1, xd=0, xs1=0, xs2=0)
    npu.custom0(proc, insn_mm_s, 0, 0)

    # Step 2: mmc_s (funct7=0x01, funct3=0). xd=0, xs1=0, xs2=0.
    _stage_fp16_matrix(0, A2)
    _stage_fp16_matrix(0x100, B2)
    insn_mmc_s = MockInsn(funct=0x01, rs1=1, xd=0, xs1=0, xs2=0)
    npu.custom0(proc, insn_mmc_s, 0, 0)

    # Step 3: mmc (funct7=0x01, funct3=2). funct3=2 means xd<<2|xs1<<1|xs2 = 010
    # -> xd=0, xs1=1, xs2=0.
    _stage_fp16_matrix(0, A3)
    _stage_fp16_matrix(0x100, B3)
    insn_mmc = MockInsn(funct=0x01, rs1=1, xd=0, xs1=1, xs2=0)
    npu.custom0(proc, insn_mmc, 0, 0)

    # Oracle: explicit 3-loop FP32 accumulate (Warning 4 fix per checker iter-1).
    # BLAS np.matmul has documented 1-4 ULP drift on 41/500 random 16x16x16 trials
    # (RESEARCH `np.matmul` Bit-Exactness Analysis); explicit 3-loop matches
    # gemm_core (Plan 02) exactly so the chain test cannot fail spuriously.
    def _oracle_matmul_3loop(A, B):
        """FP32 explicit 3-loop accumulate, mirrors gemm_core (Plan 02 D-03)."""
        af = A.astype(np.float32)
        bf = B.astype(np.float32)
        M, K = af.shape
        K2, N = bf.shape
        assert K == K2
        out = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                s = np.float32(0.0)
                for k in range(K):
                    s += af[i, k] * bf[k, j]
                out[i, j] = s
        return out

    # Full-FP32 chain, single FP16 cast at end (matches Wave 1 gemm_core dtype discipline).
    expected_f32 = (
        _oracle_matmul_3loop(A1, B1)
        + _oracle_matmul_3loop(A2, B2)
        + _oracle_matmul_3loop(A3, B3)
    )
    expected = expected_f32.astype(np.float16)

    # Read the FP16 result that mmc wrote to ADDRR (LE bytes).
    l1 = npu.mem.l1_byte(nest, spu)
    actual_flat = np.zeros(16, dtype=np.float16)
    for idx in range(16):
        off = 0x400 + idx * 2
        raw = (int(l1[off + 1]) << 8) | int(l1[off])
        actual_flat[idx] = np.frombuffer(np.uint16(raw).tobytes(), dtype=np.float16)[0]
    actual = actual_flat.reshape(4, 4)

    np.testing.assert_array_equal(
        actual.view(np.uint16), expected.view(np.uint16),
        err_msg="ADDRC chain (mm.s -> mmc.s -> mmc) result mismatch",
    )

    # Pitfall B verification: mxe_accum MUST be unchanged (mm.s/mmc.s/mmc don't touch it).
    np.testing.assert_array_equal(
        npu._mxe_accum, accum_before,
        err_msg="mm.s/mmc.s/mmc must NOT mutate mxe_accum (Pitfall B)",
    )


def test_mxe_accum_chain_continuity():
    """MM-04 / Pitfall 3: mm.o -> mmc.o chain on mxe_accum[(nest=1,spu=5)] --
    only MM_O/MMC_O/MM_V/MMC_V touch mxe_accum."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 1, 5
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = nest
    npu.warp.curr_id = spu

    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.gspr[GSPR_GTX_OPERAND3] = 0  # L0 dest at idx 0

    # Stage A1 = [1,2,3,4] -> sum=10.
    A1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    l1 = npu.mem.l1_byte(nest, spu)
    for idx, v in enumerate(A1):
        raw = int(v.view(np.uint16))
        l1[idx * 2] = raw & 0xFF
        l1[idx * 2 + 1] = (raw >> 8) & 0xFF

    # rs1: row_A=1, col_A=4, col_B=1 (col_B unused for mm_o but must be nonzero).
    rs1_packed = (1 << 48) | (4 << 16) | 1
    proc.get_state().XPR.write(1, rs1_packed)

    # Step 1: mm_o (funct7=0x00, funct3=1). xd=0, xs1=0, xs2=1.
    insn_mm_o = MockInsn(funct=0x00, rs1=1, xd=0, xs1=0, xs2=1)
    npu.custom0(proc, insn_mm_o, 0, 0)
    assert npu._mxe_accum[nest, spu] == np.float32(10.0), \
        f"after mm_o: expected 10.0, got {npu._mxe_accum[nest, spu]}"

    # Step 2: mmc_o (funct7=0x01, funct3=1). Stage A2 = [5,6,7,8] -> sum=26; chain -> 36.
    A2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16)
    for idx, v in enumerate(A2):
        raw = int(v.view(np.uint16))
        l1[idx * 2] = raw & 0xFF
        l1[idx * 2 + 1] = (raw >> 8) & 0xFF
    insn_mmc_o = MockInsn(funct=0x01, rs1=1, xd=0, xs1=0, xs2=1)
    npu.custom0(proc, insn_mmc_o, 0, 0)
    assert npu._mxe_accum[nest, spu] == np.float32(36.0), \
        f"after mmc_o: expected 36.0, got {npu._mxe_accum[nest, spu]}"

    # Warning 5 fix per checker iter-1: also assert L0 BE bytes.
    # mm_o/mmc_o write FP16(scalar) to L0 in BIG-endian (gtx_npu_mm.cc:217-218).
    # FP16(36.0) raw = 0x5080 -> BE bytes [0x50, 0x80]. Catches a bug where
    # mxe_accum is correct but L0 dump (BE byte path) is wrong.
    l0 = npu.mem.l0_byte(nest, spu)
    assert l0[0] == 0x50, \
        f"L0[0] (BE high byte of FP16(36.0)=0x5080) should be 0x50, got 0x{l0[0]:02x}"
    assert l0[1] == 0x80, \
        f"L0[1] (BE low byte of FP16(36.0)=0x5080) should be 0x80, got 0x{l0[1]:02x}"


def test_mxe_accum_per_cell_isolation():
    """MM-04: only mxe_accum[1,5] mutates; other 4*16-1=63 cells unchanged
    (snapshot diff)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 1, 5
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = nest
    npu.warp.curr_id = spu
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.gspr[GSPR_GTX_OPERAND3] = 0

    # Stage A = [1,1,1,1] -> sum=4.0.
    A = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)
    l1 = npu.mem.l1_byte(nest, spu)
    for idx, v in enumerate(A):
        raw = int(v.view(np.uint16))
        l1[idx * 2] = raw & 0xFF
        l1[idx * 2 + 1] = (raw >> 8) & 0xFF

    # Snapshot before chain.
    accum_before = npu._mxe_accum.copy()

    rs1_packed = (1 << 48) | (4 << 16) | 1
    proc.get_state().XPR.write(1, rs1_packed)

    # Single mm_o -> mxe_accum[1,5] = 4.0.
    insn_mm_o = MockInsn(funct=0x00, rs1=1, xd=0, xs1=0, xs2=1)
    npu.custom0(proc, insn_mm_o, 0, 0)

    # Per-cell isolation: only [1, 5] differs; 63 other cells unchanged.
    idx_target = nest * 16 + spu  # = 21
    flat_before = accum_before.flatten()
    flat_after = npu._mxe_accum.flatten()
    other_before = np.delete(flat_before, idx_target)
    other_after = np.delete(flat_after, idx_target)
    np.testing.assert_array_equal(
        other_after, other_before,
        err_msg="Only mxe_accum[1, 5] should mutate; other 63 cells must remain unchanged.",
    )
    # And the target cell DID change.
    assert npu._mxe_accum[nest, spu] != accum_before[nest, spu], \
        "mxe_accum[1, 5] must mutate after mm_o"


def test_mxe_accum_dtype_locked():
    """MM-04: npu._mxe_accum.dtype == np.float32 stays float32 across chain
    (Pitfall 3 dtype-slip guard)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)

    # Sanity: dtype is float32 after reset.
    assert npu._mxe_accum.dtype == np.float32, \
        f"post-reset mxe_accum dtype {npu._mxe_accum.dtype} != float32"

    # Run chain on (1, 5).
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 5
    npu.lspr[1][5][LSPR_SPM_ADDRA] = 0
    npu.gspr[GSPR_GTX_OPERAND3] = 0

    A = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)
    l1 = npu.mem.l1_byte(1, 5)
    for idx, v in enumerate(A):
        raw = int(v.view(np.uint16))
        l1[idx * 2] = raw & 0xFF
        l1[idx * 2 + 1] = (raw >> 8) & 0xFF

    proc.get_state().XPR.write(1, (1 << 48) | (4 << 16) | 1)

    for funct7 in (0x00, 0x01):  # mm_o then mmc_o
        insn = MockInsn(funct=funct7, rs1=1, xd=0, xs1=0, xs2=1)
        npu.custom0(proc, insn, 0, 0)

    # dtype must remain float32 (Pitfall 3: never slip to float16 or float64).
    assert npu._mxe_accum.dtype == np.float32, \
        f"post-chain mxe_accum dtype {npu._mxe_accum.dtype} != float32 (Pitfall 3)"
