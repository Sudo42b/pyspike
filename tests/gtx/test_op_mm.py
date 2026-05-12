"""P4 MM op unit tests -- Wave 0 scaffolds (test_op_mm.py).

Covers MM-01, MM-02, MM-03 (decode), MM-05 (verify_minimal unit).
Wave 1 plans (gemm_core / mm_engine / ops/mm) GREEN-fill these.
"""
import os
import pathlib

import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_gemm_core_explicit_3loop_matches_oracle():
    """MM-01: gemm_core uses explicit Python 3-loop FP32 accumulate (NOT np.matmul)
    per RESEARCH np.matmul Bit-Exactness section."""
    from riscv.gtx.gemm_core import gemm_core
    np.random.seed(42)
    A = np.random.randn(16, 16).astype(np.float16)
    B = np.random.randn(16, 16).astype(np.float16)
    actual = gemm_core(A, B)

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    expected_f32 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        for j in range(16):
            s = np.float32(0.0)
            for k in range(16):
                s += A_f32[i, k] * B_f32[k, j]
            expected_f32[i, j] = s
    expected = expected_f32.astype(np.float16)

    assert actual.dtype == np.float16
    assert actual.shape == (16, 16)
    # Bit-exact compare via uint16 view (D-15)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_gemm_core_fp32_internal_not_fp16():
    """MM-01 / Pitfall 2: regression -- np.float16([1.0, 1e-4]*1000).sum() must NOT inf."""
    from riscv.gtx.gemm_core import gemm_reduce_sum_a
    # Pitfall 2 textbook case: long vector, mixed magnitudes.
    # FP16-internal accumulate would inf or saturate; FP32 stays finite.
    arr = np.array([1.0, 1e-4] * 1000, dtype=np.float16)
    result = gemm_reduce_sum_a(arr)
    assert np.isfinite(result), \
        f"gemm_reduce_sum_a should accumulate in FP32, got {result}"
    # Expected ~1000 + 1000*1e-4 ~= 1000.1; FP16 input loses precision on 1e-4.
    assert 999.0 < result < 1001.0, f"sum out of expected range: {result}"
    assert isinstance(result, float), \
        f"gemm_reduce_sum_a must return Python float, got {type(result)}"


def test_handler_registry_has_all_10_mm_variants():
    """MM-02: 10 @handler entries (mm/mm_s/mm_o/mm_v/mm_t + mmc.* family) + 10 disasm mnemonics."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built -- registry uses pybind types")
    # Force ops package import to populate registry
    import riscv.gtx.unit.ins.ops  # noqa: F401
    from riscv.gtx import _registry
    table = _registry.collect_for_kind('custom0')
    # 5 funct3 entries each at funct7=0x00 (MM) and funct7=0x01 (MMC)
    assert 0x00 in table, "funct7=0x00 (MM family) missing from registry"
    assert 0x01 in table, "funct7=0x01 (MMC family) missing from registry"
    mm_keys = sorted(k for k in table[0x00].keys() if isinstance(k, int))
    mmc_keys = sorted(k for k in table[0x01].keys() if isinstance(k, int))
    assert mm_keys == [0, 1, 2, 3, 7], f"MM funct3 keys: {mm_keys}"
    assert mmc_keys == [0, 1, 2, 3, 7], f"MMC funct3 keys: {mmc_keys}"
    # Plan 04 deviation: a None-key handler IS present at funct7=0x00/0x01
    # (wrspr_gem5/rdspr_gem5 from spr.py P2). Plan 04 wired its rs1!=0 branch to
    # re-dispatch into the funct3-keyed MM/MMC handlers, so MM is reachable
    # despite the None-key precedence in npu.custom0.

    # Also verify all 10 disasm mnemonics are in the registry.
    # NOTE: pyspike's disasm_insn_t canonicalizes underscores to dots
    # (e.g. handler mnemonic 'mm_s' surfaces as 'mm.s' on the disasm object).
    # Compare against the canonical (dot) form when _RISCV_DISASM_AVAILABLE; the
    # offline NamedTuple fallback preserves the underscore form verbatim.
    from riscv.gtx import disasm as _gtx_disasm
    if _gtx_disasm._RISCV_DISASM_AVAILABLE:
        expected_mnemonics = {
            'mm', 'mm.s', 'mm.o', 'mm.v', 'mm.t',
            'mmc', 'mmc.s', 'mmc.o', 'mmc.v', 'mmc.t',
        }
    else:
        expected_mnemonics = {
            'mm', 'mm_s', 'mm_o', 'mm_v', 'mm_t',
            'mmc', 'mmc_s', 'mmc_o', 'mmc_v', 'mmc_t',
        }
    disasms = _registry.collect_disasms()
    registered_mnemonics = {d.name for d in disasms if hasattr(d, 'name')}
    missing = expected_mnemonics - registered_mnemonics
    assert not missing, \
        f"MM/MMC mnemonics missing from disasm registry: {missing}"


def test_exec_mm_basic_bit_exact():
    """MM-02: 16x16x16 mm bit-exact vs explicit 3-loop oracle via npu.custom0 dispatch."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required for GtxNpu instantiation")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRR
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = nest
    npu.warp.curr_id = spu

    # Stage A at ADDRA=0, B at ADDRB=0x200 (16*16 fp16 = 512 bytes apart)
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.lspr[nest][spu][LSPR_SPM_ADDRB] = 0x200
    npu.lspr[nest][spu][LSPR_SPM_ADDRR] = 0x400

    np.random.seed(0)
    A = np.random.randn(16, 16).astype(np.float16)
    B = np.random.randn(16, 16).astype(np.float16)

    # Pre-fill L1: A at fp16-offset 0, B at fp16-offset 0x100 (= byte 0x200)
    l1f16 = npu.mem.l1_f16(nest, spu)
    l1f16[0:256] = A.flatten()
    l1f16[0x100:0x100 + 256] = B.flatten()

    # rs1 = packed dims for 16x16x16 (row_A=16, col_A=16, col_B=16)
    rs1_packed = (16 << 48) | (16 << 16) | 16
    proc.get_state().XPR.write(1, rs1_packed)
    # funct3=2 (mm): xd=0, xs1=1, xs2=0 -> (0<<2)|(1<<1)|0 = 2
    insn = MockInsn(funct=0x00, rs1=1, rs2=0, xd=0, xs1=1, xs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0

    # Read result at ADDRR (byte 0x400 = fp16 offset 0x200)
    actual = l1f16[0x200:0x200 + 256].copy().reshape(16, 16)
    A_f32, B_f32 = A.astype(np.float32), B.astype(np.float32)
    expected_f32 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        for j in range(16):
            s = np.float32(0.0)
            for k in range(16):
                s += A_f32[i, k] * B_f32[k, j]
            expected_f32[i, j] = s
    expected = expected_f32.astype(np.float16)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_exec_mm_s_writes_fp32_to_addrc():
    """MM-02: mm_s writes FP32 result bytes to ADDRC (LSPR_SPM_ADDRC=0x902) staging."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id, npu.warp.curr_id = nest, spu

    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.lspr[nest][spu][LSPR_SPM_ADDRB] = 0x100
    npu.lspr[nest][spu][LSPR_SPM_ADDRC] = 0x200

    # 4x4 mm_s -- A and B all-ones-like for exact FP32
    A = np.array([[1.0] * 4] * 4, dtype=np.float16)
    B = np.array([[2.0] * 4] * 4, dtype=np.float16)
    l1f16 = npu.mem.l1_f16(nest, spu)
    l1f16[0:16] = A.flatten()
    l1f16[0x80:0x90] = B.flatten()  # fp16 offset 0x80 = byte 0x100

    rs1_packed = (4 << 48) | (4 << 16) | 4
    proc.get_state().XPR.write(1, rs1_packed)
    # funct3=0 (mm_s): xd=0, xs1=0, xs2=0
    insn = MockInsn(funct=0x00, rs1=1, xs1=0, xs2=0, xd=0)
    npu.custom0(proc, insn, 0, 0)

    # Read FP32 (4,4) from L1[ADDRC:] = byte 0x200
    l1bytes = npu.mem.l1_byte(nest, spu)
    actual_f32 = np.frombuffer(l1bytes[0x200:0x200 + 64].tobytes(),
                                dtype=np.float32).reshape(4, 4).copy()
    # Oracle: A @ B, A row sum = 4*1.0=4.0, * 2.0 = 8.0 (exact in FP32)
    expected = np.full((4, 4), 8.0, dtype=np.float32)
    np.testing.assert_array_equal(actual_f32, expected)


def test_exec_mm_o_writes_scalar_to_l0_be():
    """MM-02: mm_o writes scalar sum(A) to L0 in BIG-endian (gtx_npu_mm.cc:217-218)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id, npu.warp.curr_id = nest, spu
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.gspr[GSPR_GTX_OPERAND3] = 0  # L0 dest at slot 0

    # A = [1.0, 2.0, 3.0, 4.0]; sum = 10.0 -> FP16(10.0) = 0x4900
    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    npu.mem.l1_f16(nest, spu)[0:4] = A

    # rs1 packed: row_A=1, col_A=4, col_B=1
    rs1_packed = (1 << 48) | (4 << 16) | 1
    proc.get_state().XPR.write(1, rs1_packed)
    # funct3=1 (mm_o): xd=0, xs1=0, xs2=1
    insn = MockInsn(funct=0x00, rs1=1, xs1=0, xs2=1, xd=0)
    npu.custom0(proc, insn, 0, 0)

    l0 = npu.mem.l0_byte(nest, spu)
    # FP16(10.0) raw = 0x4900. BE: byte[0]=0x49, byte[1]=0x00
    assert l0[0] == 0x49, f"L0[0] (BE high byte) should be 0x49, got 0x{l0[0]:02x}"
    assert l0[1] == 0x00, f"L0[1] (BE low byte) should be 0x00, got 0x{l0[1]:02x}"
    # mxe_accum should hold the FP32 sum
    assert npu._mxe_accum[nest, spu] == np.float32(10.0)


def test_exec_mm_v_writes_dot_to_l0_le():
    """MM-02: mm_v writes scalar dot(A,B) to L0 in little-endian (asymmetry vs mm_o!)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, GSPR_GTX_OPERAND3
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id, npu.warp.curr_id = nest, spu
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.lspr[nest][spu][LSPR_SPM_ADDRB] = 0x100  # B at byte 0x100 = fp16 offset 0x80
    npu.gspr[GSPR_GTX_OPERAND3] = 0

    # A = [1, 2, 3, 4], B = [5, 6, 7, 8] -> dot = 5+12+21+32 = 70.0 -> FP16(70.0) = 0x5460
    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    B = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16)
    npu.mem.l1_f16(nest, spu)[0:4] = A
    npu.mem.l1_f16(nest, spu)[0x80:0x84] = B

    rs1_packed = (1 << 48) | (4 << 16) | 1
    proc.get_state().XPR.write(1, rs1_packed)
    # funct3=3 (mm_v): xd=0, xs1=1, xs2=1
    insn = MockInsn(funct=0x00, rs1=1, xs1=1, xs2=1, xd=0)
    npu.custom0(proc, insn, 0, 0)

    l0 = npu.mem.l0_byte(nest, spu)
    # FP16(70.0) raw = 0x5460. LE: byte[0]=0x60, byte[1]=0x54
    assert l0[0] == 0x60, f"L0[0] (LE low byte) should be 0x60, got 0x{l0[0]:02x}"
    assert l0[1] == 0x54, f"L0[1] (LE high byte) should be 0x54, got 0x{l0[1]:02x}"
    assert npu._mxe_accum[nest, spu] == np.float32(70.0)


def test_exec_mm_t_writes_transposed():
    """MM-02 / Pitfall D: mm_t writes C^T to ADDRR in NxM layout (NOT MxN)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so required")
    from riscv.gtx import GtxNpu
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRR
    from tests.gtx._mocks import MockProcessor, MockInsn

    npu = GtxNpu()
    proc = MockProcessor()
    npu.reset(proc)
    nest, spu = 0, 0
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id, npu.warp.curr_id = nest, spu
    npu.lspr[nest][spu][LSPR_SPM_ADDRA] = 0
    npu.lspr[nest][spu][LSPR_SPM_ADDRB] = 0x100
    npu.lspr[nest][spu][LSPR_SPM_ADDRR] = 0x200

    # 2x3 @ 3x2 = 2x2 result; mm_t writes 2x2 transposed = 2x2 same shape
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)  # (M=2, K=3)
    B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float16)  # (K=3, N=2)
    l1f16 = npu.mem.l1_f16(nest, spu)
    l1f16[0:6] = A.flatten()
    l1f16[0x80:0x86] = B.flatten()

    rs1_packed = (2 << 48) | (3 << 16) | 2  # row_A=2, col_A=3, col_B=2
    proc.get_state().XPR.write(1, rs1_packed)
    # funct3=7 (mm_t): xd=1, xs1=1, xs2=1 -> (1<<2)|(1<<1)|1 = 7
    insn = MockInsn(funct=0x00, rs1=1, xs1=1, xs2=1, xd=1)
    npu.custom0(proc, insn, 0, 0)

    # Oracle C = A @ B (2x2). mm_t writes C^T at ADDRR (layout N x M = 2x2 here)
    expected_C = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    # Read N*M FP16 from ADDRR (byte 0x200 = fp16 offset 0x100)
    actual_flat = l1f16[0x100:0x100 + 4].copy()  # 4 elements = N*M
    # Per Pitfall D: layout is N x M = 2x2; equivalent to C^T flat row-major.
    actual_T = actual_flat.reshape(2, 2)  # (N, M) = (2, 2)
    np.testing.assert_array_equal(actual_T.view(np.uint16),
                                   expected_C.T.view(np.uint16))


def test_decode_firmware_mm_args():
    """MM-03: rs1 packed = colB[63:48]|colA[31:16]|rowA[15:0],
    0=65536 per field (dim16 lambda)."""
    from riscv.gtx.mm_engine import decode_firmware_mm_args

    # Case 1: concrete 4x4x4 (matches mm_basic.S literal)
    d = decode_firmware_mm_args(0x0004_0000_0004_0004)
    assert d == {'row_A': 4, 'col_A': 4, 'col_B': 4}, f"4x4x4 case: {d}"

    # Case 2: Pitfall C -- per-field 0->65536 (NOT whole-word check)
    d = decode_firmware_mm_args(0)
    assert d == {'row_A': 0x10000, 'col_A': 0x10000, 'col_B': 0x10000}, \
        f"all-zero case: {d}"

    # Case 3: distinct values, exercise all three field positions
    rs1 = (0xABCD << 48) | (0x1234 << 16) | 0x5678
    d = decode_firmware_mm_args(rs1)
    assert d == {'row_A': 0x5678, 'col_A': 0x1234, 'col_B': 0xABCD}, \
        f"distinct fields: {d}"

    # Case 4: only col_A is zero -- dim16 promotes ONLY that field
    rs1 = (0xFFFF << 48) | 0xFFFF  # row_A=0xFFFF, col_A=0, col_B=0xFFFF
    d = decode_firmware_mm_args(rs1)
    assert d == {'row_A': 0xFFFF, 'col_A': 0x10000, 'col_B': 0xFFFF}, \
        f"only col_A zero: {d}"


def test_verify_minimal_be_fp16_pairs(tmp_path):
    """MM-05 / Pitfall 1: _verify_minimal.compare_hex uses BE FP16 bit-pair (verify.py:235)."""
    from tests.gtx._verify_minimal import compare_hex

    # FP16(1.0) = 0x3C00. BE bit-pair on disk: byte[0]=0x3C, byte[1]=0x00.
    actual_path = tmp_path / "actual.hex"
    golden_path = tmp_path / "golden.hex"
    # Write same content to both -> strict PASS
    actual_path.write_text("3c00\n")
    golden_path.write_text("3c00\n")
    passed, stats = compare_hex(str(actual_path), str(golden_path), strict=True)
    assert passed, f"identical files should strict-pass: {stats}"
    assert stats['exact_matches'] == 1
    assert stats['total_fp16'] == 1

    # Now mismatch by 1 ULP -- strict should FAIL even within tolerance
    actual_path.write_text("3c01\n")  # FP16(0x3C01) ~ 1.0009766
    passed, stats = compare_hex(str(actual_path), str(golden_path),
                                 ulp=1, strict=True)
    assert not passed, "strict mode requires exact match, got within_tolerance"
    assert stats['exact_matches'] == 0
    # Within ULP=1 tolerance, but strict fails (D-14)
    assert stats['within_tolerance'] == 1


def test_gemm_core_signature_stateless():
    """MM-01 / D-03: gemm_core is array-in/array-out, no npu/proc/insn dependency."""
    import inspect
    from riscv.gtx import gemm_core as gemm_core_mod
    sig = inspect.signature(gemm_core_mod.gemm_core)
    params = list(sig.parameters.keys())
    forbidden = {'npu', 'proc', 'insn', 'self', 'mem', 'memory'}
    for p in params:
        assert p not in forbidden, \
            f"gemm_core must be stateless (D-03), but found '{p}' parameter"
    src = inspect.getsource(gemm_core_mod)
    assert 'from .npu' not in src, "gemm_core must not depend on npu (leaf module)"
    assert 'from .memory' not in src, "gemm_core must not depend on memory (leaf module)"
    assert 'from .dispatch' not in src, "gemm_core must not depend on dispatch"
