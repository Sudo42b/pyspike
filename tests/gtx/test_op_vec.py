"""P5 VEC op unit tests -- Wave 1b plan 02 GREEN-fill (test_op_vec.py).

Covers VEC-01 (SASMD VS/IS), VEC-02 (DOT/VSUM precision), VEC-03 (CLAMP +
accum_v + arange_v + L0/L1 path branch), VEC-04 (exec_vec_scalar,
exec_scalar_imm, exec_vector_imm), VEC-05 (firmware_vec_op decode + rs2
staging).
"""
import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

# pylint: disable=import-error,wrong-import-position
from riscv.gtx import vec_engine
from riscv.gtx.encoding import (
    GTX_F7_VEC_SASMD, GTX_F7_VEC_DOT_SUM, GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRR,
    GSPR_GTX_OPERAND2,
)
from riscv.gtx.npu import GtxNpu

from tests.gtx._mocks import MockProcessor, MockInsn


def _new_npu():
    """GtxNpu with default L1 ADDRA/ADDRB/ADDRR offsets (avoids zero-collision)."""
    npu = GtxNpu()
    # Lay out three non-overlapping 256-FP16 (512-byte) regions in L1.
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0x0000
    npu.lspr[0][0][LSPR_SPM_ADDRB] = 0x1000
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0x2000
    return npu


def _seed_l1(npu, *, nest=0, spu=0, addr_key, values: np.ndarray) -> None:
    """Write FP16 array starting at LSPR[addr_key] (byte offset, LE FP16)."""
    addr = npu.lspr[nest][spu][addr_key]
    l1_f16 = npu.mem.l1_f16(nest, spu)
    l1_f16[addr // 2:addr // 2 + values.shape[0]] = values.astype(np.float16)


def _read_l1(npu, *, nest=0, spu=0, addr_key, n: int) -> np.ndarray:
    addr = npu.lspr[nest][spu][addr_key]
    l1_f16 = npu.mem.l1_f16(nest, spu)
    return l1_f16[addr // 2:addr // 2 + n].copy()


def _pack_fp16_low(scalar: np.float16) -> int:
    """rs2-low-16 packing: low 16 bits = FP16 bit pattern."""
    return int(np.float16(scalar).view(np.uint16))


def _pack_fp16_pair(low: np.float16, high: np.float16) -> int:
    """rs2 [hi:lo] = high<<16 | low (FP16 each, LE bit pattern)."""
    return (int(np.float16(high).view(np.uint16)) << 16) | int(np.float16(low).view(np.uint16))


def _make_insn(*, funct7: int, funct3: int, rs1_idx: int = 1,
                rs2_idx: int = 2, rd_idx: int = 0) -> MockInsn:
    """Compose a MockInsn with funct3 = (xd<<2)|(xs1<<1)|xs2."""
    xd = (funct3 >> 2) & 1
    xs1 = (funct3 >> 1) & 1
    xs2 = funct3 & 1
    return MockInsn(funct=funct7, rs1=rs1_idx, rs2=rs2_idx, rd=rd_idx,
                    xd=xd, xs1=xs1, xs2=xs2)


# =========================================================================
# VEC-01: SASMD VS / IS variants on funct7=0x10
# =========================================================================
def test_sasmd_vs_add():
    """VEC-01: add_vs (funct7=0x10, funct3=0) on L1 element-wise add with
    broadcast scalar from rs2 low-16."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.arange(16, dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 16)        # vec_size
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(2.5)))  # scalar = 2.5
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=0)
    vec_engine.firmware_vec_op(npu, proc, insn)

    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=16)
    expected = (a.astype(np.float32) + 2.5).astype(np.float16)
    assert np.array_equal(out, expected)


def test_sasmd_is_add():
    """VEC-01: add_is (funct7=0x10, funct3=4) on L0 SVR scalar broadcast.

    L0 IS path: input from L0 reg `rs1 & 0x1F`, output to L0 reg
    `gspr[GSPR_OPERAND3] & 0x1F`, scalar in rs2 low-16.
    """
    npu = _new_npu()
    proc = MockProcessor()
    # L0 input reg = 1 -> bytes [32..63] = 16 FP16 values
    l0 = npu.mem.l0_byte(0, 0)
    a = np.arange(16, dtype=np.float16)
    a_bytes = a.tobytes()  # LE FP16 pairs
    l0[32:32 + 32] = np.frombuffer(a_bytes, dtype=np.uint8)
    # Result reg = 2 -> bytes [64..95]
    npu.gspr[0x003] = 2  # GSPR_GTX_OPERAND3 = result reg
    proc._state.XPR.write(1, 1)         # rs1 low 5 = input L0 reg = 1
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(3.0)))  # scalar = 3.0
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=4, rd_idx=2)
    vec_engine.firmware_vec_op(npu, proc, insn)

    out = np.frombuffer(bytes(l0[64:96]), dtype=np.float16)
    expected = (a.astype(np.float32) + 3.0).astype(np.float16)
    assert np.array_equal(out, expected)


def test_sasmd_vs_sub():
    """VEC-01: sub_vs (funct7=0x10, funct3=1)."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 4)
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(5.0)))
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=1)
    vec_engine.firmware_vec_op(npu, proc, insn)

    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=4)
    assert out.tolist() == [5.0, 15.0, 25.0, 35.0]


def test_sasmd_vs_mul():
    """VEC-01: mul_vs (funct7=0x10, funct3=2)."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 4)
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(2.0)))
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=2)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=4)
    assert out.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_sasmd_vs_div():
    """VEC-01: div_vs (funct7=0x10, funct3=3)."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([8.0, 16.0, 32.0, 64.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 4)
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(4.0)))
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=3)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=4)
    assert out.tolist() == [2.0, 4.0, 8.0, 16.0]


# =========================================================================
# VEC-02: DOT / VSUM FP32-internal precision
# =========================================================================
def test_dot_fp32_internal():
    """VEC-02: DOT (funct7=0x1A, funct3=0) uses explicit Python for-loop
    FP32 accumulate via vec_core.dot_kernel.

    Source: gtx_npu_vec.cc:632 -- `case 0: vec_op = GTX_VEC_DOT;`.
    """
    npu = _new_npu()
    proc = MockProcessor()
    # Long vector with small + large values to exercise FP32-internal precision.
    n = 256
    a = np.array([1.0] * n, dtype=np.float16)
    b = np.array([0.01] * n, dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRB, values=b)
    proc._state.XPR.write(1, n)
    proc._state.XPR.write(2, 0)
    insn = _make_insn(funct7=GTX_F7_VEC_DOT_SUM, funct3=0)
    vec_engine.firmware_vec_op(npu, proc, insn)

    out_scalar = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=1)[0]
    s = np.float32(0.0)
    for i in range(n):
        s += np.float32(a[i]) * np.float32(b[i])
    expected = np.float16(s)
    assert out_scalar == expected


# =========================================================================
# VEC-03: CLAMP variants + L0/L1 path branch
# =========================================================================
def test_clamp_min_uses_gspr_operand2():
    """VEC-03: clamp_min_v (funct7=0x1F, funct3=0) reads scalar from
    GSPR_GTX_OPERAND2 low 16 bits per gtx_npu_vec.cc:233-242."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([1.0, 5.0, 0.5, 10.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 4)
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(2.0)))   # floor at 2
    insn = _make_insn(funct7=GTX_F7_VEC_CLAMP, funct3=0)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=4)
    assert out.tolist() == [2.0, 5.0, 2.0, 10.0]
    # rs2 must be staged into GSPR_OPERAND2:
    assert npu.gspr[GSPR_GTX_OPERAND2] == _pack_fp16_low(np.float16(2.0))


def test_accum_v_cumulative():
    """VEC-03: accum_v (funct7=0x1F, funct3=2) prefix sum.
    Source: gtx_npu_vec.cc:215-221."""
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 5)
    proc._state.XPR.write(2, 0)
    insn = _make_insn(funct7=GTX_F7_VEC_CLAMP, funct3=2)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=5)
    assert out.tolist() == [1.0, 3.0, 6.0, 10.0, 15.0]


def test_arange_v_start_step():
    """VEC-03: arange_v (funct7=0x1F, funct3=3) reads start (low 16) and
    step (high 16) from GSPR_GTX_OPERAND2.
    Source: gtx_npu_vec.cc:243-249."""
    npu = _new_npu()
    proc = MockProcessor()
    proc._state.XPR.write(1, 6)
    proc._state.XPR.write(2, _pack_fp16_pair(np.float16(2.0), np.float16(0.5)))  # start=2, step=0.5
    insn = _make_insn(funct7=GTX_F7_VEC_CLAMP, funct3=3)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=6)
    assert out.tolist() == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def test_l0_l1_path_branch():
    """VEC-03: funct3 & 4 selects L0 (immediate) path; funct3 & 3 selects sub-op.

    Source: gtx_npu_vec.cc:593-596 (funct7=0x18 ARITH branch).
    For funct7=0x18, funct3=4 (= add_ii L0 II path), the engine routes to
    L0 element-wise between two L0 SVR registers.
    """
    npu = _new_npu()
    proc = MockProcessor()
    l0 = npu.mem.l0_byte(0, 0)
    a = np.arange(16, dtype=np.float16)
    b = np.full(16, 1.0, dtype=np.float16)
    # a in L0 reg 1, b in L0 reg 2, result in L0 reg 3.
    l0[32:64] = np.frombuffer(a.tobytes(), dtype=np.uint8)
    l0[64:96] = np.frombuffer(b.tobytes(), dtype=np.uint8)
    npu.gspr[0x003] = 3  # GSPR_OPERAND3 = result reg
    proc._state.XPR.write(1, 1)
    proc._state.XPR.write(2, 2)
    insn = _make_insn(funct7=GTX_F7_VEC_ARITH, funct3=4)
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = np.frombuffer(bytes(l0[96:128]), dtype=np.float16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    assert np.array_equal(out, expected)


# =========================================================================
# VEC-04: exec_vec_scalar / exec_scalar_imm / exec_vector_imm
# =========================================================================
def test_exec_vec_scalar():
    """VEC-04: exec_vec_scalar (L1 VV path -- element-wise across two L1 vectors).

    Source: gtx_npu_vec.cc:283-342. funct7=0x18 (ARITH), funct3=0..3 routes
    to `exec_vector_op` with vec_op=GTX_VEC_ADD/SUB/MUL/DIV using ADDRA + ADDRB.
    """
    npu = _new_npu()
    proc = MockProcessor()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float16)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRB, values=b)
    proc._state.XPR.write(1, 4)
    proc._state.XPR.write(2, 0)
    insn = _make_insn(funct7=GTX_F7_VEC_ARITH, funct3=2)  # mul_vv
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=4)
    assert out.tolist() == [10.0, 40.0, 90.0, 160.0]


def test_exec_scalar_imm():
    """VEC-04: exec_scalar_imm (L0 IS path -- scalar broadcast over 16 L0 FP16).

    Source: gtx_npu_vec.cc:352-402. funct7=0x10, funct3=4..7 routes to
    L0 IS path (16-element block at L0 reg).
    """
    npu = _new_npu()
    proc = MockProcessor()
    l0 = npu.mem.l0_byte(0, 0)
    a = np.arange(16, dtype=np.float16) * 0.5  # 0, 0.5, 1, 1.5, ..., 7.5
    l0[0:32] = np.frombuffer(a.tobytes(), dtype=np.uint8)  # input reg 0
    npu.gspr[0x003] = 1  # result reg 1
    proc._state.XPR.write(1, 0)         # rs1 low 5 = input L0 reg = 0
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(2.0)))  # scalar = 2.0
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=6, rd_idx=1)  # mul_is
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = np.frombuffer(bytes(l0[32:64]), dtype=np.float16)
    expected = (a.astype(np.float32) * 2.0).astype(np.float16)
    assert np.array_equal(out, expected)


def test_exec_vector_imm():
    """VEC-04: exec_vector_imm (L0 II path -- element-wise across two SVR regs).

    Already covered by test_l0_l1_path_branch but verified here separately
    with a different op (sub_ii at funct7=0x18, funct3=5).
    """
    npu = _new_npu()
    proc = MockProcessor()
    l0 = npu.mem.l0_byte(0, 0)
    a = np.full(16, 10.0, dtype=np.float16)
    b = np.arange(16, dtype=np.float16)
    l0[32:64] = np.frombuffer(a.tobytes(), dtype=np.uint8)   # reg 1
    l0[64:96] = np.frombuffer(b.tobytes(), dtype=np.uint8)   # reg 2
    npu.gspr[0x003] = 3
    proc._state.XPR.write(1, 1)
    proc._state.XPR.write(2, 2)
    insn = _make_insn(funct7=GTX_F7_VEC_ARITH, funct3=5)  # sub_ii
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = np.frombuffer(bytes(l0[96:128]), dtype=np.float16)
    expected = (a.astype(np.float32) - b.astype(np.float32)).astype(np.float16)
    assert np.array_equal(out, expected)


# =========================================================================
# VEC-05: firmware_vec_op packed-rs1 decode + rs2 GSPR staging
# =========================================================================
def test_firmware_vec_op_decode():
    """VEC-05: rs1 = vec_size in low 16 bits; HW conv 0 -> 0x10000.

    Source: gtx_npu_vec.cc:572-580. We test the basic decode by exercising
    a modest vec_size = 8 via SASMD VS path; result count must equal 8.
    """
    npu = _new_npu()
    proc = MockProcessor()
    a = np.arange(8, dtype=np.float16)
    _seed_l1(npu, addr_key=LSPR_SPM_ADDRA, values=a)
    proc._state.XPR.write(1, 8)
    proc._state.XPR.write(2, _pack_fp16_low(np.float16(1.0)))
    insn = _make_insn(funct7=GTX_F7_VEC_SASMD, funct3=0)  # add_vs
    vec_engine.firmware_vec_op(npu, proc, insn)
    out = _read_l1(npu, addr_key=LSPR_SPM_ADDRR, n=10)
    expected_8 = (a.astype(np.float32) + 1.0).astype(np.float16)
    assert np.array_equal(out[:8], expected_8)
    # Tail beyond vec_size must NOT be mutated (still zero).
    assert out[8] == np.float16(0.0)
    assert out[9] == np.float16(0.0)


def test_firmware_vec_op_stages_rs2():
    """VEC-05: rs2 read from XPR via proc.state.XPR[insn.rs2], staged into
    npu.gspr[GSPR_GTX_OPERAND2] for CLAMP / ARANGE / scalar SASMD ops.

    Source: gtx_npu_vec.cc:736-737.
    """
    npu = _new_npu()
    proc = MockProcessor()
    rs2_value = 0xDEADBEEF
    proc._state.XPR.write(1, 1)         # vec_size = 1 (minimal)
    proc._state.XPR.write(2, rs2_value)
    insn = _make_insn(funct7=GTX_F7_VEC_DOT_SUM, funct3=1)  # vsum (no scalar needed)
    vec_engine.firmware_vec_op(npu, proc, insn)
    assert npu.gspr[GSPR_GTX_OPERAND2] == rs2_value
