"""P5 VEC op unit tests -- Wave 0 RED scaffolds (test_op_vec.py).

Covers VEC-01 (SASMD VS/IS), VEC-02 (DOT/VSUM precision), VEC-03 (CLAMP +
accum_v + arange_v + L0/L1 path branch), VEC-04 (exec_vec_scalar,
exec_scalar_imm, exec_vector_imm), VEC-05 (firmware_vec_op decode + rs2
staging).

Wave 1b plans 02-04 GREEN-fill these. Plan 01 ships pytest.skip(...) bodies
per P3 plan-01 D-5 lock (RED-via-skip -- never `assert hasattr(...)`).
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


# =========================================================================
# VEC-01: SASMD VS / IS variants on funct7=0x10
# =========================================================================
def test_sasmd_vs_add():
    """VEC-01: add_vs (funct7=0x10, funct3=0) on L1 element-wise add."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.sasmd_kernel + ops/vec.py @handler")


def test_sasmd_is_add():
    """VEC-01: add_is (funct7=0x10, funct3=4) on L0 SVR scalar broadcast."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: L0-path SASMD via funct3 & 4")


def test_sasmd_vs_sub():
    """VEC-01: sub_vs (funct7=0x10, funct3=1)."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.sasmd_kernel(op=SUB)")


def test_sasmd_vs_mul():
    """VEC-01: mul_vs (funct7=0x10, funct3=2)."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.sasmd_kernel(op=MUL)")


def test_sasmd_vs_div():
    """VEC-01: div_vs (funct7=0x10, funct3=3)."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.sasmd_kernel(op=DIV)")


# =========================================================================
# VEC-02: DOT / VSUM FP32-internal precision
# =========================================================================
def test_dot_fp32_internal():
    """VEC-02: DOT uses explicit Python for-loop FP32 accumulate (NOT np.dot
    or np.sum -- both use pairwise summation; ULP-different from C++ scalar
    accumulate per RESEARCH Pitfall 2)."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.dot_kernel FP32 internal")


# =========================================================================
# VEC-03: CLAMP variants + L0/L1 path branch
# =========================================================================
def test_clamp_min_uses_gspr_operand2():
    """VEC-03: clamp_min_v (funct7=0x1F, funct3=0) reads scalar from
    GSPR_GTX_OPERAND2 low 16 bits per gtx_npu_vec.cc:233-242."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_engine GSPR_OPERAND2 unpack")


def test_accum_v_cumulative():
    """VEC-03: accum_v (funct7=0x1F, funct3=2) prefix sum.
    Source: gtx_npu_vec.cc:215-221."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.accum_kernel")


def test_arange_v_start_step():
    """VEC-03: arange_v (funct7=0x1F, funct3=3) reads start (low 16) and
    step (high 16) from GSPR_GTX_OPERAND2.
    Source: gtx_npu_vec.cc:243-249."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_core.arange_kernel + GSPR unpack")


def test_l0_l1_path_branch():
    """VEC-03: funct3 & 4 selects L0 (immediate) path; funct3 & 3 selects sub-op.
    Source: gtx_npu_vec.cc:593-596."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_engine L0/L1 path branch")


# =========================================================================
# VEC-04: exec_vec_scalar / exec_scalar_imm / exec_vector_imm
# =========================================================================
def test_exec_vec_scalar():
    """VEC-04: exec_vec_scalar (L1 VS path scalar arith).
    Source: gtx_npu_vec.cc:283-342."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: VS L1 scalar broadcast")


def test_exec_scalar_imm():
    """VEC-04: exec_scalar_imm (L0 IS path).
    Source: gtx_npu_vec.cc:352-402."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: IS L0 scalar broadcast")


def test_exec_vector_imm():
    """VEC-04: exec_vector_imm (L0 II path -- element-wise across two SVR regs).
    Source: gtx_npu_vec.cc:410-454."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: II L0 element-wise")


# =========================================================================
# VEC-05: firmware_vec_op packed-rs1 decode + rs2 GSPR staging
# =========================================================================
def test_firmware_vec_op_decode():
    """VEC-05: rs1 = vec_size in low 16 bits; HW conv 0 -> 0x10000.
    Source: gtx_npu_vec.cc:572-580."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: vec_engine.firmware_vec_op decode")


def test_firmware_vec_op_stages_rs2():
    """VEC-05: rs2 read from XPR via proc.state.XPR[insn.rs2], staged into
    npu.gspr[GSPR_GTX_OPERAND2] for CLAMP / ARANGE / scalar SASMD ops.
    Source: gtx_npu_vec.cc:736-737."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: rs2 -> GSPR_OPERAND2 staging")
