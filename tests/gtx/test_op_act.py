"""P5 ACT op unit tests -- Wave 0 RED scaffolds (test_op_act.py).

Covers ACT-01 (RELU/SOFTMAX/ESUM forward), ACT-02 (PRELU/GELU/TANH/SIGM
reversed; direction asymmetry), ACT-05 (_imm L0 path variants).

Wave 1b plan 03 GREEN-fills these. Plan 01 ships pytest.skip(...) bodies
per P3 plan-01 D-5 lock.
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
# ACT-01: Forward-direction activations (ADDRA -> ADDRR)
# =========================================================================
def test_relu_forward_direction():
    """ACT-01: RELU forward writes max(0, ADDRA[i]) to ADDRR.
    Source: gtx_npu_act.cc (forward branch)."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.relu + forward direction")


def test_softmax_forward():
    """ACT-01: SOFTMAX forward (max + exp + sum + normalize, ADDRA -> ADDRR).
    Source: gtx_npu_act.cc:78-100."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.softmax")


def test_esum_writes_l0_scalar():
    """ACT-01: ESUM writes a single FP16 scalar to L0 at offset
    `(gspr[GSPR_OPERAND3] & 0x1F) * 32` -- NOT ADDRR.
    Source: gtx_npu_act.cc:133-148. Pitfall 8."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: ESUM L0 scalar writeback")


# =========================================================================
# ACT-02: Reversed-direction activations (ADDRR -> ADDRA)
# =========================================================================
def test_prelu_reversed_direction():
    """ACT-02: PRELU reversed reads ADDRR, writes ADDRA.
    Source: gtx_npu_act.cc:37-42 direction asymmetry."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.prelu + reversed direction")


def test_gelu_reversed_direction():
    """ACT-02: GELU reversed reads ADDRR, writes ADDRA."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.gelu + reversed direction")


def test_tanh_reversed_direction():
    """ACT-02: TANH reversed reads ADDRR, writes ADDRA."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.tanh_act + reversed direction")


def test_sigm_reversed_direction():
    """ACT-02: SIGMOID reversed reads ADDRR, writes ADDRA."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_core.sigmoid + reversed direction")


def test_direction_asymmetry_table():
    """ACT-02: All 8 activations exercised with DISTINCT non-zero patterns at
    ADDRA AND ADDRR (Pitfall 3 / ROADMAP P5 success #2). Asserts that the
    correct buffer was overwritten matching the direction table."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: parametrized direction-asymmetry table")


# =========================================================================
# ACT-05: _imm L0 path variants
# =========================================================================
def test_act_imm_l0():
    """ACT-05: PRELU/GELU/TANH/SIGM _imm variants on L0 (funct3 & 4).
    Source: gtx_npu_act.cc:374-431 exec_act_imm."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: firmware_act_imm L0 path")


def test_softmax_imm_l0():
    """ACT-05: ESUM/SOFTMAX _imm variants on L0.
    Source: gtx_npu_act.cc:436-487 exec_softmax_imm."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: firmware_softmax_imm L0 path")


def test_act_funct3_l0_branch():
    """ACT-05: funct3 & 4 selects L0 immediate path; funct3 & 3 selects op.
    Source: RESEARCH Activation Direction Asymmetry table."""
    pytest.skip("Wave 1b plan 03 GREEN-fills: act_engine L0/L1 funct3 & 4 branch")
