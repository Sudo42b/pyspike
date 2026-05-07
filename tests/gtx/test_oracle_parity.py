"""VRF-02 oracle parity -- 20+ directly-mapped GTX ops match host-side oracles.

Source: vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226 OPS dict.
RESEARCH Adjustment 4: 30 oracles in vendor; 20 portable + 10 deferred.

Each parametrize entry:
  1. Generates a seeded random FP16 input vector of length 64.
  2. Sets up GtxNpu state (LSPR ADDRA/B/R + L1 input pre-load + MockProcessor XPR).
  3. Builds MockInsn with funct7/funct3 from DIRECT_MAPPED_ORACLES.
  4. Calls firmware_vec_op (vec_kind) or firmware_act (act_kind).
  5. Reads result from L1[ADDRR] (or ADDRA for reversed activations).
  6. Compares actual vs oracle(input) using compare_fp16(ulp=1, atol=0.001).

compare_fp16 thresholds match verify_ref.py:318-326 (ULP-1 + |delta| < 0.01).
We use atol=0.001 (tighter than verify_ref's 0.01) to match the project-wide
strict gate (ROADMAP P5 success criteria + verify.py default).
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
from riscv.gtx import vec_engine, act_engine
from riscv.gtx.encoding import (
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRR,
    GTX_ACT_RELU, GTX_ACT_SIGMOID, GTX_ACT_TANH, GTX_ACT_GELU,
)
from riscv.gtx.npu import GtxNpu

from tests.gtx._oracles import DIRECT_MAPPED_ORACLES
from tests.gtx._mocks import MockProcessor, MockInsn


# ---------------------------------------------------------------------------
# compare_fp16 -- direct port of verify_ref.py:318-326
# ---------------------------------------------------------------------------
def compare_fp16(actual, expected, ulp: int = 1, atol: float = 0.001) -> bool:
    """Port of verify_ref.py:318-326 -- ULP-1 + atol fallback comparator.

    NaN-NaN equivalence is treated as a match (vendor verify_ref doesn't
    explicitly handle this, but inputs to op_log etc. can introduce them).
    """
    a = float(actual)
    e = float(expected)
    if np.isnan(a) and np.isnan(e):
        return True
    if a == e:
        return True
    a_u16 = int(np.float16(actual).view(np.uint16))
    e_u16 = int(np.float16(expected).view(np.uint16))
    if abs(a_u16 - e_u16) <= ulp:
        return True
    if abs(a - e) < atol:
        return True
    return False


# ---------------------------------------------------------------------------
# Test setup helpers
# ---------------------------------------------------------------------------
def _new_npu():
    """GtxNpu with non-overlapping ADDRA/ADDRB/ADDRR layout in L1."""
    npu = GtxNpu()
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0x0000
    npu.lspr[0][0][LSPR_SPM_ADDRB] = 0x0200
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0x0400
    return npu


def _make_insn(*, funct7: int, funct3: int,
                rs1_idx: int = 1, rs2_idx: int = 2,
                rd_idx: int = 0) -> MockInsn:
    """Compose MockInsn. funct3 = (xd<<2) | (xs1<<1) | xs2 per RoCC encoding."""
    xd = (funct3 >> 2) & 1
    xs1 = (funct3 >> 1) & 1
    xs2 = funct3 & 1
    return MockInsn(funct=funct7, rs1=rs1_idx, rs2=rs2_idx, rd=rd_idx,
                    xd=xd, xs1=xs1, xs2=xs2)


def _domain_safe_input(op_name: str, n: int) -> np.ndarray:
    """Generate a seeded FP16 input vector that respects each op's domain.

    seed = hash(op_name) % 2**32 -> reproducible per op.
    """
    rng = np.random.default_rng(hash(op_name) % (2 ** 32))
    if op_name in ('sqrt', 'log'):
        # Strictly positive (op_log returns -inf at 0, NaN below).
        return (np.abs(rng.standard_normal(n)) +
                np.float32(0.1)).astype(np.float16)
    return (rng.standard_normal(n) * 2.0).astype(np.float16)


def _binary_b_input(op_name: str, n: int) -> np.ndarray:
    """B-operand for binary ops; avoids zero divisor for 'div'."""
    rng = np.random.default_rng((hash(op_name) ^ 0xDEADBEEF) % (2 ** 32))
    b = (rng.standard_normal(n) * 2.0 + 0.5).astype(np.float16)
    if op_name == 'div':
        # Replace |b| < 0.5 with 1.0 -- avoids near-zero division ULP blow-up.
        b = np.where(np.abs(b) < np.float16(0.5), np.float16(1.0), b)
    return b


# ---------------------------------------------------------------------------
# Parametrized parity test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_name", sorted(DIRECT_MAPPED_ORACLES.keys()))
def test_oracle_parity(op_name):
    """Each GTX op produces FP16 output bit-equal (ULP-1 / atol 0.001) to oracle.

    Runs the GTX engine entry on a 64-FP16 random input and compares to the
    matching `op_*` oracle in _oracles.py.
    """
    oracle_fn, funct7, funct3, kind = DIRECT_MAPPED_ORACLES[op_name]

    n = 64
    a = _domain_safe_input(op_name, n)
    scalar = np.float16(2.5)

    # ---- Compute oracle expected ----
    if kind == 'vec_binary':
        b = _binary_b_input(op_name, n)
        expected = oracle_fn(a, b)
    elif kind == 'vec_binary_aa':
        # SQR = mul(a, a) -- pass a as both operands; oracle is op_sqr(a).
        b = a.copy()
        expected = oracle_fn(a)
    elif kind == 'vec_scalar':
        expected = oracle_fn(a, scalar)
    else:
        expected = oracle_fn(a)

    # ---- Set up GtxNpu state ----
    npu = _new_npu()
    nest, spu = 0, 0
    addr_a = npu.lspr[nest][spu][LSPR_SPM_ADDRA]
    addr_b = npu.lspr[nest][spu][LSPR_SPM_ADDRB]
    addr_r = npu.lspr[nest][spu][LSPR_SPM_ADDRR]

    l1_f16 = npu.mem.l1_f16(nest, spu)
    l1_f16[addr_a // 2:addr_a // 2 + n] = a
    if kind in ('vec_binary', 'vec_binary_aa'):
        l1_f16[addr_b // 2:addr_b // 2 + n] = b

    # ---- Build instruction + processor state ----
    proc = MockProcessor()
    insn = _make_insn(funct7=funct7, funct3=funct3)
    proc._state.XPR.write(1, n)  # vec_size / length

    if kind == 'vec_scalar':
        # rs2 carries scalar in low-16 (FP16 LE bit pattern).
        proc._state.XPR.write(2, int(np.float16(scalar).view(np.uint16)))

    # ---- Dispatch into the appropriate engine entry ----
    if kind in ('vec_unary', 'vec_binary', 'vec_binary_aa', 'vec_scalar'):
        vec_engine.firmware_vec_op(npu, proc, insn)
        # Forward writeback -> L1[ADDRR]
        actual = l1_f16[addr_r // 2:addr_r // 2 + n].copy()

    elif kind == 'act_reversed':
        # Reversed activation reads ADDRR, writes ADDRA. Pre-clear ADDRA, load
        # input at ADDRR.
        op_id_map = {
            'sigmoid': GTX_ACT_SIGMOID,
            'tanh':    GTX_ACT_TANH,
            'gelu':    GTX_ACT_GELU,
        }
        op_id = op_id_map[op_name]
        l1_f16[addr_a // 2:addr_a // 2 + n] = np.zeros(n, dtype=np.float16)
        l1_f16[addr_r // 2:addr_r // 2 + n] = a
        act_engine.firmware_act(npu, proc, insn, op_id=op_id, is_reversed=True)
        # Reversed writeback -> L1[ADDRA]
        actual = l1_f16[addr_a // 2:addr_a // 2 + n].copy()

    elif kind == 'act_forward_dispatch':
        # RELU forward via firmware_act op_id=GTX_ACT_RELU.
        # firmware_act reads ADDRA (already loaded), writes ADDRR.
        act_engine.firmware_act(npu, proc, insn,
                                  op_id=GTX_ACT_RELU, is_reversed=False)
        actual = l1_f16[addr_r // 2:addr_r // 2 + n].copy()

    else:
        pytest.fail(f"unknown op_kind: {kind}")

    # ---- Element-wise comparison with diagnostic on first mismatch ----
    for i in range(n):
        if not compare_fp16(actual[i], expected[i], ulp=1, atol=0.001):
            a_u16 = int(np.float16(actual[i]).view(np.uint16))
            e_u16 = int(np.float16(expected[i]).view(np.uint16))
            input_val = a[i] if kind != 'act_reversed' else a[i]
            pytest.fail(
                f"{op_name}[{i}]: actual={float(actual[i]):.6f}(0x{a_u16:04x}) "
                f"expected={float(expected[i]):.6f}(0x{e_u16:04x}) "
                f"input={float(input_val):.6f} delta_ulp={abs(a_u16 - e_u16)}"
            )
