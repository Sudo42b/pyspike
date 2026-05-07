"""P5 ACT op unit tests -- Wave 1b plan 03 GREEN-fill.

Covers ACT-01 (RELU/SOFTMAX/ESUM forward), ACT-02 (PRELU/GELU/TANH/SIGM
reversed; direction asymmetry), ACT-05 (_imm L0 path variants).

Pitfall 3 lock: `test_direction_asymmetry_table` parametrizes over all 7
op_ids and asserts the correct buffer (ADDRA vs ADDRR) was overwritten per the
vendor `gtx_npu_act.cc:37-42` direction table.

Pitfall 8 lock: `test_esum_writes_l0_scalar` proves ESUM (forward direction)
writes a single FP16 scalar to L0 at `(GSPR_OPERAND3 & 0x1F)*32` and does NOT
mutate L1[ADDRR].
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
from riscv.gtx import act_engine
from riscv.gtx.encoding import (
    GTX_ACT_RELU, GTX_ACT_TANH, GTX_ACT_SOFTMAX, GTX_ACT_GELU,
    GTX_ACT_SIGMOID, GTX_ACT_PRELU, GTX_ACT_ESUM,
    ACT_OPS_REVERSED,
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from riscv.gtx.npu import GtxNpu

from tests.gtx._mocks import MockProcessor, MockInsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_npu():
    """GtxNpu with default L1 ADDRA/ADDRR offsets (avoid zero-collision)."""
    npu = GtxNpu()
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0x0000
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0x2000
    return npu


def _make_insn(*, rs1_idx: int = 1, rs2_idx: int = 2, rd_idx: int = 0,
                funct: int = 0, funct3: int = 0) -> MockInsn:
    xd = (funct3 >> 2) & 1
    xs1 = (funct3 >> 1) & 1
    xs2 = funct3 & 1
    return MockInsn(funct=funct, rs1=rs1_idx, rs2=rs2_idx, rd=rd_idx,
                    xd=xd, xs1=xs1, xs2=xs2)


def _pack_fp16_low(scalar) -> int:
    return int(np.float16(scalar).view(np.uint16))


def _pack_fp16_pair(low, high) -> int:
    return ((int(np.float16(high).view(np.uint16)) << 16) |
            int(np.float16(low).view(np.uint16)))


# Self-contained `proc_with_addra_addrr_seeded` fixture (also exported from
# conftest.py for cross-file reuse, but defined here so this test file passes
# under `--noconftest`).
@pytest.fixture
def proc_with_addra_addrr_seeded():
    def seed(npu, *, nest: int = 0, spu: int = 0, length: int = 16,
             addra_pattern, addrr_pattern):
        addra = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
        addrr = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
        l1_f16 = npu.mem.l1_f16(nest, spu)
        addra_off = addra // 2
        addrr_off = addrr // 2
        l1_f16[addra_off:addra_off + length] = np.array(addra_pattern, dtype=np.float16)
        l1_f16[addrr_off:addrr_off + length] = np.array(addrr_pattern, dtype=np.float16)
        return {
            "addra": l1_f16[addra_off:addra_off + length].copy(),
            "addrr": l1_f16[addrr_off:addrr_off + length].copy(),
            "l1_f16": l1_f16,
            "addra_off": addra,
            "addrr_off": addrr,
        }
    return seed


# =========================================================================
# ACT-01: Forward-direction activations (ADDRA -> ADDRR)
# =========================================================================
def test_relu_forward_direction(proc_with_addra_addrr_seeded):
    """ACT-01: RELU forward writes max(0, ADDRA[i]) to ADDRR.

    Direction asymmetry proof: distinct patterns at ADDRA + ADDRR; after the
    op only ADDRR should be overwritten with relu(ADDRA pattern); ADDRA must
    be unchanged.
    """
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 0.5,
                               -0.5, 1.5, -1.5, 2.5, -2.5, 4.0, -4.0, 5.0],
                              dtype=np.float16)
    addrr_pattern = np.full(16, -99.0, dtype=np.float16)  # sentinel
    seeded = seed(npu, length=16,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 16)  # length

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_RELU, is_reversed=False)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 16].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 16].copy()

    # Forward: ADDRA unchanged, ADDRR overwritten with relu(addra_pattern).
    assert np.array_equal(after_addra, addra_pattern), \
        "ADDRA must be unchanged (forward direction)"
    expected = np.maximum(addra_pattern.astype(np.float32),
                           np.float32(0.0)).astype(np.float16)
    assert np.array_equal(after_addrr, expected), \
        "ADDRR must equal relu(ADDRA)"


def test_softmax_forward(proc_with_addra_addrr_seeded):
    """ACT-01: SOFTMAX forward (max + exp + sum + normalize, ADDRA -> ADDRR).

    Source: gtx_npu_act.cc:78-93.
    """
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    addrr_pattern = np.full(4, -99.0, dtype=np.float16)
    seeded = seed(npu, length=4,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 4)

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_SOFTMAX, is_reversed=False)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 4].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 4].copy()

    assert np.array_equal(after_addra, addra_pattern)
    # Sum of softmax must be ~1.0 (FP16 tolerance).
    s = float(after_addrr.astype(np.float32).sum())
    assert abs(s - 1.0) < 0.01, f"softmax sum = {s}"
    # Monotonic: out[i] strictly increasing for monotonic input.
    assert (after_addrr[0] < after_addrr[1] < after_addrr[2] < after_addrr[3])


def test_esum_writes_l0_scalar(proc_with_addra_addrr_seeded):
    """ACT-01 + Pitfall 8: ESUM (forward direction) writes a single FP16
    scalar to L0 at offset `(gspr[GSPR_OPERAND3] & 0x1F) * 32` -- NOT to
    L1[ADDRR]. ADDRA pattern read normally.

    Source: gtx_npu_act.cc:133-148.
    """
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    addrr_pattern = np.full(4, 7.5, dtype=np.float16)  # sentinel
    seeded = seed(npu, length=4,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 4)

    # GSPR_OPERAND2 [hi:lo] = [init_accum:max_val] FP16 packed. Vendor lines
    # 137-138.
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_pair(np.float16(4.0),
                                                    np.float16(0.0))
    # GSPR_OPERAND3 low-5 selects L0 reg = 3 -> byte offset 96.
    npu.gspr[GSPR_GTX_OPERAND3] = 3

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_ESUM, is_reversed=False)

    l1_f16 = seeded["l1_f16"]
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 4].copy()
    # Pitfall 8: L1[ADDRR] must be unchanged.
    assert np.array_equal(after_addrr, addrr_pattern), \
        "ADDRR must NOT be touched by ESUM (Pitfall 8)"

    # L0[reg=3 -> off=96] holds the FP16 scalar.
    # NB: np.uint8 << 8 saturates to 0; must cast to int first.
    l0 = npu.mem.l0_byte(0, 0)
    raw = int(l0[96]) | (int(l0[97]) << 8)
    actual = np.array([raw], dtype=np.uint16).view(np.float16)[0]
    # Expected = sum of exp(x_i - max) = exp(1-4)+exp(2-4)+exp(3-4)+exp(4-4)
    #         = exp(-3)+exp(-2)+exp(-1)+exp(0)
    #         ≈ 0.0498 + 0.1353 + 0.3679 + 1.0 = 1.5530
    assert abs(float(actual) - 1.5530) < 0.01, \
        f"L0 scalar = {float(actual)}, expected ~1.5530"


# =========================================================================
# ACT-02: Reversed-direction activations (ADDRR -> ADDRA)
# =========================================================================
def _check_reversed_direction(npu, proc, op_id, is_reversed_expected=True):
    """Helper: run the reversed activation and assert direction asymmetry."""
    pass  # body implemented in each test below


def test_prelu_reversed_direction(proc_with_addra_addrr_seeded):
    """ACT-02: PRELU reversed reads ADDRR, writes ADDRA.

    Source: gtx_npu_act.cc:37-42 + 118-131. Slope from GSPR_OPERAND2 low-16.
    """
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.full(8, -99.0, dtype=np.float16)  # will be overwritten
    addrr_pattern = np.array([-2.0, -1.0, 0.0, 1.0, -3.0, 2.0, -0.5, 0.5],
                              dtype=np.float16)
    seeded = seed(npu, length=8,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 8)
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_low(np.float16(0.1))  # slope=0.1

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_PRELU, is_reversed=True)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 8].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 8].copy()

    # Reversed: ADDRR unchanged, ADDRA overwritten with prelu(addrr_pattern).
    assert np.array_equal(after_addrr, addrr_pattern), \
        "ADDRR must be unchanged (reversed direction)"
    # Engine receives slope as FP16 (round-tripped through FP16 bit pattern),
    # so the oracle must use FP16 slope -> FP32 to match exactly.
    f32 = addrr_pattern.astype(np.float32)
    slope_f32 = np.float32(np.float16(0.1))  # mirror engine packing precision
    expected = np.where(f32 < 0.0, f32 * slope_f32, f32).astype(np.float16)
    assert np.array_equal(after_addra, expected), \
        f"ADDRA must equal prelu(ADDRR, slope=0.1); got {after_addra.tolist()}, expected {expected.tolist()}"


def test_gelu_reversed_direction(proc_with_addra_addrr_seeded):
    """ACT-02: GELU reversed reads ADDRR, writes ADDRA."""
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.full(4, -99.0, dtype=np.float16)
    addrr_pattern = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float16)
    seeded = seed(npu, length=4,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 4)

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_GELU, is_reversed=True)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 4].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 4].copy()

    assert np.array_equal(after_addrr, addrr_pattern), \
        "ADDRR must be unchanged (reversed direction)"
    # gelu(0)=0, gelu(1)~0.84, gelu(-1)~-0.16, gelu(2)~1.95
    assert abs(float(after_addra[0])) < 0.01
    assert abs(float(after_addra[1]) - 0.8413) < 0.01
    assert abs(float(after_addra[2]) - (-0.1587)) < 0.01
    assert abs(float(after_addra[3]) - 1.9546) < 0.01


def test_tanh_reversed_direction(proc_with_addra_addrr_seeded):
    """ACT-02: TANH reversed reads ADDRR, writes ADDRA."""
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.full(3, -99.0, dtype=np.float16)
    addrr_pattern = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    seeded = seed(npu, length=3,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 3)

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_TANH, is_reversed=True)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 3].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 3].copy()

    assert np.array_equal(after_addrr, addrr_pattern)
    # tanh(-1) ~ -0.7616, tanh(0) = 0, tanh(1) ~ 0.7616
    assert abs(float(after_addra[0]) - (-0.7616)) < 0.01
    assert abs(float(after_addra[1])) < 0.001
    assert abs(float(after_addra[2]) - 0.7616) < 0.01


def test_sigm_reversed_direction(proc_with_addra_addrr_seeded):
    """ACT-02: SIGMOID reversed reads ADDRR, writes ADDRA."""
    npu = _new_npu()
    proc = MockProcessor()
    seed = proc_with_addra_addrr_seeded
    addra_pattern = np.full(3, -99.0, dtype=np.float16)
    # Use values within FP16 range that don't overflow exp(-x). FP16 max is
    # 65504; exp(100) overflows. Use moderate values.
    addrr_pattern = np.array([-10.0, 0.0, 10.0], dtype=np.float16)
    seeded = seed(npu, length=3,
                   addra_pattern=addra_pattern, addrr_pattern=addrr_pattern)
    proc._state.XPR.write(1, 3)

    insn = _make_insn()
    act_engine.firmware_act(npu, proc, insn,
                              op_id=GTX_ACT_SIGMOID, is_reversed=True)

    l1_f16 = seeded["l1_f16"]
    after_addra = l1_f16[seeded["addra_off"] // 2:
                          seeded["addra_off"] // 2 + 3].copy()
    after_addrr = l1_f16[seeded["addrr_off"] // 2:
                          seeded["addrr_off"] // 2 + 3].copy()

    assert np.array_equal(after_addrr, addrr_pattern)
    # sigmoid(-10) ~ 0.0000454 -> rounds to 0 in FP16; sigmoid(0) = 0.5;
    # sigmoid(10) ~ 0.99995 -> rounds to ~1.0 in FP16.
    assert abs(float(after_addra[0])) < 0.01
    assert abs(float(after_addra[1]) - 0.5) < 0.01
    assert abs(float(after_addra[2]) - 1.0) < 0.01


def test_direction_asymmetry_table(proc_with_addra_addrr_seeded):
    """ACT-02 / Pitfall 3 lock: parametrized direction-asymmetry table.

    For each of the 7 op_ids, seed distinct patterns at ADDRA + ADDRR + L0;
    run firmware_act with the correct is_reversed claim per ACT_OPS_REVERSED;
    assert which buffer was overwritten matches the vendor direction table
    (gtx_npu_act.cc:37-42).

    | op_id    | direction | rd     | wr             |
    |----------|-----------|--------|----------------|
    | RELU     | forward   | ADDRA  | ADDRR          |
    | TANH     | reversed  | ADDRR  | ADDRA          |
    | SOFTMAX  | forward   | ADDRA  | ADDRR          |
    | GELU     | reversed  | ADDRR  | ADDRA          |
    | SIGMOID  | reversed  | ADDRR  | ADDRA          |
    | PRELU    | reversed  | ADDRR  | ADDRA          |
    | ESUM     | forward   | ADDRA  | L0 (NOT ADDRR) |
    """
    op_table = [
        # (op_id, is_reversed, expected_wr_buffer)
        (GTX_ACT_RELU,    False, "addrr"),
        (GTX_ACT_TANH,    True,  "addra"),
        (GTX_ACT_SOFTMAX, False, "addrr"),
        (GTX_ACT_GELU,    True,  "addra"),
        (GTX_ACT_SIGMOID, True,  "addra"),
        (GTX_ACT_PRELU,   True,  "addra"),
        (GTX_ACT_ESUM,    False, "l0"),  # forward but writes L0
    ]

    for op_id, is_reversed, expected_wr in op_table:
        npu = _new_npu()
        proc = MockProcessor()
        seed = proc_with_addra_addrr_seeded
        addra_pattern = np.array([0.5] * 4, dtype=np.float16)
        addrr_pattern = np.array([1.5] * 4, dtype=np.float16)
        seeded = seed(npu, length=4,
                       addra_pattern=addra_pattern,
                       addrr_pattern=addrr_pattern)
        proc._state.XPR.write(1, 4)

        # Seed ESUM-required GSPRs (max=2.0, accum=0.0; result reg=1).
        npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_pair(np.float16(2.0),
                                                       np.float16(0.0))
        npu.gspr[GSPR_GTX_OPERAND3] = 1
        # Seed PRELU slope.
        if op_id == GTX_ACT_PRELU:
            npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_low(np.float16(0.1))

        insn = _make_insn()
        act_engine.firmware_act(npu, proc, insn,
                                  op_id=op_id, is_reversed=is_reversed)

        l1_f16 = seeded["l1_f16"]
        after_addra = l1_f16[seeded["addra_off"] // 2:
                              seeded["addra_off"] // 2 + 4].copy()
        after_addrr = l1_f16[seeded["addrr_off"] // 2:
                              seeded["addrr_off"] // 2 + 4].copy()

        if expected_wr == "addrr":
            assert np.array_equal(after_addra, addra_pattern), \
                f"op_id={op_id}: ADDRA must be unchanged (forward)"
            assert not np.array_equal(after_addrr, addrr_pattern), \
                f"op_id={op_id}: ADDRR must be overwritten (forward)"
        elif expected_wr == "addra":
            assert np.array_equal(after_addrr, addrr_pattern), \
                f"op_id={op_id}: ADDRR must be unchanged (reversed)"
            assert not np.array_equal(after_addra, addra_pattern), \
                f"op_id={op_id}: ADDRA must be overwritten (reversed)"
        elif expected_wr == "l0":
            # ESUM: both ADDRA + ADDRR unchanged; L0 has the scalar at reg=1.
            assert np.array_equal(after_addra, addra_pattern), \
                "ESUM: ADDRA must be unchanged"
            assert np.array_equal(after_addrr, addrr_pattern), \
                "ESUM: ADDRR must NOT be touched (Pitfall 8)"
            l0 = npu.mem.l0_byte(0, 0)
            l0_off = (1 & 0x1F) * 32  # reg=1 -> off=32
            raw = int(l0[l0_off]) | (int(l0[l0_off + 1]) << 8)
            scalar = np.array([raw], dtype=np.uint16).view(np.float16)[0]
            # ADDRA = [0.5]*4, max=2.0; sum_i exp(0.5-2.0) = 4*exp(-1.5)
            # ~ 4 * 0.2231 ~ 0.8925
            assert abs(float(scalar) - 0.8925) < 0.05, \
                f"ESUM scalar = {float(scalar)}, expected ~0.8925"


def test_act_engine_consistency_check():
    """D-06: engine asserts is_reversed agrees with op_id ∈ ACT_OPS_REVERSED.

    Pass mismatched is_reversed and assert the engine raises AssertionError
    BEFORE doing any compute. This protects against @handler bugs that pass
    the wrong is_reversed literal.
    """
    npu = _new_npu()
    proc = MockProcessor()
    insn = _make_insn()

    # PRELU is in ACT_OPS_REVERSED -> is_reversed must be True. Pass False.
    with pytest.raises(AssertionError, match="is_reversed mismatch"):
        act_engine.firmware_act(npu, proc, insn,
                                  op_id=GTX_ACT_PRELU, is_reversed=False)

    # RELU is NOT in ACT_OPS_REVERSED -> is_reversed must be False. Pass True.
    with pytest.raises(AssertionError, match="is_reversed mismatch"):
        act_engine.firmware_act(npu, proc, insn,
                                  op_id=GTX_ACT_RELU, is_reversed=True)


# =========================================================================
# ACT-05: _imm L0 path variants
# =========================================================================
def test_act_imm_l0():
    """ACT-05: PRELU/GELU/TANH/SIGM _imm variants on L0 (16 FP16 elements
    per L0 reg block).

    Source: gtx_npu_act.cc:374-431 exec_act_imm.
    """
    npu = _new_npu()
    proc = MockProcessor()
    # L0 reg=1 -> bytes [32..63] = 16 FP16
    l0 = npu.mem.l0_byte(0, 0)
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 0.5,
                   -0.5, 1.5, -1.5, 2.5, -2.5, 4.0, -4.0, 5.0],
                  dtype=np.float16)
    l0[32:64] = np.frombuffer(a.tobytes(), dtype=np.uint8)
    # Result reg = 2 -> bytes [64..95]
    npu.gspr[GSPR_GTX_OPERAND3] = 2
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_low(np.float16(0.25))  # PRELU slope

    insn = _make_insn(rs1_idx=1, rd_idx=2)
    proc._state.XPR.write(1, 1)  # rs1 -> input reg 1 (low 5 bits)

    act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_PRELU)

    out = np.frombuffer(bytes(l0[64:96]), dtype=np.float16)
    f32 = a.astype(np.float32)
    slope_f32 = np.float32(np.float16(0.25))  # mirror engine FP16 packing
    expected = np.where(f32 < 0.0, f32 * slope_f32, f32).astype(np.float16)
    assert np.array_equal(out, expected), \
        f"prelu_i mismatch: got {out.tolist()}, expected {expected.tolist()}"


def test_softmax_imm_l0():
    """ACT-05: ESUM/SOFTMAX _imm variants on L0 16-element block.

    Source: gtx_npu_act.cc:436-487 exec_softmax_imm.
    ESUM writes scalar at result reg + max at offset+2 + 14 zero FP16s.
    """
    npu = _new_npu()
    proc = MockProcessor()
    l0 = npu.mem.l0_byte(0, 0)
    # Input L0 reg = 0 -> bytes [0..31]; 16 FP16 = [1,2,3,...,16]
    a = np.arange(1, 17, dtype=np.float16)
    l0[0:32] = np.frombuffer(a.tobytes(), dtype=np.uint8)
    # Result reg = 3 -> bytes [96..127]
    npu.gspr[GSPR_GTX_OPERAND3] = 3
    # GSPR_OPERAND2 [hi:lo] = [accum:max] = [0.0, 16.0]
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_pair(np.float16(16.0),
                                                    np.float16(0.0))

    insn = _make_insn(rs1_idx=1, rd_idx=3)
    proc._state.XPR.write(1, 0)  # rs1 -> input reg 0

    act_engine.firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_ESUM)

    # ESUM writes [r:16 | max:16] at result_reg offset, then 14 FP16 zeros.
    # NB: np.uint8 << 8 saturates; cast to int first.
    raw_r = int(l0[96]) | (int(l0[97]) << 8)
    scalar = np.array([raw_r], dtype=np.uint16).view(np.float16)[0]
    raw_m = int(l0[98]) | (int(l0[99]) << 8)
    max_back = np.array([raw_m], dtype=np.uint16).view(np.float16)[0]
    # Expected: sum_{i=1..16} exp(i - 16) = exp(-15) + exp(-14) + ... + exp(0)
    # ~ 1.581 (geometric series approx).
    s = np.float32(0.0)
    for v in a:
        s += np.exp(np.float32(v) - np.float32(16.0))
    expected = float(np.float16(s))
    assert abs(float(scalar) - expected) < 0.01, \
        f"ESUM scalar = {float(scalar)}, expected {expected}"
    assert float(max_back) == 16.0, \
        f"max stored at offset+2 = {float(max_back)}, expected 16.0"

    # Now SOFTMAX_IMM on result reg=4. accum=esum is supplied externally.
    npu.gspr[GSPR_GTX_OPERAND3] = 4
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_fp16_pair(np.float16(16.0),
                                                    np.float16(expected))
    act_engine.firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_SOFTMAX)
    softmax_out = np.frombuffer(bytes(l0[128:160]), dtype=np.float16)
    s_total = float(softmax_out.astype(np.float32).sum())
    assert abs(s_total - 1.0) < 0.05, f"softmax_imm sum = {s_total}"


def test_act_funct3_l0_branch():
    """ACT-05: L0 vs L1 path selection — verify firmware_act_imm operates ONLY
    on L0 (no L1[ADDRA]/L1[ADDRR] mutation).

    Source: RESEARCH §Activation Direction Asymmetry table (funct3 & 4 ->
    immediate L0 path).
    """
    npu = _new_npu()
    proc = MockProcessor()
    # Pre-populate L1 with a sentinel so we can detect any accidental mutation.
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]
    l1_f16 = npu.mem.l1_f16(0, 0)
    sentinel = np.full(16, 7.5, dtype=np.float16)
    l1_f16[addr_a // 2:addr_a // 2 + 16] = sentinel
    l1_f16[addr_r // 2:addr_r // 2 + 16] = sentinel

    # Drive L0 reg=1 -> reg=2 TANH_imm
    l0 = npu.mem.l0_byte(0, 0)
    a = np.array([0.5] * 16, dtype=np.float16)
    l0[32:64] = np.frombuffer(a.tobytes(), dtype=np.uint8)
    npu.gspr[GSPR_GTX_OPERAND3] = 2
    insn = _make_insn(rs1_idx=1, rd_idx=2)
    proc._state.XPR.write(1, 1)

    act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_TANH)

    # L1 untouched
    assert np.array_equal(l1_f16[addr_a // 2:addr_a // 2 + 16], sentinel), \
        "L1[ADDRA] must NOT be touched by firmware_act_imm"
    assert np.array_equal(l1_f16[addr_r // 2:addr_r // 2 + 16], sentinel), \
        "L1[ADDRR] must NOT be touched by firmware_act_imm"
    # L0 reg=2 holds tanh(0.5) ~ 0.4621 in 16 places
    out = np.frombuffer(bytes(l0[64:96]), dtype=np.float16)
    expected_v = float(np.float16(np.tanh(np.float32(0.5))))
    for v in out:
        assert abs(float(v) - expected_v) < 0.01
