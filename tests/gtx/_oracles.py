"""Host-side scalar oracles for VRF-02 parity.

Direct port of vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226 OPS dict.

VRF-02 OUTCOME (Plan 05 wave 2 GREEN):
  - 20 directly-mapped oracle bodies (FP32-internal compute, single FP16 cast).
  - DIRECT_MAPPED_ORACLES dict: op_name -> (oracle_fn, gtx_funct7, gtx_funct3, op_kind).
  - 10+ deferred oracles documented + skip-reasons table.

PORTABLE OPS (20 of 30) -- GREEN:
  Unary (12):     ABS, NEG, SQR, SQRT, EXP, LOG, CEIL, FLOOR, TRUNC, ROUND,
                  STEP, SGN
  Activation (4): RELU (forward), SIGMOID, TANH, GELU (reversed; tanh approx.)
  Binary (4):     ADD, SUB, MUL, DIV
  Scalar (2):     ADD1, SCALE  (op_add1 listed in DEFERRED_REASONS as redundant
                  with op_scale; op_scale is the canonical scalar entry)

DEFERRED OPS (10+ of 30) -- documented skip reasons:
  | Op           | Reason                                                          |
  |--------------|-----------------------------------------------------------------|
  | SIN          | NOT IMPLEMENTED in vendor exec_vector_op                        |
  | COS          | NOT IMPLEMENTED in vendor exec_vector_op                        |
  | SILU         | composed (x*sigmoid(x)); not a single GTX hardware op           |
  | GELU_ERF     | requires scipy.special.erf -- CLAUDE.md scipy ban (use op_gelu  |
  |              | tanh approximation, which IS bit-exact vs vendor act_core.gelu) |
  | GELU_QUICK   | composed; not a single GTX hardware op                          |
  | ELU          | composed                                                        |
  | SOFTPLUS     | composed                                                        |
  | LEAKY_RELU   | composed                                                        |
  | HARDSIGMOID  | composed                                                        |
  | HARDSWISH    | composed                                                        |
  | FILL         | P3 territory (DMA-01); already covered                          |
  | ADD1         | redundant with op_scale (broadcast scalar over add); kept       |
  |              | implemented for verify_ref parity but not in DIRECT_MAPPED      |

CLAUDE.md scipy ban: GELU_ERF body calls `pytest.skip(...)` to be safe; never
invoked in production tests.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Unary -- Plan 05 GREEN-fill (port from verify_ref.py:101-115 + :117)
# Each oracle: FP32 internal compute + single FP16 cast at writeback.
# ============================================================================
def op_abs(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """ABS: |a[i]| element-wise. Source: verify_ref.py:102."""
    return np.abs(a.astype(np.float32)).astype(np.float16)


def op_neg(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """NEG: -a[i] element-wise. Source: verify_ref.py:103."""
    return (-a.astype(np.float32)).astype(np.float16)


def op_sqr(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """SQR: a[i] * a[i] element-wise. Source: verify_ref.py:104.

    Synthesized via mul(a, a) -- shares funct7=0x18 with op_mul.
    """
    f32 = a.astype(np.float32)
    return (f32 * f32).astype(np.float16)


def op_sqrt(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """SQRT: sqrt(a[i]) element-wise. Source: verify_ref.py:105."""
    return np.sqrt(a.astype(np.float32)).astype(np.float16)


def op_exp(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """EXP: exp(a[i]) element-wise. Source: verify_ref.py:106."""
    return np.exp(a.astype(np.float32)).astype(np.float16)


def op_log(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """LOG: log(a[i]) element-wise. Source: verify_ref.py:107.

    Note: vendor exec_vector_op (gtx_npu_vec.cc:142) returns 0 for a <= 0;
    the host-side oracle here returns -inf/NaN per NumPy semantics. Tests
    must constrain inputs to a > 0 (op_oracle_parity uses |randn| + 0.1).
    """
    return np.log(a.astype(np.float32)).astype(np.float16)


def op_ceil(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """CEIL: ceil(a[i]) element-wise. Source: verify_ref.py:108."""
    return np.ceil(a.astype(np.float32)).astype(np.float16)


def op_floor(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """FLOOR: floor(a[i]) element-wise. Source: verify_ref.py:109."""
    return np.floor(a.astype(np.float32)).astype(np.float16)


def op_trunc(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """TRUNC: trunc(a[i]) element-wise. Source: verify_ref.py:110."""
    return np.trunc(a.astype(np.float32)).astype(np.float16)


def op_round(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """ROUND: rne(a[i]) element-wise -- banker's rounding (round-half-to-even).

    Source: verify_ref.py:111 (np.round). NumPy 2.x default rounding mode is RNE.
    """
    return np.rint(a.astype(np.float32)).astype(np.float16)


def op_step(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """STEP: 1.0 if a[i] > 0 else 0.0 element-wise. Source: verify_ref.py:112."""
    f32 = a.astype(np.float32)
    return np.where(f32 > np.float32(0.0),
                     np.float32(1.0), np.float32(0.0)).astype(np.float16)


def op_sgn(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """SGN: sign(a[i]) element-wise -- {-1, 0, +1}. Source: verify_ref.py:113."""
    return np.sign(a.astype(np.float32)).astype(np.float16)


def op_sin(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SIN is NOT IMPLEMENTED in vendor exec_vector_op."""
    raise NotImplementedError("DEFERRED: SIN not implemented in vendor exec_vector_op")


def op_cos(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: COS is NOT IMPLEMENTED in vendor exec_vector_op."""
    raise NotImplementedError("DEFERRED: COS not implemented in vendor exec_vector_op")


# ============================================================================
# Activations -- Plan 05 GREEN-fill
# ============================================================================
def op_relu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """RELU: max(0, a[i]) element-wise. Forward direction.

    Source: verify_ref.py:117-118. Matches act_core.relu output bit-exact.
    """
    f32 = a.astype(np.float32)
    return np.maximum(f32, np.float32(0.0)).astype(np.float16)


def op_silu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SILU = x * sigmoid(x); composed, not a single GTX hardware op."""
    raise NotImplementedError("DEFERRED: SILU is composed, not single GTX hardware op")


def op_sigmoid(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """SIGMOID: 1 / (1 + exp(-a[i])) element-wise. Reversed direction.

    Source: verify_ref.py:123-124. Matches act_core.sigmoid output bit-exact.
    """
    f32 = a.astype(np.float32)
    return (np.float32(1.0) /
            (np.float32(1.0) + np.exp(-f32))).astype(np.float16)


def op_tanh(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """TANH: tanh(a[i]) element-wise. Reversed direction.

    Source: verify_ref.py:126-127. Matches act_core.tanh_act output bit-exact.
    """
    return np.tanh(a.astype(np.float32)).astype(np.float16)


def op_gelu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """GELU (tanh approximation). Reversed direction.

    Source: verify_ref.py:129-131:
        a * 0.5 * (1.0 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))

    Matches act_core.gelu output bit-exact (same coefficients, same FP32-internal
    compute order). When verify_ref scipy-based op_gelu_erf is needed, this
    oracle is the bit-exact substitute (vendor uses the tanh approximation as
    primary; GELU_ERF is an alternate test-only formulation).
    """
    f32 = a.astype(np.float32)
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    inner = sqrt_2_over_pi * (f32 + np.float32(0.044715) * f32 * f32 * f32)
    return (np.float32(0.5) * f32 *
            (np.float32(1.0) + np.tanh(inner))).astype(np.float16)


def op_gelu_erf(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: GELU_ERF requires scipy.special.erf -- CLAUDE.md scipy ban.

    The GELU tanh approximation in op_gelu IS bit-exact vs vendor act_core.gelu;
    use op_gelu instead. Calling this oracle from a test triggers pytest.skip().
    """
    import pytest
    pytest.skip("GELU_ERF requires scipy.special.erf -- CLAUDE.md scipy ban; "
                "use op_gelu (tanh approximation) instead")


def op_gelu_quick(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: GELU_QUICK = a * sigmoid(1.702*a); composed, not single op."""
    raise NotImplementedError("DEFERRED: GELU_QUICK is composed, not single GTX hardware op")


def op_elu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: ELU is composed, not a single GTX hardware op."""
    raise NotImplementedError("DEFERRED: ELU is composed, not single GTX hardware op")


def op_softplus(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SOFTPLUS = log(1 + exp(a)); composed, not single op."""
    raise NotImplementedError("DEFERRED: SOFTPLUS is composed, not single GTX hardware op")


def op_leaky_relu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: LEAKY_RELU is composed, not a single GTX hardware op."""
    raise NotImplementedError("DEFERRED: LEAKY_RELU is composed, not single GTX hardware op")


def op_hardsigmoid(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: HARDSIGMOID is composed."""
    raise NotImplementedError("DEFERRED: HARDSIGMOID is composed")


def op_hardswish(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: HARDSWISH is composed."""
    raise NotImplementedError("DEFERRED: HARDSWISH is composed")


# ============================================================================
# Binary -- Plan 05 GREEN-fill (port from verify_ref.py:156-159)
# ============================================================================
def op_add(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """ADD: a[i] + b[i] element-wise. Source: verify_ref.py:156."""
    return (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)


def op_sub(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """SUB: a[i] - b[i] element-wise. Source: verify_ref.py:157."""
    return (a.astype(np.float32) - b.astype(np.float32)).astype(np.float16)


def op_mul(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """MUL: a[i] * b[i] element-wise. Source: verify_ref.py:158."""
    return (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)


def op_div(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """DIV: a[i] / b[i] element-wise. Source: verify_ref.py:159."""
    return (a.astype(np.float32) / b.astype(np.float32)).astype(np.float16)


# ============================================================================
# Scalar -- Plan 05 GREEN-fill (port from verify_ref.py:162-163)
# ============================================================================
def op_add1(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """ADD1: a[i] + scalar element-wise. Source: verify_ref.py:162.

    Listed in DEFERRED_REASONS as redundant with op_scale (different semantics
    but the same broadcast pattern; GTX maps add1 to add_is and scale to mul_is).
    Kept implemented for vendor parity; not registered in DIRECT_MAPPED_ORACLES
    to keep the test surface focused on op_scale (the more interesting case).
    """
    return (a.astype(np.float32) + np.float32(scalar)).astype(np.float16)


def op_scale(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """SCALE: a[i] * scalar element-wise. Source: verify_ref.py:163."""
    return (a.astype(np.float32) * np.float32(scalar)).astype(np.float16)


# ============================================================================
# Fill (deferred -- P3 territory)
# ============================================================================
def op_fill(n: int, scalar: np.float16) -> NDArray[np.float16]:
    """DEFERRED: FILL is P3 exec_fill, already covered by DMA-01."""
    raise NotImplementedError("DEFERRED: FILL is P3 exec_fill, already covered by DMA-01")


# ============================================================================
# DIRECT_MAPPED_ORACLES -- VRF-02 parametrize source-of-truth
# Each entry: op_name -> (oracle_fn, gtx_funct7, gtx_funct3, op_kind)
# op_kind ∈ {'vec_unary', 'vec_binary', 'vec_binary_aa', 'vec_scalar',
#            'act_forward_dispatch', 'act_reversed'}
# ============================================================================
DIRECT_MAPPED_ORACLES: dict = {
    # Unary VEC ops (funct7=0x1C MATH, 0x1D SIGN, 0x1E ROUND)
    'abs':    (op_abs,    0x1D, 0, 'vec_unary'),
    'neg':    (op_neg,    0x1D, 1, 'vec_unary'),
    'sgn':    (op_sgn,    0x1D, 2, 'vec_unary'),
    'step':   (op_step,   0x1D, 3, 'vec_unary'),
    'sqrt':   (op_sqrt,   0x1C, 0, 'vec_unary'),
    'exp':    (op_exp,    0x1C, 1, 'vec_unary'),
    'log':    (op_log,    0x1C, 2, 'vec_unary'),
    'ceil':   (op_ceil,   0x1E, 0, 'vec_unary'),
    'trunc':  (op_trunc,  0x1E, 1, 'vec_unary'),
    'floor':  (op_floor,  0x1E, 2, 'vec_unary'),
    'round':  (op_round,  0x1E, 3, 'vec_unary'),

    # SQR synthesized as mul(a, a) on funct7=0x18 (vec_binary on same buffer)
    'sqr':    (op_sqr,    0x18, 2, 'vec_binary_aa'),

    # Binary VEC arith (funct7=0x18 ARITH on L1 VV path)
    'add':    (op_add,    0x18, 0, 'vec_binary'),
    'sub':    (op_sub,    0x18, 1, 'vec_binary'),
    'mul':    (op_mul,    0x18, 2, 'vec_binary'),
    'div':    (op_div,    0x18, 3, 'vec_binary'),

    # Scalar broadcast (funct7=0x10 SASMD VS path; mul_vs == op_scale)
    'scale':  (op_scale,  0x10, 2, 'vec_scalar'),

    # Activations (firmware DISPATCH_ACT for RELU; reversed for sigmoid/tanh/gelu)
    'relu':   (op_relu,    0x06, 0, 'act_forward_dispatch'),
    'sigmoid':(op_sigmoid, 0x2D, 0, 'act_reversed'),
    'tanh':   (op_tanh,    0x2C, 0, 'act_reversed'),
    'gelu':   (op_gelu,    0x2A, 0, 'act_reversed'),
}

# Skip-reasons table for the 10+ deferred oracles (mirrored in module docstring).
DEFERRED_REASONS: dict = {
    'sin':         'NOT IMPLEMENTED in vendor exec_vector_op',
    'cos':         'NOT IMPLEMENTED in vendor exec_vector_op',
    'silu':        'composed (x*sigmoid(x)); not a single GTX hardware op',
    'gelu_erf':    'requires scipy.special.erf -- CLAUDE.md scipy ban; '
                   'op_gelu (tanh approximation) is bit-exact vs vendor',
    'gelu_quick':  'composed; not a single GTX hardware op',
    'elu':         'composed',
    'softplus':    'composed',
    'leaky_relu':  'composed',
    'hardsigmoid': 'composed',
    'hardswish':   'composed',
    'fill':        'P3 territory -- DMA-01 already covers',
    'add1':        'redundant with op_scale (broadcast scalar over add); '
                   'op_scale is the canonical scalar entry',
}
