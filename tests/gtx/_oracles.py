"""Host-side scalar oracles for VRF-02 parity.

Direct port of vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226 OPS dict.

Plan 05 wave 2 fills bodies (~30 LOC each). Plan 01 ships skeleton:
each function has signature + docstring + raises NotImplementedError.

DEFERRED OPS (10 of 30) -- documented but not implemented:
  - GELU_ERF: requires scipy.special.erf (CLAUDE.md scipy ban). pytest skip.
  - SILU, GELU_QUICK, ELU, SOFTPLUS, LEAKY_RELU, HARDSIGMOID, HARDSWISH:
      composed ops, not single GTX hardware ops; not exercised by VEC/ACT
      direct dispatch.
  - SIN, COS: NOT IMPLEMENTED in C++ exec_vector_op.

PORTABLE OPS (20 of 30) -- Plan 05 GREEN-fills:
  - Unary (12): ABS, NEG, SQR, SQRT, EXP, LOG, CEIL, FLOOR, TRUNC, ROUND,
                STEP, SGN, RELU
  - Activation (3): SIGMOID, TANH, GELU
  - Binary (4): ADD, SUB, MUL, DIV
  - Scalar (2): ADD1, SCALE
  Plus FILL (P3 territory; documented as deferred to phase 3 coverage).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# Unary
def op_abs(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """|a[i]| element-wise. Plan 05 wave 2 GREEN-fill."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_neg(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """-a[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_sqr(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """a[i] * a[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_sqrt(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """sqrt(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_exp(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """exp(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_log(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """log(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_ceil(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """ceil(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_floor(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """floor(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_trunc(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """trunc(a[i]) element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_round(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """round(a[i]) RNE element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_step(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """1.0 if a[i] > 0 else 0.0 element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_sgn(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """sign(a[i]) element-wise: -1, 0, +1."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_sin(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SIN not implemented in vendor exec_vector_op."""
    raise NotImplementedError("DEFERRED: SIN not implemented in vendor exec_vector_op")


def op_cos(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: COS not implemented in vendor exec_vector_op."""
    raise NotImplementedError("DEFERRED: COS not implemented in vendor exec_vector_op")


def op_relu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """max(0, a[i]) element-wise. Forward-direction activation."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


# Activations
def op_silu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SILU is composed (x * sigmoid(x)), not a single GTX hardware op."""
    raise NotImplementedError("DEFERRED: SILU is composed, not single GTX hardware op")


def op_sigmoid(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """sigmoid(a[i]) element-wise. Reversed-direction activation."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_tanh(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """tanh(a[i]) element-wise. Reversed-direction activation."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_gelu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """GELU (tanh approximation). Reversed-direction activation."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_gelu_erf(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: GELU_ERF requires scipy.special.erf; CLAUDE.md scipy ban."""
    import pytest
    pytest.skip("GELU_ERF requires scipy.special.erf -- CLAUDE.md scipy ban")


def op_gelu_quick(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: GELU_QUICK is composed, not single GTX hardware op."""
    raise NotImplementedError("DEFERRED: GELU_QUICK is composed, not single GTX hardware op")


def op_elu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: ELU is composed, not single GTX hardware op."""
    raise NotImplementedError("DEFERRED: ELU is composed, not single GTX hardware op")


def op_softplus(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: SOFTPLUS is composed, not single GTX hardware op."""
    raise NotImplementedError("DEFERRED: SOFTPLUS is composed, not single GTX hardware op")


def op_leaky_relu(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: LEAKY_RELU is composed, not single GTX hardware op."""
    raise NotImplementedError("DEFERRED: LEAKY_RELU is composed, not single GTX hardware op")


# Sat-clip (deferred -- composed, not single hardware ops)
def op_hardsigmoid(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: HARDSIGMOID is composed."""
    raise NotImplementedError("DEFERRED: HARDSIGMOID is composed")


def op_hardswish(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """DEFERRED: HARDSWISH is composed."""
    raise NotImplementedError("DEFERRED: HARDSWISH is composed")


# Binary
def op_add(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """a[i] + b[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_sub(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """a[i] - b[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_mul(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """a[i] * b[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_div(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
    """a[i] / b[i] element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


# Scalar
def op_add1(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """a[i] + scalar element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


def op_scale(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """a[i] * scalar element-wise."""
    raise NotImplementedError("Plan 05 wave 2 GREEN-fill")


# Fill (deferred -- P3 territory, already covered by DMA-01)
def op_fill(n: int, scalar: np.float16) -> NDArray[np.float16]:
    """DEFERRED: FILL is P3 exec_fill, already covered by DMA-01."""
    raise NotImplementedError("DEFERRED: FILL is P3 exec_fill, already covered by DMA-01")


# Plan 05 builds DIRECT_MAPPED_ORACLES dict (20 entries) for parametrize.
# Plan 01 ships the empty dict so Plan 05 can fill it without
# touching multiple sites:
DIRECT_MAPPED_ORACLES: dict = {}  # Plan 05 fills with 20 entries
