"""NumPy implementations of common ggml ops.

Covers what a typical LLM forward pass needs: unary activations, elementwise
arithmetic, MUL_MAT, normalisations, softmax, GLU variants, GET_ROWS, ROPE,
SCALE/CLAMP/CPY, and the metadata-only view ops. Quantised types and the
heavyweight FLASH_ATTN_EXT / CONV ops are left as NotImplementedError —
clients can either avoid them or extend this table.

Entry point: `compute(node, sources)` where `node` is the RpcTensor for the
output and `sources` is the list of (already-resolved-to-numpy) input arrays.
Returns the numpy array to be written back to the output tensor.
"""

from __future__ import annotations

import math
import numpy as np

import op_registry as opr
from wire_protocol import RpcTensor


# ------------- helpers --------------------------------------------------------

def _f32(a: np.ndarray) -> np.ndarray:
    """Promote to float32 for math; the caller will cast back to dst dtype."""
    return a.astype(np.float32, copy=False)


def _safe_exp(x):
    return np.exp(np.minimum(x, 80.0))   # clamp to avoid overflow in fp16


# ------------- unary table (ggml_unary_op) -----------------------------------

UNARY_FN = {
    opr.GGML_UNARY_OP_ABS:         lambda x: np.abs(x),
    opr.GGML_UNARY_OP_SGN:         lambda x: np.sign(x),
    opr.GGML_UNARY_OP_NEG:         lambda x: -x,
    opr.GGML_UNARY_OP_STEP:        lambda x: (x > 0).astype(x.dtype),
    opr.GGML_UNARY_OP_TANH:        lambda x: np.tanh(x),
    opr.GGML_UNARY_OP_ELU:         lambda x: np.where(x > 0, x, _safe_exp(x) - 1.0),
    opr.GGML_UNARY_OP_RELU:        lambda x: np.maximum(x, 0),
    opr.GGML_UNARY_OP_SIGMOID:     lambda x: 1.0 / (1.0 + _safe_exp(-x)),
    opr.GGML_UNARY_OP_GELU:        lambda x: 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))),
    opr.GGML_UNARY_OP_GELU_QUICK:  lambda x: x * (1.0 / (1.0 + _safe_exp(-1.702 * x))),
    opr.GGML_UNARY_OP_SILU:        lambda x: x * (1.0 / (1.0 + _safe_exp(-x))),
    opr.GGML_UNARY_OP_HARDSWISH:   lambda x: x * np.clip(x + 3.0, 0, 6) / 6.0,
    opr.GGML_UNARY_OP_HARDSIGMOID: lambda x: np.clip(x + 3.0, 0, 6) / 6.0,
    opr.GGML_UNARY_OP_EXP:         lambda x: _safe_exp(x),
    opr.GGML_UNARY_OP_EXPM1:       lambda x: np.expm1(x),
    opr.GGML_UNARY_OP_SOFTPLUS:    lambda x: np.log1p(_safe_exp(x)),
    opr.GGML_UNARY_OP_GELU_ERF:    lambda x: 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2))),
    opr.GGML_UNARY_OP_XIELU:       lambda x: np.where(x > 0, x, _safe_exp(x) - 1.0),
    opr.GGML_UNARY_OP_FLOOR:       lambda x: np.floor(x),
    opr.GGML_UNARY_OP_CEIL:        lambda x: np.ceil(x),
    opr.GGML_UNARY_OP_ROUND:       lambda x: np.round(x),
    opr.GGML_UNARY_OP_TRUNC:       lambda x: np.trunc(x),
}


# ------------- glu table ------------------------------------------------------
# GLU ops split the last dim in half: x[..., :half] gated by act(x[..., half:]).

def _glu_split(a):
    half = a.shape[-1] // 2
    return a[..., :half], a[..., half:]


GLU_FN = {
    opr.GGML_GLU_OP_REGLU:       lambda x: _glu_split(x)[0] * np.maximum(_glu_split(x)[1], 0),
    opr.GGML_GLU_OP_GEGLU:       lambda x: _glu_split(x)[0] * (0.5 * _glu_split(x)[1] *
        (1.0 + np.tanh(math.sqrt(2.0 / math.pi) *
                       (_glu_split(x)[1] + 0.044715 * _glu_split(x)[1] ** 3)))),
    opr.GGML_GLU_OP_SWIGLU:      lambda x: _glu_split(x)[0] * (_glu_split(x)[1] *
        (1.0 / (1.0 + _safe_exp(-_glu_split(x)[1])))),
    opr.GGML_GLU_OP_SWIGLU_OAI:  lambda x: _glu_split(x)[0] * (_glu_split(x)[1] *
        (1.0 / (1.0 + _safe_exp(-_glu_split(x)[1])))),
    opr.GGML_GLU_OP_GEGLU_ERF:   lambda x: _glu_split(x)[0] * (0.5 * _glu_split(x)[1] *
        (1.0 + np.vectorize(math.erf)(_glu_split(x)[1] / math.sqrt(2)))),
    opr.GGML_GLU_OP_GEGLU_QUICK: lambda x: _glu_split(x)[0] * (_glu_split(x)[1] *
        (1.0 / (1.0 + _safe_exp(-1.702 * _glu_split(x)[1])))),
}


# ------------- main dispatcher -----------------------------------------------

def compute(node: RpcTensor, srcs: list[np.ndarray]) -> np.ndarray:
    """Run `node`'s op on the source numpy arrays. Returns dst-shaped result.

    Output dtype is the dst tensor's dtype (caller passes that via node.type
    and we cast at the end). Internal math is fp32 to dodge fp16 over/underflow.
    """
    op = node.op
    params = node.op_params

    # --- unary / glu meta-ops ---
    if op == opr.GGML_OP_UNARY:
        x = _f32(srcs[0])
        sub = params[0]
        fn = UNARY_FN.get(sub)
        if fn is None:
            raise NotImplementedError(f"GGML_OP_UNARY sub={sub}")
        return fn(x)

    if op == opr.GGML_OP_GLU:
        x = _f32(srcs[0])
        sub = params[0]
        fn = GLU_FN.get(sub)
        if fn is None:
            raise NotImplementedError(f"GGML_OP_GLU sub={sub}")
        return fn(x)

    # --- binary elementwise ---
    if op == opr.GGML_OP_ADD:
        return _f32(srcs[0]) + _f32(srcs[1])
    if op == opr.GGML_OP_SUB:
        return _f32(srcs[0]) - _f32(srcs[1])
    if op == opr.GGML_OP_MUL:
        return _f32(srcs[0]) * _f32(srcs[1])
    if op == opr.GGML_OP_DIV:
        return _f32(srcs[0]) / _f32(srcs[1])
    if op == opr.GGML_OP_ACC:
        return _f32(srcs[0]) + _f32(srcs[1])    # accumulator

    # --- unary-shaped direct ops ---
    if op == opr.GGML_OP_SQR:
        x = _f32(srcs[0]); return x * x
    if op == opr.GGML_OP_SQRT:
        return np.sqrt(np.maximum(_f32(srcs[0]), 0))
    if op == opr.GGML_OP_LOG:
        return np.log(np.maximum(_f32(srcs[0]), 1e-30))
    if op == opr.GGML_OP_SIN:
        return np.sin(_f32(srcs[0]))
    if op == opr.GGML_OP_COS:
        return np.cos(_f32(srcs[0]))
    if op == opr.GGML_OP_LEAKY_RELU:
        slope = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                              dtype=np.float32)[0]
        x = _f32(srcs[0])
        return np.where(x >= 0, x, x * slope)

    # --- parameterised ---
    if op == opr.GGML_OP_SCALE:
        scale = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                              dtype=np.float32)[0]
        bias = np.frombuffer(np.array(params[1], dtype=np.int32).tobytes(),
                             dtype=np.float32)[0]
        return _f32(srcs[0]) * scale + bias
    if op == opr.GGML_OP_ADD1:
        return _f32(srcs[0]) + _f32(srcs[1]).item()
    if op == opr.GGML_OP_CLAMP:
        mn = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                           dtype=np.float32)[0]
        mx = np.frombuffer(np.array(params[1], dtype=np.int32).tobytes(),
                           dtype=np.float32)[0]
        return np.clip(_f32(srcs[0]), mn, mx)
    if op == opr.GGML_OP_FILL:
        val = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                            dtype=np.float32)[0]
        dtype = srcs[0].dtype if srcs else np.float32
        shape = tuple(n for n in node.ne if n > 0) or (1,)
        return np.full(shape[::-1], val, dtype=dtype)

    # --- reductions ---
    if op == opr.GGML_OP_SUM:
        return np.array([_f32(srcs[0]).sum()], dtype=np.float32)
    if op == opr.GGML_OP_SUM_ROWS:
        return _f32(srcs[0]).sum(axis=-1, keepdims=True)
    if op == opr.GGML_OP_MEAN:
        return _f32(srcs[0]).mean(axis=-1, keepdims=True)
    if op == opr.GGML_OP_ARGMAX:
        return np.argmax(_f32(srcs[0]), axis=-1).astype(np.int32)
    if op == opr.GGML_OP_COUNT_EQUAL:
        a = srcs[0].astype(np.int64)
        b = srcs[1].astype(np.int64)
        return np.array([int(np.sum(a == b))], dtype=np.int64)

    # --- matmul ---
    if op == opr.GGML_OP_MUL_MAT:
        # ggml mul_mat: dst[i,j] = sum_k src0[k,i] * src1[k,j]
        # i.e. matmul(src1, src0^T) when our numpy views already reverse axes.
        a = _f32(srcs[0])    # weight-like
        b = _f32(srcs[1])    # activation-like
        # With our reversed numpy convention, a.shape[-1] and b.shape[-1] are
        # both ggml ne[0] (the inner reduction dim).
        return b @ a.swapaxes(-1, -2)
    if op == opr.GGML_OP_OUT_PROD:
        a = _f32(srcs[0])
        b = _f32(srcs[1])
        return np.einsum('...i,...j->...ij', b, a)

    # --- normalisations ---
    if op == opr.GGML_OP_SOFT_MAX:
        x = _f32(srcs[0])
        scale_bits = np.array(params[0], dtype=np.int32).tobytes()
        scale = np.frombuffer(scale_bits, dtype=np.float32)[0] or 1.0
        x = x * scale
        if len(srcs) > 1:    # mask
            x = x + _f32(srcs[1])
        x -= x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)
    if op == opr.GGML_OP_NORM:
        eps = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                            dtype=np.float32)[0]
        x = _f32(srcs[0])
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    if op == opr.GGML_OP_RMS_NORM:
        eps = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                            dtype=np.float32)[0]
        x = _f32(srcs[0])
        rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
        return x / rms
    if op == opr.GGML_OP_L2_NORM:
        eps = np.frombuffer(np.array(params[0], dtype=np.int32).tobytes(),
                            dtype=np.float32)[0]
        x = _f32(srcs[0])
        n = np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)
        return x / n
    if op == opr.GGML_OP_GROUP_NORM:
        # params: [n_groups:i32, eps:f32]
        n_groups = params[0]
        eps = np.frombuffer(np.array(params[1], dtype=np.int32).tobytes(),
                            dtype=np.float32)[0]
        x = _f32(srcs[0])
        # Reshape into (..., n_groups, group_size) for stats over the last 2 dims.
        # ggml groups along the channel dim (ne[2] in 4D); for our reversed numpy
        # view that's axis -3 typically. Practical implementation: treat the whole
        # last dim as channels for now and split.
        shp = x.shape
        ch = shp[-1]
        gs = ch // n_groups
        xg = x.reshape(*shp[:-1], n_groups, gs)
        mean = xg.mean(axis=-1, keepdims=True)
        var = ((xg - mean) ** 2).mean(axis=-1, keepdims=True)
        out = (xg - mean) / np.sqrt(var + eps)
        return out.reshape(shp)

    # --- data movement (view-like ops are no-ops because the dst tensor
    #     metadata already points to a view of the source storage). Only the
    #     materialising ones need to actually copy. ---
    if op in (opr.GGML_OP_DUP, opr.GGML_OP_CPY, opr.GGML_OP_CONT):
        return _f32(srcs[0])    # caller casts to dst type
    if op == opr.GGML_OP_REPEAT:
        # Repeat src0 to fit dst shape. We get dst shape from node.ne.
        dst_shape = tuple(node.ne[d] for d in range(4) if node.ne[d] > 0)[::-1]
        return np.broadcast_to(_f32(srcs[0]), dst_shape).copy()
    if op == opr.GGML_OP_CONCAT:
        axis_ggml = params[0]
        # ggml axis is in ne-space; numpy axis is reversed.
        ndim_a = sum(1 for n in srcs[0].shape)
        axis_np = ndim_a - 1 - axis_ggml
        return np.concatenate([_f32(srcs[0]), _f32(srcs[1])], axis=axis_np)
    if op == opr.GGML_OP_PAD:
        # Up to 4 pad dims in op_params (pad after each ggml dim).
        pads = [(0, params[d]) for d in range(4)][::-1]
        return np.pad(_f32(srcs[0]), pads[:srcs[0].ndim])

    # --- embedding lookup ---
    if op == opr.GGML_OP_GET_ROWS:
        table = _f32(srcs[0])      # (vocab, embd)  in numpy
        ids = srcs[1].astype(np.int64).reshape(-1)
        return table[ids].astype(np.float32)

    # --- masks ---
    if op == opr.GGML_OP_DIAG_MASK_INF:
        n_past = params[0]
        x = _f32(srcs[0])
        n = x.shape[-1]
        m = np.triu(np.ones((n, n), dtype=bool), k=1 + n_past)
        return np.where(m, -np.inf, x)
    if op == opr.GGML_OP_DIAG_MASK_ZERO:
        n_past = params[0]
        x = _f32(srcs[0])
        n = x.shape[-1]
        m = np.triu(np.ones((n, n), dtype=bool), k=1 + n_past)
        return np.where(m, 0, x)

    # --- view-only metadata ops: dst already aliases src buffer; nothing to do.
    if op in (opr.GGML_OP_NONE, opr.GGML_OP_RESHAPE, opr.GGML_OP_VIEW,
              opr.GGML_OP_PERMUTE, opr.GGML_OP_TRANSPOSE):
        return None

    raise NotImplementedError(f"ggml op {op} not implemented in NumPy backend")
