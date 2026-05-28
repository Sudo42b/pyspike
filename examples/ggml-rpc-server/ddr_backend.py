"""DDR ↔ numpy adapter.

Each ggml tensor on the wire carries:
    type, ne[4], nb[4], data (absolute remote address)

`nb` is a byte-stride array — nb[0] = element size, nb[i] = nb[i-1] * ne[i-1]
for contiguous tensors, but may differ for views/permutes. We map this to a
numpy view onto the host bytearray that shadows the client's device buffer.

Note: ggml stores tensors with ne[0] as the fastest-changing dim (column-major
in matmul terms). To make this look natural in numpy we reverse shape/strides
when creating the view.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

from wire_protocol import RpcTensor, GGML_MAX_DIMS


# --- ggml_type enum → numpy dtype (subset) ---
# Source: include/ggml.h enum ggml_type (ordering matters).
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30


_GGML_TO_NP: dict[int, np.dtype] = {
    GGML_TYPE_F32: np.dtype(np.float32),
    GGML_TYPE_F16: np.dtype(np.float16),
    GGML_TYPE_I8:  np.dtype(np.int8),
    GGML_TYPE_I16: np.dtype(np.int16),
    GGML_TYPE_I32: np.dtype(np.int32),
    GGML_TYPE_I64: np.dtype(np.int64),
    GGML_TYPE_F64: np.dtype(np.float64),
}


def ggml_dtype(ggml_type: int) -> np.dtype | None:
    """Map a ggml_type to a numpy dtype, or None if unsupported (e.g. quantized)."""
    return _GGML_TO_NP.get(ggml_type)


def tensor_nbytes(t: RpcTensor) -> int:
    """Total contiguous byte span = max over dims of nb[d] * ne[d].

    For non-contiguous (view/permute) tensors this still bounds the byte
    extent we need to read out of the source buffer.
    """
    span = 0
    for d in range(GGML_MAX_DIMS):
        if t.ne[d] > 0 and t.nb[d] > 0:
            end = t.nb[d] * t.ne[d]
            if end > span:
                span = end
    return span


def view_tensor(t: RpcTensor, host_buf: bytearray, base_off: int) -> np.ndarray:
    """Zero-copy numpy view onto the host buffer for tensor `t`.

    `base_off` = offset of t.data within `host_buf` (caller resolved via
    DeviceMemory.buffer_for_data_addr).

    Shape/strides are reversed so the returned array reads naturally in numpy
    convention (ne[0]=cols becomes shape[-1]).
    """
    dtype = ggml_dtype(t.type)
    if dtype is None:
        raise NotImplementedError(f"ggml_type={t.type} not supported (quantized?)")

    # Determine effective ndim — ggml pads unused dims with ne=1, but we still
    # want to track them to mirror upstream shape semantics.
    ndim = GGML_MAX_DIMS
    while ndim > 1 and t.ne[ndim - 1] == 1:
        ndim -= 1

    shape = tuple(t.ne[d] for d in range(ndim))[::-1]      # reverse to numpy
    strides = tuple(t.nb[d] for d in range(ndim))[::-1]

    # We need a numpy "buffer" view covering the byte span the strided view
    # might touch — that's the maximum reachable byte = sum((ne[d]-1)*nb[d]) + elem_size.
    elem_size = dtype.itemsize
    reach = sum((t.ne[d] - 1) * t.nb[d] for d in range(ndim)) + elem_size

    # Contiguous fast path — direct frombuffer view.
    if _is_contiguous(t, ndim, elem_size):
        n_elems = 1
        for d in range(ndim):
            n_elems *= t.ne[d]
        return np.frombuffer(host_buf, dtype=dtype, count=n_elems,
                             offset=base_off).reshape(shape)

    # Non-contiguous (PERMUTE/TRANSPOSE/VIEW): build a typed strided view, then
    # copy into a contiguous shape-correct array.
    # numpy's as_strided takes BYTE strides regardless of source dtype, so the
    # ggml nb[] values (byte strides) plug in directly when we start from a
    # dtype-typed `raw`. The earlier uint8-based path scaled the byte count
    # incorrectly (copy yields prod(shape) uint8 bytes → view(fp16) halves
    # element count → reshape mismatch on every non-contiguous yolov10+ view).
    raw_typed = np.frombuffer(host_buf, dtype=dtype,
                              count=(reach + elem_size - 1) // elem_size,
                              offset=base_off)
    arr = as_strided(raw_typed, shape=shape, strides=strides)
    return arr.copy()


def _is_contiguous(t: RpcTensor, ndim: int, elem_size: int) -> bool:
    if t.nb[0] != elem_size:
        return False
    for d in range(1, ndim):
        if t.nb[d] != t.nb[d - 1] * t.ne[d - 1]:
            return False
    return True


def write_tensor(t: RpcTensor, host_buf: bytearray, base_off: int,
                 arr: np.ndarray) -> None:
    """Write `arr` (in numpy/row-major shape, dtype matching t.type) into the
    host buffer at the tensor's location.

    Currently only supports contiguous destination tensors — sufficient for
    op outputs in a normal ggml graph (ggml allocates fresh contiguous storage
    for op results, views/permutes don't need a write-back).
    """
    dtype = ggml_dtype(t.type)
    if dtype is None:
        raise NotImplementedError(f"ggml_type={t.type} not supported")

    ndim = GGML_MAX_DIMS
    while ndim > 1 and t.ne[ndim - 1] == 1:
        ndim -= 1

    elem_size = dtype.itemsize
    if not _is_contiguous(t, ndim, elem_size):
        raise NotImplementedError("non-contiguous tensor write not supported")

    expected_shape = tuple(t.ne[d] for d in range(ndim))[::-1]
    if arr.shape != expected_shape:
        # Try to reshape if elements match — covers cases where compute
        # returns e.g. (M,) and we want (1, M).
        if arr.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"write_tensor shape mismatch: arr.shape={arr.shape} "
                f"expected={expected_shape} (ne={t.ne})")
        arr = arr.reshape(expected_shape)

    if arr.dtype != dtype:
        arr = arr.astype(dtype)

    raw = arr.tobytes(order="C")
    end = base_off + len(raw)
    host_buf[base_off:end] = raw
