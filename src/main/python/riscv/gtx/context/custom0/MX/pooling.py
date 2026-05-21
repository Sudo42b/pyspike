"""2D pooling ops — pool.m (max) / pool.a (avg).

Params come straight from rs1 / rs2 (not OPERAND SPRs), per the firmware
intrinsics (gtx-firmware intrin_level1: __pool_m / __pool_a):

  rs1 = col_OUT[63:48] | row_OUT[47:32] | col_IN[31:16] | row_IN[15:0]
  rs2 = col_stride[31:24] | row_stride[23:16] | col_K[15:8] | row_K[7:0]
        pool.a also packs k_value above the kernel fields — the avg multiplier
        (1/(row_K*col_K)). Its width tracks config_params.MX_IO_DTYPE:
        FP16 → bits[47:32], FP32 → bits[63:32]; decoded as ``_io_low(rs2 >> 32)``.

Forward only: input row_IN×col_IN at SPM_ADDRA → output row_OUT×col_OUT at
SPM_ADDRR. Element I/O width is MX_IO_DTYPE (config-gated). pool.a computes the
window SUM then multiplies by k_value (the HW supplies the reciprocal, so this
matches it exactly rather than torch's divide-by-window-size average).
"""
import numpy as np
from numpy import ndarray as Tensor

from ...inst_handler import inst_register
from ....config_params import MX_IO_DTYPE, MX_IO_BYTES
from ....csr import LSPR
from ... import _resolve_nest_spu
from . import _io_low


def _pool(npu, proc, inst, *, is_max: bool) -> int:
    nest, spu = _resolve_nest_spu(npu)

    rs1 = int(proc.state.XPR[inst.rs1])
    rs2 = int(proc.state.XPR[inst.rs2])
    row_IN = rs1 & 0xFFFF
    col_IN = (rs1 >> 16) & 0xFFFF
    row_OUT = (rs1 >> 32) & 0xFFFF
    col_OUT = (rs1 >> 48) & 0xFFFF
    row_K = rs2 & 0xFF
    col_K = (rs2 >> 8) & 0xFF
    row_stride = (rs2 >> 16) & 0xFF
    col_stride = (rs2 >> 24) & 0xFF
    if row_K == 0 or col_K == 0 or row_IN == 0 or col_IN == 0:
        return 0   # vendor guards kernel/dims > 0 -> silent NOP.
    k_value = None if is_max else _io_low(rs2 >> 32)   # config-gated width

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    l1_io = npu.mem.l1_io(nest, spu)
    in_off = (addr_a // MX_IO_BYTES) % l1_io.shape[0]
    in_view = l1_io[in_off:in_off + row_IN * col_IN].reshape(row_IN, col_IN)

    result = pool2d(in_view, is_max, (row_K, col_K), (row_stride, col_stride), k_value)

    out_off = (addr_r // MX_IO_BYTES) % l1_io.shape[0]
    n_out = row_OUT * col_OUT
    l1_io[out_off:out_off + n_out] = result.reshape(-1)[:n_out]
    return 0


@inst_register.custom0(name='pool.m', funct7=0b0110000, funct3=0)
def pool_m(npu, proc, inst, cxt) -> int:
    """2D max-pool, forward only."""
    return _pool(npu, proc, inst, is_max=True)


@inst_register.custom0(name='pool.a', funct7=0b0110001, funct3=0)
def pool_a(npu, proc, inst, cxt) -> int:
    """2D avg-pool (sum * k_value), forward only."""
    return _pool(npu, proc, inst, is_max=False)


# =============================================================================
# Pool kernel — FP32 internal, MX_IO_DTYPE output
# =============================================================================
def pool2d(arr: Tensor, is_max: bool, kernel, stride, k_value) -> Tensor:
    from numpy.lib.stride_tricks import sliding_window_view
    x = arr.astype(np.float32)
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    # (OH, OW, kh, kw) windows, strided to the requested step.
    win = sliding_window_view(x, (kh, kw))[::sh, ::sw]
    if is_max:
        out = win.max(axis=(-1, -2))
    else:
        # SUM over the window (matches torch divisor_override=1) then scale by
        # the HW-supplied reciprocal k_value — a multiply, not a divide.
        out = win.sum(axis=(-1, -2)) * float(k_value)
    return out.reshape(-1).astype(MX_IO_DTYPE)
