"""Pooling ops — port of gtx_npu_act.cc:166-220 (exec_pooling).

Forward direction only (ADDRA -> ADDRR). pool.m (funct7=0x30) and pool.a
(funct7=0x31) are distinct funct7, both emitted with funct3=0.
"""
import torch
from torch import Tensor

from ...inst_handler import inst_register
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu


def _pool(npu, proc, inst, *, is_max: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:166-220`` (``exec_pooling``)."""
    nest, spu = _resolve_nest_spu(npu)

    length = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND1'].address, 0)) & 0xFFFF
    if length == 0:
        length = 0x10000
    kernel_size = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0)) & 0xFFFF
    if kernel_size == 0:
        return 0   # vendor guards `kernel_size > 0` -> silent NOP.

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    l1_f16 = npu.mem.l1_f16(nest, spu)
    in_off = (addr_a // 2) % l1_f16.shape[0]
    in_view = l1_f16[in_off:in_off + length]

    result = pool_max(in_view, kernel_size) if is_max else pool_avg(in_view, kernel_size)

    out_off = (addr_r // 2) % l1_f16.shape[0]
    l1_f16[out_off:out_off + length // kernel_size] = result
    return 0


@inst_register.custom0(name='pool.m', funct7=0b0110000, funct3=0)
def pool_m(npu, proc, inst, cxt) -> int:
    """Max-pool, forward only."""
    return _pool(npu, proc, inst, is_max=True)


@inst_register.custom0(name='pool.a', funct7=0b0110001, funct3=0)
def pool_a(npu, proc, inst, cxt) -> int:
    """Avg-pool, forward only."""
    return _pool(npu, proc, inst, is_max=False)


# =============================================================================
# Pool kernels
# =============================================================================
def pool_max(arr_f16: Tensor, kernel_size: int) -> Tensor:
    out = torch.max_pool1d(arr_f16.to(torch.float32),
                           kernel_size=kernel_size, stride=kernel_size)
    return out.to(torch.float16)


def pool_avg(arr_f16: Tensor, kernel_size: int) -> Tensor:
    out = torch.avg_pool1d(arr_f16.to(torch.float32),
                           kernel_size=kernel_size, stride=kernel_size,
                           count_include_pad=False)
    return out.to(torch.float16)
