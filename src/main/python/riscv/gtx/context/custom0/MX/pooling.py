"""
pool.m	4'b0110	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	row_IN[15:0], col_IN[31:16], row_OUT[47:32], col_OUT[63:48]	row_K[7:0], col_K[15:8], row_stride[23:16], col_stride[31:24]	N/A	N/A	N/A	N/A	fp16 max pooling
pool.a	4'b0110	3'b001	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	row_IN[15:0], col_IN[31:16], row_OUT[47:32], col_OUT[63:48]	row_K[7:0], col_K[15:8], row_stride[23:16], col_stride[31:24], K_value[47:32]	N/A	N/A	N/A	N/A	fp16 average pooling
"""


# =========================================================================
# Phase 5: POOL funct7 constants
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:160-161
# =========================================================================
F7_POOL_MAX: int = 0x30         # pool_m
F7_POOL_AVG: int = 0x31         # pool_a
import torch
from torch import Tensor

def _pool(npu, proc, insn, *, is_max: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:166-220`` (``exec_pooling``).

    Forward direction only (ADDRA -> ADDRR per CONTEXT D-08).
    """
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

# ----- Pool @handlers -------------------------------------------------------

@handler(kind='custom0', funct7=F7_POOL_MAX, mnemonic='pool_m')
def _pool_m(npu, proc, insn, xs1, xs2):
    """Max-pool, forward only."""
    return _pool(npu, proc, insn, is_max=True)


@handler(kind='custom0', funct7=F7_POOL_AVG, mnemonic='pool_a')
def _pool_a(npu, proc, insn, xs1, xs2):
    """Avg-pool with -0.0 -> +0.0 canonicalization."""
    return _pool(npu, proc, insn, is_max=False)

# =============================================================================
# 4. Pool kernels
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

