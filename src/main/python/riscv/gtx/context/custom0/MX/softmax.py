import numpy as np
from numpy import ndarray as Tensor

from ...inst_handler import inst_register
from ....config_params import L0_SIZE_BYTES, MX_IO_DTYPE
from ....csr import GSPR
from ... import _resolve_nest_spu, operand3
from . import _io_low, _io_high, _l0_block_view_io, _write_l0_io_pair

# activation op-id enum (must match act.py's ACT_* values).
ACT_SOFTMAX, ACT_ESUM = 2, 6


def softmax_imm(npu, proc, inst, *, op_id: int) -> int:
    """L0 immediate path for ESUM / SOFTMAX. ``gtx_npu_act.cc:436-487``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[inst.rs1]) & 0x1F
    op3_raw = operand3(npu, 0xFFFFFFFF)
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (inst.rd & 0x1F)

    view_in = _l0_block_view_io(npu, nest, spu, in_reg)

    op2 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    max_val = _io_low(op2)
    accum_val = _io_high(op2)

    if op_id == ACT_ESUM:
        # Result SVR: max_value[low], esum_value[high] (FP32: [31:0]/[63:32]).
        scalar = esum(view_in, max_val=max_val, init_accum=accum_val)
        r_off = ((out_reg & 0x1F) * 32) % L0_SIZE_BYTES
        _write_l0_io_pair(npu, nest, spu, r_off, max_val, scalar)
    elif op_id == ACT_SOFTMAX:
        result = softmax_step3(view_in, max_val, accum_val)
        view_out = _l0_block_view_io(npu, nest, spu, out_reg)
        view_out[:] = result
    return 0


def softmax_step3(arr: Tensor, max_val, esum_val) -> Tensor:
    """Vendor softmax step3: ``r[i] = exp(x[i] - max - ln(esum))`` using the
    pre-computed ``max`` / ``esum`` supplied in operand2 (NOT a fresh full
    softmax). Both forward (L1) and imm (L0) paths share this."""
    f32 = arr.astype(np.float32)
    max_f = np.asarray(max_val, dtype=np.float32)
    esum_f = np.asarray(esum_val, dtype=np.float32)
    ln_esum = (np.log(esum_f) if float(esum_f) > 0.0
               else np.zeros_like(esum_f))
    return np.exp(f32 - max_f - ln_esum).astype(MX_IO_DTYPE)


def esum(arr: Tensor, max_val: float, init_accum: float) -> Tensor:
    arr_f32 = arr.astype(np.float32)
    max_val_f32 = np.asarray(max_val, dtype=np.float32)
    init_accum_f32 = np.asarray(init_accum, dtype=np.float32)
    return (init_accum_f32 + np.sum(np.exp(arr_f32 - max_val_f32))).astype(MX_IO_DTYPE)

@inst_register.custom0(name='esum', funct7=0b0101111, funct3=0b001)
def _esum(npu, proc, inst, cxt) -> int:
    """ESUM: funct7=0x2F funct3=1, FORWARD — writes scalar to L0 (Pitfall 8)."""
    from .act import act
    return act(npu, proc, inst, op_id=ACT_ESUM, is_reversed=False)

@inst_register.custom0(name='softmax', funct7=0b0101111, funct3=0b010)
def _softmax(npu, proc, inst, cxt) -> int:
    from .act import act
    return act(npu, proc, inst,
                         op_id=ACT_SOFTMAX, is_reversed=False)

@inst_register.custom0(name='esum.i', funct7=0b0101111, funct3=0b101)
def _esum_i(npu, proc, inst, cxt) -> int:
    return softmax_imm(npu, proc, inst, op_id=ACT_ESUM)


@inst_register.custom0(name='softmax.i', funct7=0b0101111, funct3=0b110)
def _softmax_i(npu, proc, inst, cxt) -> int:
    return softmax_imm(npu, proc, inst, op_id=ACT_SOFTMAX)