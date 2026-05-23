"""DMA op handlers — custom0 entry points (port of gtx_npu_dma.cc dispatch).

Thin shim layer: read proc/insn/npu, decode, delegate to the pure functions in
:mod:`dma_imp` (firmware load/store/copy) or the local mcast/copy_mem/SVR
helpers. Loop flags (is_sloop/is_tloop) derive from npu.CONTEXT — C2 (S-loop)
is DDR↔L2 DMA, C3 (T-loop) is L2↔L1 DMA.
"""
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from ...inst_handler import inst_register
from . import dma_imp

from ....csr import GSPR
from ....config_params import (
    NEST_NUM, SPU_NUM, DDR_BASE,
    L0_SIZE_BYTES, L1_SIZE_BYTES, L2_SIZE_BYTES,
)

if TYPE_CHECKING:
    from ....memory import GtxMemory
    from ....npu import GtxNpu

_OPERAND3_ADDR = GSPR['GSPR_GTX_OPERAND3'].address & 0x3FF   # 0x003

# ============================================================================
# Helpers
# ============================================================================
def _select_nest(npu) -> int:
    nest = npu.warp.current_nest if npu.warp.is_ploop else 0
    assert nest < NEST_NUM, f"NEST id {nest} >= NEST_NUM={NEST_NUM}"
    return nest


def _select_spu(npu) -> int:
    spu = npu.warp.current_spu
    assert spu < SPU_NUM, f"SPU id {spu} >= SPU_NUM={SPU_NUM}"
    return spu


def _xflags(inst) -> tuple:
    """(xd, xs1, xs2) reconstructed from the RoCC fn3 bits."""
    f3 = inst.fn3
    return (f3 >> 2) & 1, (f3 >> 1) & 1, f3 & 1


def _operand3(npu) -> int:
    # Direct-tensor SPR read — mirrors opset's direct-tensor write
    # (DL/spr.py). Skips RegisterFile.get()/__getitem__ on the hot
    # (opset, load, abs.v, opset, store) cadence.
    return int(npu.gspr.tensor[_OPERAND3_ADDR])

# ============================================================================
# Multicast + copy.mem (funct7=0x42 / 0x44) — bodies are vendor-parity ports.
# ============================================================================
def mcast_s2l(mem: 'GtxMemory', *, nest: int, l2_addr: int, l1_addr: int,
                       height: int, length: int, rd_stride: int,
                       target_spu_mask: int) -> int:
    """L2 → L1 multicast to selected SPUs (gtx_npu_custom0.cc:230-273)."""
    if height == 0:
        height = 1
    if length == 0:
        length = 0x10000
    effective_stride = rd_stride if rd_stride > 0 else length
    assert effective_stride >= length, "rd_stride < length (firmware bug)"
    l2 = mem.l2_byte(nest)
    src_end = l2_addr + (height - 1) * effective_stride + length
    assert src_end <= L2_SIZE_BYTES, "L2 src wraps — firmware bug"
    src_2d = l2[l2_addr:l2_addr + height * effective_stride] \
        .reshape(height, effective_stride)[:, :length]
    l1_end = l1_addr + height * length
    assert l1_end <= L1_SIZE_BYTES, "L1 dst wraps — firmware bug"
    for s in range(SPU_NUM):
        if (target_spu_mask >> s) & 1:
            mem.l1_byte(nest, s)[l1_addr:l1_end].reshape(height, length)[...] = src_2d
    return 0


def mcast_g2s(mem: 'GtxMemory', *, ddr_addr: int, l2_addr: int,
                       height: int, length: int, rd_stride: int,
                       target_nest_mask: int) -> int:
    """DDR → L2 multicast to selected NESTs (gtx_npu_custom0.cc:545-583)."""
    if height == 0:
        height = 1
    if length == 0:
        length = 0x10000
    effective_stride = rd_stride if rd_stride > 0 else length
    assert effective_stride >= length, "rd_stride < length (firmware bug)"
    ddr_off_base = (ddr_addr - DDR_BASE) if ddr_addr >= DDR_BASE else ddr_addr
    max_off = ddr_off_base + (height - 1) * effective_stride + length
    mem.ensure_ddr(max_off)
    ddr_span = mem.ddr.read(ddr_off_base, max_off - ddr_off_base)
    src_2d_cpu = ddr_span[:height * effective_stride].reshape(height, effective_stride)[:, :length]
    l2_end = l2_addr + height * length
    assert l2_end <= L2_SIZE_BYTES, "L2 dst wraps — firmware bug"
    for k in range(NEST_NUM):
        if (target_nest_mask >> k) & 1:
            l2 = mem.l2_byte(k)
            l2[l2_addr:l2_end].reshape(height, length)[...] = src_2d_cpu
    return 0


def mcast_s2s(mem: 'GtxMemory', *, src_tmu: int, src_addr: int, dst_addr: int,
                       src_stride: int, dst_stride: int, length: int, height: int,
                       target_nest_mask: int) -> int:
    """L2 → L2 multicast across NESTs (gtx_npu_dispatch.cc:732-762)."""
    if height == 0:
        height = 1
    assert height > 0 and length > 0, f"height={height} length={length}"
    if src_tmu >= NEST_NUM:
        src_tmu = 0
    src_l2 = mem.l2_byte(src_tmu)
    for row in range(height):
        s_off = (src_addr + row * src_stride) % L2_SIZE_BYTES
        d_off = (dst_addr + row * dst_stride) % L2_SIZE_BYTES
        copy_len = min(length, L2_SIZE_BYTES - max(s_off, d_off))
        if copy_len <= 0:
            continue
        tmp = src_l2[s_off:s_off + copy_len].copy()
        for k in range(NEST_NUM):
            if (target_nest_mask >> k) & 1:
                dst_l2 = mem.l2_byte(k)
                dst_l2[d_off:d_off + copy_len][...] = tmp
    return 0


def copy_mem(npu: Any, *, nest_id: int, src_addr_raw: int, dst_addr_raw: int,
                      src_stride: int, dst_stride: int, length: int, height: int) -> int:
    """DDR↔DDR / L2↔DDR / L2↔L2 copy (gtx_npu_dispatch.cc:763-846)."""
    if height == 0:
        height = 1
    assert height > 0 and length > 0, f"height={height} length={length}"
    mem = npu.mem
    src_is_ddr = src_addr_raw >= L2_SIZE_BYTES
    dst_is_ddr = dst_addr_raw >= L2_SIZE_BYTES

    if src_is_ddr or dst_is_ddr:
        npu.flush_deferred_ddr_stores()   # mandatory before DDR touch (vendor :784)
        src_off = (src_addr_raw - DDR_BASE) if src_addr_raw >= DDR_BASE else src_addr_raw
        dst_off = (dst_addr_raw - DDR_BASE) if dst_addr_raw >= DDR_BASE else dst_addr_raw
        if src_is_ddr and dst_is_ddr:
            max_src = src_off + (height - 1) * src_stride + length
            max_dst = dst_off + (height - 1) * dst_stride + length
            mem.ensure_ddr(max(max_src, max_dst))
            for row in range(height):
                s_base = src_off + row * src_stride
                d_base = dst_off + row * dst_stride
                copy_len = length
                cap = mem.ddr.capacity()
                if s_base + copy_len > cap:
                    copy_len = max(0, cap - s_base)
                if d_base + copy_len > cap:
                    copy_len = max(0, cap - d_base)
                if copy_len > 0:
                    mem.ddr.write(d_base, mem.ddr.read(s_base, copy_len))
        elif src_is_ddr:
            n = nest_id if nest_id < NEST_NUM else 0
            l2 = mem.l2_byte(n)
            mem.ensure_ddr(src_off + (height - 1) * src_stride + length)
            for row in range(height):
                s = src_off + row * src_stride
                d = (dst_addr_raw + row * dst_stride) % L2_SIZE_BYTES
                copy_len = length
                cap = mem.ddr.capacity()
                if s + copy_len > cap:
                    copy_len = max(0, cap - s)
                if d + copy_len > L2_SIZE_BYTES:
                    copy_len = max(0, L2_SIZE_BYTES - d)
                if copy_len > 0:
                    l2[d:d + copy_len][...] = mem.ddr.read(s, copy_len)
        else:
            n = nest_id if nest_id < NEST_NUM else 0
            l2 = mem.l2_byte(n)
            mem.ensure_ddr(dst_off + (height - 1) * dst_stride + length)
            for row in range(height):
                s = (src_addr_raw + row * src_stride) % L2_SIZE_BYTES
                d = dst_off + row * dst_stride
                copy_len = length
                cap = mem.ddr.capacity()
                if s + copy_len > L2_SIZE_BYTES:
                    copy_len = max(0, L2_SIZE_BYTES - s)
                if d + copy_len > cap:
                    copy_len = max(0, cap - d)
                if copy_len > 0:
                    mem.ddr.write(d, l2[s:s + copy_len].cpu())
    else:
        n = nest_id if nest_id < NEST_NUM else 0
        l2 = mem.l2_byte(n)
        for row in range(height):
            s_off = (src_addr_raw + row * src_stride) % L2_SIZE_BYTES
            d_off = (dst_addr_raw + row * dst_stride) % L2_SIZE_BYTES
            copy_len = min(length, L2_SIZE_BYTES - max(s_off, d_off))
            if copy_len <= 0:
                continue
            tmp = l2[s_off:s_off + copy_len].copy()
            l2[d_off:d_off + copy_len][...] = tmp
    return 0

# ============================================================================
# SVR L1↔L0 transfers (funct7=0x41, funct3=0/1)
# ============================================================================
def exec_load_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                  l1_addr: int, l0_reg: int) -> None:
    """L1 → L0 (32 B = one SVR register)."""
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32
    assert l1_off + 32 <= L1_SIZE_BYTES, "L1 SVR window wraps — firmware bug"
    l0_buf[l0_off:l0_off + 32] = l1_buf[l1_off:l1_off + 32]


def exec_store_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                   l1_addr: int, l0_reg: int) -> None:
    """L0 → L1 (32 B = one SVR register)."""
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32
    assert l1_off + 32 <= L1_SIZE_BYTES, "L1 SVR window wraps — firmware bug"
    l1_buf[l1_off:l1_off + 32] = l0_buf[l0_off:l0_off + 32]

@inst_register.custom0(name='load', funct7=0b1000000, funct3=0)
def _load(npu, proc, inst, cxt) -> int:
    """dma LOAD. C2 (S-loop) DDR→L2; C3 (T-loop) L2→L1."""
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    xd, xs1, xs2 = _xflags(inst)
    args = dma_imp.decode_dma_args(rs1, rs2, _operand3(npu), xd=xd, xs1=xs1, xs2=xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        # A deferred store reads L2 lazily; flush any whose source this load is
        # about to overwrite (CONCAT reuses one L2 row across load/store pairs).
        wr = args['wr_stride'] or args['length']
        l2_lo = args['addr_lo']
        l2_hi = l2_lo + (args['height'] - 1) * wr + args['length']
        npu.flush_deferred_if_l2_overlap(nest, l2_lo, l2_hi)
        return dma_imp.dma_sloop_load(
            npu.mem, nest=nest, addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        return dma_imp.dma_tloop_load_store(
            npu.mem, nest=nest, spu=_select_spu(npu), is_store=False,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@inst_register.custom0(name='store', funct7=0b1000000, funct3=1)
def _store(npu, proc, inst, cxt) -> int:
    """dma STORE. S-loop store is deferred (pushes onto npu queue)."""
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    xd, xs1, xs2 = _xflags(inst)
    args = dma_imp.decode_dma_args(rs1, rs2, _operand3(npu), xd=xd, xs1=xs1, xs2=xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        return dma_imp.dma_sloop_store(
            npu, nest=nest, addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        if os.environ.get("GTX_DEBUG_DMA"):
            import sys as _sys
            _spu = _select_spu(npu)
            _l1 = npu.mem.l1_byte(nest, _spu)
            _src = _l1[args['addr_lo']:args['addr_lo'] + args['length']]
            print(f"[DBG store] spu={_spu} hi={args['addr_hi']:#x} lo={args['addr_lo']:#x} "
                  f"len={args['length']} src={bytes(_src.tolist()).hex()}", file=_sys.stderr)
        return dma_imp.dma_tloop_load_store(
            npu.mem, nest=nest, spu=_select_spu(npu), is_store=True,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@inst_register.custom0(name='copy', funct7=0b1000000, funct3=2)
def _copy(npu, proc, inst, cxt) -> int:
    """dma COPY (T-loop L1→L1). addr_hi=dst, addr_lo=src."""
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    xd, xs1, xs2 = _xflags(inst)
    args = dma_imp.decode_dma_args(rs1, rs2, _operand3(npu), xd=xd, xs1=xs1, xs2=xs2)
    nest = _select_nest(npu)
    if os.environ.get("GTX_DEBUG_DMA"):
        import sys as _sys
        _sb = npu.mem.l1_byte(nest, _select_spu(npu))[args['addr_lo']:args['addr_lo']+args['length']]
        print(f"[DBG copy] tloop={npu.warp.is_tloop} src={args['addr_lo']:#x} "
              f"dst={args['addr_hi']:#x} len={args['length']} h={args['height']} "
              f"src_bytes={bytes(_sb.tolist()).hex()}", file=_sys.stderr)
    if npu.warp.is_tloop:
        return dma_imp.dma_tloop_copy(
            npu.mem, nest=nest, spu=_select_spu(npu),
            src_addr=args['addr_lo'], dst_addr=args['addr_hi'],
            length=args['length'], height=args['height'])
    # Shared (NEST) context → L2 → L2 strided 2D copy (REPEAT/CONCAT tiling).
    return dma_imp.dma_sloop_copy(
        npu.mem, nest=nest,
        src_addr=args['addr_lo'], dst_addr=args['addr_hi'],
        length=args['length'], height=args['height'],
        rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])


@inst_register.custom0(name='load.svr', funct7=0b1000001, funct3=0)
def _load_svr(npu, proc, inst, cxt) -> int:
    l1_addr = proc.state.XPR[inst.rs1] & 0x7FFFFFF
    l0_reg = proc.state.XPR[inst.rs2] & 0x1F
    exec_load_svr(npu.mem, nest_id=_select_nest(npu), spu_id=_select_spu(npu),
                  l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


@inst_register.custom0(name='store.svr', funct7=0b1000001, funct3=1)
def _store_svr(npu, proc, inst, cxt) -> int:
    l1_addr = proc.state.XPR[inst.rs1] & 0x7FFFFFF
    l0_reg = proc.state.XPR[inst.rs2] & 0x1F
    exec_store_svr(npu.mem, nest_id=_select_nest(npu), spu_id=_select_spu(npu),
                   l1_addr=l1_addr, l0_reg=l0_reg)
    return 0



@inst_register.custom0(name='mcast.s2l', funct7=0b1000010, funct3=0)
def _mcast_s2l(npu, proc, inst, cxt) -> int:
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    rs3 = _operand3(npu)
    return mcast_s2l(
        npu.mem, nest=_select_nest(npu),
        l2_addr=(rs1 >> 32) & 0xFFFFFFFF, l1_addr=rs1 & 0xFFFFFFFF,
        height=(rs2 >> 48) & 0xFFFF, length=(rs2 >> 32) & 0xFFFF,
        rd_stride=rs2 & 0xFFFFFFFF, target_spu_mask=rs3 & 0xFFFF)


@inst_register.custom0(name='mcast.g2s', funct7=0b1000100, funct3=0)
def _mcast_g2s(npu, proc, inst, cxt) -> int:
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    rs3 = _operand3(npu)
    return mcast_g2s(
        npu.mem,
        ddr_addr=(rs1 >> 27) & 0x1FFFFFFFFF, l2_addr=rs1 & 0x7FFFFFF,
        height=(rs2 >> 48) & 0xFFFF, length=(rs2 >> 32) & 0xFFFF,
        rd_stride=rs2 & 0xFFFFFFFF, target_nest_mask=rs3 & 0xFFFF)


@inst_register.custom0(name='mcast.s2s', funct7=0b1000100, funct3=2)
def _mcast_s2s(npu, proc, inst, cxt) -> int:
    op1 = proc.state.XPR[inst.rs1]
    op2 = proc.state.XPR[inst.rs2]
    op3 = _operand3(npu)
    return mcast_s2s(
        npu.mem,
        src_tmu=(op1 >> 56) & 0x3F, src_addr=op1 & 0x7FFFFFF, dst_addr=(op1 >> 27) & 0x7FFFFFF,
        src_stride=op2 & 0xFFFFFFFF, dst_stride=op3 & 0xFFFFFFFF,
        length=(op2 >> 32) & 0xFFFF, height=(op2 >> 48) & 0xFFFF,
        target_nest_mask=(op3 >> 32) & 0xFFFFFFFF)


@inst_register.custom0(name='copy.mem', funct7=0b1000100, funct3=3)
def _copy_mem(npu, proc, inst, cxt) -> int:
    op1 = proc.state.XPR[inst.rs1]
    op2 = proc.state.XPR[inst.rs2]
    op3 = _operand3(npu)
    return copy_mem(
        npu, nest_id=_select_nest(npu),
        src_addr_raw=op1 & 0x1FFFFFFFFF, dst_addr_raw=op3 & 0x1FFFFFFFFF,
        src_stride=op2 & 0xFFFFFFFF,
        dst_stride=((op1 >> 48) & 0xFFFF) | (((op3 >> 48) & 0xFFFF) << 16),
        length=(op2 >> 32) & 0xFFFF, height=(op2 >> 48) & 0xFFFF)
