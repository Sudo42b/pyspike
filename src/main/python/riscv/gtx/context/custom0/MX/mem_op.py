from __future__ import annotations

import numpy as np

from ...inst_handler import inst_register
from ... import operand3
from ....config_params import DDR_BASE, NEST_NUM, SPU_NUM
from ....csr import GSPR, LSPR
from ....memory import GtxMemory
from ..DL.dma import _select_nest, _select_spu

# ============================================================================
# transpose -- direct port of gtx_npu_dma.cc:143-167
# ============================================================================
def transpose(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                    rows: int, cols: int, addr_a: int, addr_r: int) -> int:
    """In-place L1 matrix transpose (FP16, 2 bytes per elem).

    Invariants (asserted): src and dst FP16 windows both fit within L1
    without wrap. ``.copy()`` clones the transposed view, so
    ``src == dst`` (in-place transpose) is safe. ``rows == 1`` or
    ``cols == 1`` is a degenerate transpose that still costs one
    contiguous copy — no special case needed.
    """
    assert nest_id < NEST_NUM, f"nest_id {nest_id} >= NEST_NUM {NEST_NUM}"
    assert spu_id < SPU_NUM, f"spu_id {spu_id} >= SPU_NUM {SPU_NUM}"
    assert rows > 0 and cols > 0, f"rows {rows} or cols {cols} is 0"

    l1_f16 = mem.l1_byte(nest_id, spu_id).view(np.float16)
    nelem_total = l1_f16.shape[0]
    nelem = rows * cols
    a_h = (addr_a // 2) % nelem_total
    r_h = (addr_r // 2) % nelem_total

    assert a_h + nelem <= nelem_total, (
        f"src window [{a_h}, {a_h + nelem}) wraps L1 fp16 capacity "
        f"{nelem_total} — firmware bug"
    )
    assert r_h + nelem <= nelem_total, (
        f"dst window [{r_h}, {r_h + nelem}) wraps L1 fp16 capacity "
        f"{nelem_total} — firmware bug"
    )

    src_view = l1_f16[a_h:a_h + nelem].reshape(rows, cols)
    l1_f16[r_h:r_h + nelem] = src_view.T.copy().reshape(-1)
    return 0


# ============================================================================
# transpose_ddr -- direct port of gtx_npu_dma.cc:175-225
# ============================================================================
def transpose_ddr(mem: 'GtxMemory', *, src_addr: int, dst_addr: int,
                        dim2: int, dim1: int, dim0: int,
                        p2: int, p1: int, p0: int) -> None:
    """DDR-to-DDR 3D tensor transpose/permute (FP16).

    Reads ``[dim2][dim1][dim0]`` from ``src_addr``, writes the permuted
    layout to ``dst_addr``. ``(p2, p1, p0)`` selects which old axis
    drives each new axis.

    Axis mapping: src axis k holds ``dim_(2-k)`` (axis 0 = dim2, axis
    1 = dim1, axis 2 = dim0). The output shape is
    ``(old_dims[p2], old_dims[p1], old_dims[p0])`` → ``torch.permute(
    2 - p2, 2 - p1, 2 - p0)``. ``.copy()`` flattens row-major,
    matching the vendor ``dst_idx = oi[p2]*new_s2 + oi[p1]*new_s1 +
    oi[p0]``.
    """
    src_off = (src_addr - DDR_BASE) if src_addr >= DDR_BASE else src_addr
    dst_off = (dst_addr - DDR_BASE) if dst_addr >= DDR_BASE else dst_addr

    assert dim2 > 0 and dim1 > 0 and dim0 > 0, (
        f"dims must be positive: dim2={dim2} dim1={dim1} dim0={dim0}"
    )

    nelem = dim2 * dim1 * dim0
    max_off = max(src_off + nelem * 2, dst_off + nelem * 2)
    mem.ensure_ddr(max_off)
    cap = mem.ddr.capacity()

    assert src_off + nelem * 2 <= cap, (
        f"src region [{src_off}, {src_off + nelem * 2}) exceeds DDR "
        f"capacity {cap} — firmware bug"
    )
    assert dst_off + nelem * 2 <= cap, (
        f"dst region [{dst_off}, {dst_off + nelem * 2}) exceeds DDR "
        f"capacity {cap} — firmware bug"
    )

    src_span = mem.ddr.read(src_off, nelem * 2)
    src_3d = src_span.view(np.float16).reshape(dim2, dim1, dim0)
    permuted = src_3d.transpose(2 - p2, 2 - p1, 2 - p0).copy()
    mem.ddr.write(dst_off, permuted.view(np.uint8).reshape(-1))

# ============================================================================
# tpose / fill (funct7=0x38 / 0x39)
# ============================================================================
@inst_register.custom0(name='tpose', funct7=0b00111000, funct3=0)
def _tpose(npu, proc, inst, cxt) -> int:
    # tpose	4'b0111	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	smu	2	N/A	src_addr[36:0], rw_dir[49:48], dtype[56]	dim0[1:0], dim1[5:4], dim2[9:8], dim0_size[31:16], dim1_size[47:32], dim2_size[63:48]	dst_addr[36:0]	N/A	N/A	N/A	transpose (maximum 3D)
    state = proc.state
    rs1 = state.XPR[inst.rs1]
    rs2 = state.XPR[inst.rs2]
    rows = rs1 & 0xFFFF
    cols = rs2 & 0xFFFF
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0) & 0xFFFFFFFF
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0) & 0xFFFFFFFF
    return transpose(
        npu.mem, nest_id=nest, spu_id=spu, rows=rows, cols=cols,
        addr_a=addr_a, addr_r=addr_r)



@inst_register.custom0(name='fill', funct7=0b00111001, funct3=0)
def _fill(npu, proc, inst, cxt) -> int:
    """Fill ``height`` rows of ``length`` BYTES with the 64-bit ``fill_pattern``.

    Port of SystemC ``TMU::fill`` (vendor/simulator/src/TMU.cpp:420):
      rs1 = dir[48] | dst_addr[36:0]
      rs2 = height[63:48] | length[47:32] (BYTES) | write_stride[31:0]
      rs3 = fill_pattern[63:0]  ← staged via ``__opset(0, rs3)`` → OPERAND3
      dir 0 → L2 (per NEST), dir 1 → DDR. The 64-bit pattern tiles every 8 B.
    """
    rs1 = int(proc.state.XPR[inst.rs1])
    rs2 = int(proc.state.XPR[inst.rs2])
    dst_addr = rs1 & 0x1FFFFFFFFF
    rw_dir = (rs1 >> 48) & 0x1
    write_stride = rs2 & 0xFFFFFFFF
    length = (rs2 >> 32) & 0xFFFF                       # BYTES per row
    height = (rs2 >> 48) & 0xFFFF
    pattern = operand3(npu) & 0xFFFFFFFFFFFFFFFF        # fill_pattern (rs3)
    if length == 0 or height == 0:
        return 0

    # Tile the 8-byte LE pattern across one row's worth of bytes.
    pat8 = np.frombuffer(int(pattern).to_bytes(8, 'little'), dtype=np.uint8)
    full, rem = divmod(length, 8)
    row = np.empty(length, dtype=np.uint8)
    if full:
        row[:full * 8] = np.tile(pat8, full)
    if rem:
        row[full * 8:] = pat8[:rem]

    if rw_dir == 0:                                     # → L2 (per NEST)
        nest_id = _select_nest(npu)
        assert nest_id < NEST_NUM, f"invalid nest_id {nest_id}"
        l2 = npu.mem.l2_byte(nest_id)
        cap = l2.shape[0]
        for i in range(height):
            off = (dst_addr + i * write_stride) % cap
            l2[off:off + length] = row
    else:                                               # → DDR
        ddr = npu.mem.ddr
        base = npu.mem._ddr_offset(dst_addr)
        ddr.ensure(base + (height - 1) * write_stride + length)
        for i in range(height):
            ddr.write(base + i * write_stride, row)
    return 0
