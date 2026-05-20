"""im2col handlers — port of gtx_npu_custom0.cc:696-805 (firmware IM2COL).

ISA encoding (RoCC custom0 0x0b). funct7 = family, funct3 = 0:

  im2col.n  4'b0001  3'b000  gpr gpr  ...  im2col for normal convolution
  im2col.d  4'b0001  3'b001  gpr gpr  ...  im2col for depth-wise convolution

Operand layout (firmware __im2col_n / __im2col_d — packed in rs1/rs2):
  rs1[15:0]  = row_A_size (input height)
  rs1[31:16] = col_A_size (input width)
  rs2[4:0]   = kernel_size
  rs2[9:8]   = dilate (1=none, 2=filt5, 3=filt7)
  rs2[31:16] = stride
  rs2[47:32] = #channel

HW convention (vendor): read input from ADDR_R (src), write patches to ADDR_A
(dst). im2col only copies FP16 halfwords (no arithmetic), so this port moves
2-byte halfword pairs through the raw byte view — endianness-agnostic and
byte-for-byte faithful to the vendor's rd16_raw/wr16_raw halfword round-trip.

NOTE: im2col.n and im2col.d carry DISTINCT funct7 (0x08/0x09); funct3 is left
at 0 (the registry keys on (funct7, funct3)).

Vendor loop (gtx_npu_custom0.cc do_im2col):
  IM2COL_N: output [out_h*out_w, nch*ksz²]   loop (j,k)→(i)→(l,m)
  IM2COL_D: output [nch*out_h*out_w, ksz²]   loop (i)→(j,k)→(l,m)
  src = r_addr + ((row_A*col_A*i)+(col_A*j+k)+(dil*l*col_A+dil*m))*2
  spu.wr16_raw(a_addr + t, spu.rd16_raw(src)); t += 2
"""
from __future__ import annotations

import torch

from ...inst_handler import inst_register

from ....csr import GSPR, LSPR  # noqa: F401  (LSPR keys read via npu.lspr below)
from ... import _resolve_nest_spu


def _im2col(npu, proc, inst, *, is_depthwise: bool) -> int:
    """Direct port of ``gtx_npu_custom0.cc:706-805`` firmware IM2COL path."""
    # Pitfall F: rs1 index 0 ⇒ firmware WRSPR/RDSPR alias, not im2col ⇒ NOP.
    if inst.rs1 == 0:
        return 0

    rs1_val = int(proc.state.XPR[inst.rs1])
    rs2_val = int(proc.state.XPR[inst.rs2])

    row_A = rs1_val & 0xFFFF
    col_A = (rs1_val >> 16) & 0xFFFF
    ksz = rs2_val & 0x1F
    dil = (rs2_val >> 8) & 0x3
    stride = (rs2_val >> 16) & 0xFFFF
    nch = (rs2_val >> 32) & 0xFFFF

    # Validate dilation (vendor: 0 → skip).
    if dil == 0:
        return 0

    # Effective filter size with dilation (vendor switch).
    if dil == 1:
        filt_sz = ksz
    elif dil == 2:
        filt_sz = 5
        if ksz != 3:
            ksz = 3
    elif dil == 3:
        filt_sz = 7
        if ksz != 3:
            ksz = 3
    else:
        return 0  # dilation too big → skip

    if stride == 0:
        stride = 1
    if nch == 0:
        nch = 1

    out_h = ((row_A - filt_sz) // stride + 1) if row_A >= filt_sz else 0
    out_w = ((col_A - filt_sz) // stride + 1) if col_A >= filt_sz else 0
    if out_h <= 0 or out_w <= 0:
        return 0

    nest, spu = _resolve_nest_spu(npu)
    l1 = npu.mem.l1_byte(nest, spu)            # raw byte view (endian-agnostic)
    l1_len = l1.shape[0]

    addr_a = int(npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0))  # dst
    addr_r = int(npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0))  # src

    # Build the gather list of source halfword offsets in vendor loop order,
    # then copy the 2-byte halfwords to consecutive dst offsets (t += 2).
    src_hw: list[int] = []
    if not is_depthwise:
        # IM2COL_N: output [out_h*out_w, nch * ksz²]
        # Loop: (j,k) spatial → (i) channel → (l,m) kernel
        j = 0
        while (j + filt_sz) <= row_A:
            k = 0
            while (k + filt_sz) <= col_A:
                for i in range(nch):
                    for l in range(ksz):
                        for m in range(ksz):
                            src_hw.append(
                                (row_A * col_A * i)
                                + (col_A * j + k)
                                + (dil * l * col_A + dil * m)
                            )
                k += stride
            j += stride
    else:
        # IM2COL_D: output [nch * out_h*out_w, ksz²]
        # Loop: (i) channel → (j,k) spatial → (l,m) kernel
        for i in range(nch):
            j = 0
            while (j + filt_sz) <= row_A:
                k = 0
                while (k + filt_sz) <= col_A:
                    for l in range(ksz):
                        for m in range(ksz):
                            src_hw.append(
                                (row_A * col_A * i)
                                + (col_A * j + k)
                                + (dil * l * col_A + dil * m)
                            )
                    k += stride
                j += stride

    if not src_hw:
        return 0

    # src byte offset = addr_r + hw*2 ; dst byte offset = addr_a + t (t += 2).
    src_idx = torch.tensor(src_hw, dtype=torch.int64)
    src_lo = (addr_r + src_idx * 2) % l1_len
    src_hi = (src_lo + 1) % l1_len

    n = src_idx.numel()
    dst_lo = (addr_a + torch.arange(n, dtype=torch.int64) * 2) % l1_len
    dst_hi = (dst_lo + 1) % l1_len

    # Snapshot source bytes first (dst may overlap src for in-place layouts).
    lo_vals = l1[src_lo].clone()
    hi_vals = l1[src_hi].clone()
    l1[dst_lo] = lo_vals
    l1[dst_hi] = hi_vals
    return 0


@inst_register.custom0(name='im2col.n', funct7=0b0001000, funct3=0)
def _im2col_n(npu, proc, inst, cxt) -> int:
    """im2col for normal convolution."""
    return _im2col(npu, proc, inst, is_depthwise=False)


@inst_register.custom0(name='im2col.d', funct7=0b0001001, funct3=0)
def _im2col_d(npu, proc, inst, cxt) -> int:
    """im2col for depth-wise convolution."""
    return _im2col(npu, proc, inst, is_depthwise=True)
