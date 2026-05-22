"""Credit counters + checks — port of gtx_npu_custom0.cc:646-694.

Per-(NEST, SPU) load/store credit counters that gate the DMA<->compute
hand-off. In pyspike's eager functional model the *_chk spins always pass
(DMA is instantaneous), so the counters are tracked for vendor 1:1 parity
but never stall control flow.

The load-bearing behavior is ``credit.st.chk``'s deferred-DDR-store flush:
plan-style (WSPLIT) firmware suppresses the ``end.p`` flush
(``control.py:endp`` checks ``wsplit_seen``) and relies on ``credit.st.chk``
to commit the S-loop deferred queue mid-execution. Multi-tile firmware needs
this per-tile flush — the queue snapshots L2 at flush time, so deferring all
tiles to the atexit flush would make every entry read the final tile's L2.
"""
import sys

import numpy as np

from ...inst_handler import inst_register
from ....config_params import NEST_NUM, SPU_NUM


def _dec_one(row, kind: str, nest: int, spu: int) -> None:
    """Strict single-SPU credit decrement: a decrement when already 0 is a
    firmware protocol violation — report it, clamp at 0 (never go negative)."""
    if row[spu] == 0:
        print(f"[GTX_CREDIT_ERROR] credit.{kind} decrement when already 0 "
              f"(plz check firmware) - nest{nest} spu{spu}",
              file=sys.stderr, flush=True)
    else:
        row[spu] -= 1


def _dec_all(row, kind: str, nest: int) -> None:
    """Strict all-SPU credit decrement — vectorized (no per-SPU Python loop).
    Reports underflow only on violation; otherwise one ``np.maximum`` clamp."""
    zero = (row == 0)
    if zero.any():
        for spu in np.nonzero(zero)[0]:
            print(f"[GTX_CREDIT_ERROR] credit.{kind} decrement when already 0 "
                  f"(plz check firmware) - nest{nest} spu{int(spu)}",
                  file=sys.stderr, flush=True)
    np.subtract(row, 1, out=row)
    np.maximum(row, 0, out=row)


@inst_register.custom0(name='credit.ld', funct7=0b1010000, funct3=0)
def credit_ld(npu, proc, inst, cxt) -> int:
    """credit.ld (0x50): S-loop DMA-load done → inc all SPUs; T-loop consume → dec."""
    warp = npu.warp
    nest = warp.current_nest if warp.is_ploop else 0
    if nest < NEST_NUM:
        if warp.is_sloop:
            npu._credit_ld[nest, :] += 1
        elif warp.is_tloop and warp.current_spu < SPU_NUM:
            _dec_one(npu._credit_ld[nest], "ld", nest, warp.current_spu)
    return 0


@inst_register.custom0(name='credit.st', funct7=0b1010001, funct3=0)
def credit_st(npu, proc, inst, cxt) -> int:
    """credit.st (0x51): T-loop compute done → inc; S-loop DMA-store consume → dec all."""
    warp = npu.warp
    nest = warp.current_nest if warp.is_ploop else 0
    if nest < NEST_NUM:
        if warp.is_tloop and warp.current_spu < SPU_NUM:
            npu._credit_st[nest, warp.current_spu] += 1
        elif warp.is_sloop:
            _dec_all(npu._credit_st[nest], "st", nest)
    return 0


@inst_register.custom0(name='credit.ld.chk', funct7=0b1010010, funct3=0)
def credit_ld_chk(npu, proc, inst, cxt) -> int:
    """credit.ld.chk (0x52): NOP — spin passes unconditionally (DMA instantaneous)."""
    return 0


@inst_register.custom0(name='credit.st.chk', funct7=0b1010011, funct3=0)
def credit_st_chk(npu, proc, inst, cxt) -> int:
    """credit.st.chk (0x53): S-loop commits the deferred L2→DDR store queue.

    Sole flush trigger for plan-style (WSPLIT) firmware — see module docstring.
    """
    if npu.warp.is_sloop:
        npu.flush_deferred_ddr_stores()
    return 0
