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


# SPU bit positions for expanding a credit.st target_spu mask into a per-SPU
# decrement vector: ``(mask >> _SPU_BITS) & 1``.
_SPU_BITS = np.arange(SPU_NUM, dtype=np.int32)


def apply_deferred_st(npu) -> None:
    """Apply the S-loop ``credit.st`` decrements deferred to the plan boundary.

    Called from ``end.p`` and ``WJOIN`` (control.py). Sequential pyspike runs
    the S-loop store consume (``credit.st--``) before the T-loop produce
    (``credit.st++``) within a tile, so an eager decrement underflows on the
    first tile and leaves ``credit.st==1`` at every WJOIN. Hardware runs
    SMU/TMU concurrently (``++`` then ``--``); deferring the decrement to the
    plan boundary restores that balance. The deferred amount is per-SPU,
    masked to the SPUs the store actually covers (``target_spu``), so inactive
    SPUs in a partial last tile are never spuriously decremented.

    Data-path-independent: nothing reads the counters mid-execution
    (``credit.chk`` is a NOP/flush), so this only affects the strict
    diagnostics, never the DDR result.
    """
    pend = npu._credit_st_deferred
    if not pend.any():
        return
    for nest in range(NEST_NUM):
        dec = pend[nest]
        if not dec.any():
            continue
        row = npu._credit_st[nest]
        short = dec > row
        if short.any():
            for spu in np.nonzero(short)[0]:
                print(f"[GTX_CREDIT_ERROR] credit.st decrement when already 0 "
                      f"(plz check firmware) - nest{nest} spu{int(spu)}",
                      file=sys.stderr, flush=True)
        np.subtract(row, dec, out=row)
        np.maximum(row, 0, out=row)
    pend.fill(0)


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
    """credit.st (0x51): T-loop compute done → inc; S-loop DMA-store consume → dec.

    rs1 is the ``target_spu`` mask (``__store_cr``'s active_tid_mask). The
    S-loop consume is *deferred* per-SPU to the plan boundary (end.p / WJOIN)
    so it settles after the T-loop produce — see :func:`apply_deferred_st`."""
    warp = npu.warp
    nest = warp.current_nest if warp.is_ploop else 0
    if nest < NEST_NUM:
        if warp.is_tloop and warp.current_spu < SPU_NUM:
            npu._credit_st[nest, warp.current_spu] += 1
        elif warp.is_sloop:
            mask = int(proc.state.XPR[inst.rs1]) & 0xFFFF
            npu._credit_st_deferred[nest] += (mask >> _SPU_BITS) & 1
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
