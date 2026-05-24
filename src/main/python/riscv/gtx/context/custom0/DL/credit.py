"""Credit counters + checks — port of gtx_npu_custom0.cc:646-694.

Per-(NEST, SPU) load/store credit counters that gate the DMA<->compute
hand-off. In pyspike's eager functional model the *_chk spins always pass
(DMA is instantaneous), so the counters are tracked for vendor 1:1 parity
but never stall control flow.

The deferred L2→DDR store queue is committed at the **plan boundary** — WJOIN
(``control.py:join``) for WSPLIT firmware, ``end.p`` otherwise — plus the
on-overwrite ``flush_deferred_if_l2_overlap`` guard. Both run AFTER the T-loop
thread section, so a shared ``store`` emitted before its producing thread (in
program order) still reads correct, post-compute L2. ``credit.chk`` does NOT
flush (it did, historically — see :func:`credit_chk` — but that committed
store-before-compute kernels like MUL_MAT/NORM with stale L2). Per-tile
correctness comes from the per-(``__split``/``__join``) WJOIN flush, not from a
credit.chk flush.

ISA note: there is a single ``credit.chk`` (funct7 0x53) — no separate
credit.ld.chk/st.chk. The firmware emits 0x53 (the encoding the SystemC ISS and
vendor spike decode); 0x52 is a legacy binutils mis-encoding kept as an alias.
"""
import os
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


def credit_chk(npu, proc, inst, cxt) -> int:
    """credit.chk: the single credit barrier (there is no credit.ld.chk/st.chk —
    that was a mis-split of this one instruction). In pyspike's eager functional
    model the spin passes immediately (DMA instantaneous), so it is a NOP.

    It does NOT flush the deferred L2→DDR store queue. Flushing here was wrong:
    when a kernel emits the shared ``store`` BEFORE the producing T-loop thread
    (program order: ``__start_shared``/store … then ``__start_thread``/compute,
    e.g. MUL_MAT's per-thread store loop or NORM/GROUP_NORM/RMS_NORM), the
    credit.chk that precedes/interleaves the store committed it while L2 still
    held stale (pre-compute) data → partial/zero output. On real HW (ISS) the
    barrier blocks until the thread's ``credit.st``; pyspike models that by
    deferring the commit to the plan boundary (WJOIN, control.py:join) plus the
    on-overwrite ``flush_deferred_if_l2_overlap`` guard — both run AFTER the
    thread section, so the queue reads correct L2. Verified 2026-05-24: a full
    95-op noflush sweep matched ISS on MUL_MAT/NORM/GROUP_NORM/RMS_NORM with
    0 regressions. Set ``GTX_CHK_FLUSH=1`` to restore the legacy eager flush.
    """
    if npu.warp.is_sloop and os.environ.get("GTX_CHK_FLUSH") == "1":
        npu.flush_deferred_ddr_stores()
    return 0


# Canonical encoding funct7=0x53 (what the SystemC ISS / vendor spike decode, and
# what the firmware now emits via .insn). 0x52 is the legacy mis-encoding older
# binutils produced for the same `credit.chk` mnemonic — aliased for robustness.
inst_register.custom0(name='credit.chk', funct7=0b1010011, funct3=0)(credit_chk)
inst_register.custom0(name='credit.chk', funct7=0b1010010, funct3=0)(credit_chk)
