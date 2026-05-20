"""

-------------------------------------------------------------------------
C2 (start_p + start_s): ``credit_ld`` inc, ``credit_st`` dec,
                        ``credit_st_chk``, ``credit_ld_chk`` all valid.
C3 (start_p + start_t): ``credit_ld`` dec, ``credit_st`` inc,
                        ``credit_st_chk``, ``credit_ld_chk`` all valid.

=> ``credit_st_chk`` is what SMU (C2) consults — dequeues ``_sloop_buf``.
   ``credit_ld_chk`` is what TMU (C3) consults — dequeues ``_tloop_buf``.
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

from ...csr import GSPR

if TYPE_CHECKING:
    from ...npu import GtxNpu


# ----------------------------------------------------------------------
# Bufferable mnemonic set — minimal SMU-only emission set.
#
# Rationale (260517-s9k): SMU emits only DMA + counter mnemonics between
# ``start_s..end_s`` in vendor `.elf` (verified against ABS firmware in
# P8 SUMMARYs; see STATE.md Phase 8 Plan 04 "credit_ld_chk flush wiring"
# entry). Anything else inside an S-loop section would be a firmware
# anomaly — let it fall through to the eager FSM path and surface as a
# regression rather than silently buffering it.
# ----------------------------------------------------------------------
SLOOP_BUFFERABLE_MNEMONICS = frozenset({
    # DMA DDR <-> L2 (S-loop branch)
    'load', 'store', 'copy',
    # Functional-model counter inc/dec
    'credit_ld', 'credit_st',
})


# ----------------------------------------------------------------------
# Transparent mnemonics — run eagerly inside the FSM but do NOT force a
# hard flush. ``credit_*_chk`` are listed here because the chk handler
# itself owns the dequeue (see :mod:`unit.context.dma` post-260517-s9k);
# the FSM must hand it off eagerly rather than draining the buffer
# AHEAD of the handler — that would defeat the producer-consumer ordering.
# ----------------------------------------------------------------------
SLOOP_TRANSPARENT_MNEMONICS = frozenset({
    'opset',
    'wrspr',
    'credit_chk', 'credit_ld_chk', 'credit_st_chk',
})


# ----------------------------------------------------------------------
# Buffer entry — identical layout to :class:`tloop_buffer.TLoopEntry` so
# the shim machinery (XPRShim / StateShim / ProcShim / InsnShim) can be
# duplicated cheaply. We deliberately do NOT share/refactor across the
# two modules — the user spec ("Existing tloop_buffer.py fusion logic
# MUST be preserved — barrier semantics are additive") requires the
# T-loop path to remain untouched; cross-module import risks fragile
# coupling.
# ----------------------------------------------------------------------
SLoopEntry = namedtuple(
    'SLoopEntry',
    ('handler', 'mnemonic',
     'rs1', 'rs2', 'op3', 'op5',
     'funct', 'xd', 'xs1_bit', 'xs2_bit', 'rd'),
)


class _XPRShim:
    """Two-slot integer register file stand-in for replay.

    Mirror of :class:`tloop_buffer._XPRShim` — handlers index XPR by
    ``insn.rs1`` / ``insn.rs2``; :class:`_InsnShim` pins those to 0 / 1
    so this dict-free shim returns the snapshotted scalar per index.
    """
    __slots__ = ('_v0', '_v1')

    def __init__(self, v0: int, v1: int) -> None:
        self._v0 = v0
        self._v1 = v1

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self._v0
        if idx == 1:
            return self._v1
        return 0

    def write(self, idx: int, val: int) -> None:
        # SMU bufferable handlers never write XPR (return 0 / void).
        pass


class _StateShim:
    __slots__ = ('XPR',)

    def __init__(self, xpr: _XPRShim) -> None:
        self.XPR = xpr


class _ProcShim:
    __slots__ = ('state',)

    def __init__(self, state: _StateShim) -> None:
        self.state = state


class _InsnShim:
    """Frozen rocc_insn_t stand-in for replay.

    ``rs1`` / ``rs2`` pinned to 0 / 1 so :class:`_XPRShim` hands out the
    snapshotted values via a 2-slot lookup. Other fields mirror the live
    encoding so handlers that decode ``funct`` / ``xd`` / ``xs1`` /
    ``xs2`` / ``rd`` (DMA) behave identically.
    """
    __slots__ = ('funct', 'xd', 'xs1', 'xs2', 'rd', 'rs1', 'rs2')

    def __init__(self, funct: int, xd: int, xs1: int, xs2: int, rd: int) -> None:
        self.funct = funct
        self.xd = xd
        self.xs1 = xs1
        self.xs2 = xs2
        self.rd = rd
        self.rs1 = 0
        self.rs2 = 1


# ----------------------------------------------------------------------
# Public API — invoked from :mod:`execute`, :mod:`unit.context.control`,
# and :mod:`unit.context.dma` (credit-chk dequeue).
# ----------------------------------------------------------------------
def try_buffer(npu: 'GtxNpu') -> bool:
    """Snapshot the current FSM instruction into the S-loop buffer.

    Returns True iff the instruction was buffered (caller must skip the
    handler call). False means the caller should flush any pending
    buffer and run the handler eagerly.

    Preconditions: ``npu._sloop_buf is not None`` and
    ``npu.warp.is_sloop`` are both true (checked by caller for hot-path
    parity with :func:`tloop_buffer.try_buffer`).
    """
    mnemonic = npu._ctx.get("mnemonic")
    if mnemonic not in SLOOP_BUFFERABLE_MNEMONICS:
        return False

    handler = npu._ctx["handler"]
    proc = npu._ctx["proc"]
    insn = npu._ctx["insn"]
    state = proc.state

    npu._sloop_buf.append(SLoopEntry(
        handler,
        mnemonic,
        int(state.XPR[insn.rs1]),
        int(state.XPR[insn.rs2]),
        int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)),
        int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND5'].address, 0)),
        insn.funct,
        insn.xd,
        insn.xs1,
        insn.xs2,
        insn.rd,
    ))
    return True


def flush(npu: 'GtxNpu') -> None:
    """Drain the S-loop buffer sequentially.

    No fusion: SMU instruction mix is DMA + control (not the 1.18 M-entry
    vec hot loop tloop_buffer is optimized for), so sequential replay is
    correctness-sufficient and keeps surface area minimal. If a future
    SMU hot path appears, mirror :func:`tloop_buffer._drain` /
    :func:`tloop_buffer._try_fuse_unary`.

    Re-buffering is blocked during the drain by swapping
    ``_sloop_buf`` to ``None``; the caller (``_do_ends`` or
    ``_credit_st_chk``) re-arms it (or leaves it ``None`` if the S-loop
    section is closing).
    """
    buf = npu._sloop_buf
    if not buf:
        return
    npu._sloop_buf = None
    try:
        for entry in buf:
            _replay(npu, entry)
    finally:
        npu._sloop_buf = []


def dequeue_one_batch(npu: 'GtxNpu') -> int:
    """Pop and replay ONE batch from the S-loop buffer.

    A batch = entries up to AND INCLUDING the next ``credit_ld``
    mnemonic — the producer-consumer handshake unit in the vendor
    ``use_tmu_queue`` push/pop pattern. Returns the count of entries
    replayed. If ``_sloop_buf is None`` or empty, returns 0.

    Currently called by :func:`unit.context.dma._credit_st_chk` for
    cycle-accurate-parity scaffolding. The functional model treats one
    chk-invocation as "drain everything emitted so far this section"
    because DMA is instantaneous; this helper exists so the future
    cycle-accurate path can switch from :func:`flush` to per-batch
    dequeue without changing the chk-handler call site.
    """
    buf = npu._sloop_buf
    if not buf:
        return 0

    # Find the end of the next batch: first credit_ld OR end of buffer.
    end = len(buf)
    for i, entry in enumerate(buf):
        if entry.mnemonic == 'credit_ld':
            end = i + 1  # inclusive
            break

    npu._sloop_buf = None
    try:
        replayed = 0
        for entry in buf[:end]:
            _replay(npu, entry)
            replayed += 1
    finally:
        # Restore remainder for subsequent dequeue calls.
        npu._sloop_buf = buf[end:] if end < len(buf) else []

    return replayed


def _replay(npu: 'GtxNpu', entry: SLoopEntry) -> None:
    """Invoke ``entry.handler`` with shims that return snapshotted values.

    Staging GSPRs (OPERAND3 / OPERAND5) are restored just before the
    handler call and re-cleared after — mirror of
    :func:`tloop_buffer._replay`. Every SMU bufferable mnemonic is
    non-OPSET so the post-clear is unconditional.
    """
    npu.gspr[GSPR['GSPR_GTX_OPERAND3'].address] = entry.op3
    npu.gspr[GSPR['GSPR_GTX_OPERAND5'].address] = entry.op5

    insn = _InsnShim(
        funct=entry.funct, xd=entry.xd,
        xs1=entry.xs1_bit, xs2=entry.xs2_bit, rd=entry.rd,
    )
    proc = _ProcShim(_StateShim(_XPRShim(entry.rs1, entry.rs2)))
    entry.handler(proc, insn, entry.rs1, entry.rs2)

    npu.gspr[GSPR['GSPR_GTX_OPERAND3'].address] = 0
    npu.gspr[GSPR['GSPR_GTX_OPERAND5'].address] = 0
