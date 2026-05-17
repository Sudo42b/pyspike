"""EXECUTE state — invoke the resolved handler.

Miss (``handler=None`` — funct7/context/funct3 unmapped) leaves
``rd=0`` (silent NOP), matching pre-FSM dispatch semantics and vendor
parity.

T-loop buffering hook (inserted 2026-05-13): when ``npu._tloop_buf``
is non-None and ``npu.warp.is_tloop`` is True, bufferable mnemonics
(see :mod:`gtx.tloop_buffer`) are snapshotted into the buffer and the
handler call is skipped. Non-bufferable ops act as flush boundaries —
they drain the pending buffer first, then run eagerly.
"""
from __future__ import annotations

from .fsm import NpuState
from .tloop_buffer import (
    BUFFERABLE_MNEMONICS, TRANSPARENT_MNEMONICS, try_buffer, flush,
)
from .sloop_buffer import (
    SLOOP_BUFFERABLE_MNEMONICS,
    SLOOP_TRANSPARENT_MNEMONICS,
    try_buffer as sloop_try_buffer,
    flush as sloop_flush,
)


def state_execute(npu) -> NpuState:
    """Call ``ctx['handler'](proc, insn, xs1, xs2)``; store result in
    ``ctx['rd']``. Returns :attr:`NpuState.WRITEBACK`.
    """
    handler = npu._ctx["handler"]
    if handler is None:
        return NpuState.WRITEBACK

    buf = npu._tloop_buf
    if buf is not None and npu.warp.is_tloop:
        mnemonic = npu._ctx.get("mnemonic")
        if mnemonic in BUFFERABLE_MNEMONICS:
            try_buffer(npu)
            return NpuState.WRITEBACK
        # Transparent ops (opset / wrspr / credit_*_chk) run eagerly so the
        # next bufferable snapshot sees their state mutation, but they do
        # NOT drain the buffer — that keeps the inner-loop opset…load…
        # abs.v…opset…store cadence in one batch all the way to ``end_t``.
        # Anything else (warp markers, fill, tpose, MM, etc.) is a hard
        # flush boundary: drain in firmware-emitted order before running.
        if buf and mnemonic not in TRANSPARENT_MNEMONICS:
            flush(npu)

    # S-loop buffering hook (260517-s9k). Mirror of the T-loop branch above;
    # ``warp.is_sloop`` and ``warp.is_tloop`` are mutually exclusive per the
    # FSM. ``credit_*_chk`` is in SLOOP_TRANSPARENT_MNEMONICS so the chk
    # handler runs eagerly and owns the dequeue itself (see dma.py).
    sbuf = npu._sloop_buf
    if sbuf is not None and npu.warp.is_sloop:
        mnemonic = npu._ctx.get("mnemonic")
        if mnemonic in SLOOP_BUFFERABLE_MNEMONICS:
            sloop_try_buffer(npu)
            return NpuState.WRITEBACK
        if sbuf and mnemonic not in SLOOP_TRANSPARENT_MNEMONICS:
            sloop_flush(npu)

    npu._ctx["rd"] = handler(
        npu._ctx["proc"],
        npu._ctx["insn"],
        npu._ctx["xs1"],
        npu._ctx["xs2"],
    )
    return NpuState.WRITEBACK
