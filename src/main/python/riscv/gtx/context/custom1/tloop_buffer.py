"""T-loop instruction buffering — scaffolding for thread-block fusion.

Inside ``__start_thread(tid)`` ... ``__end_thread(tid)`` the firmware
emits a tight ``(opset, load, abs.v, opset, store)`` cadence per row,
hundreds of times per SPU. Each row pays Python dispatch + 3 micro
torch ops; the actual ABS work is trivial in comparison.

This module sets up the *infrastructure* to capture those instructions
without changing per-handler code: when T-loop buffering is enabled
(by ``_do_startt`` in :mod:`unit.context.control`), :mod:`execute`
calls :func:`try_buffer` for bufferable mnemonics, which snapshots all
register-file values the handler will need and skips the handler call.
At ``__end_thread`` (or any non-bufferable boundary), :func:`flush`
replays each entry in order through the same handler the FSM would
have resolved.

Fusion (turning the replay loop into one bulk ``np.abs`` over the
whole tile) is intentionally NOT done here — it lands in a follow-up
once this layer is proven correctness-neutral against ABS/NEG/EXP
regressions.

Why snapshots, not deferred ``proc``/``insn`` references:
  - ``proc.state.XPR[i]`` mutates between RoCC instructions (RISC-V
    scalar code advances the row counter, etc.), so values must be
    captured at buffer time.
  - ``gspr[GSPR['GSPR_GTX_OPERAND3'].address]`` (OPSET staging) is cleared by
    :mod:`writeback` immediately after each non-OPSET custom0, so the
    snapshot must also happen before ``state_writeback`` runs.
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

from ...csr import GSPR, LSPR
from ..disasm import Custom0_Insn
from .. import _resolve_nest_spu

if TYPE_CHECKING:
    from ...npu import GtxNpu

# OPSET staging-word offsets — direct-tensor access on the gspr buffer,
# mirroring DL/spr.py opset and DL/dma.py _operand3.
_OPERAND3_ADDR = GSPR['GSPR_GTX_OPERAND3'].address & 0x3FF   # 0x003
_OPERAND5_ADDR = GSPR['GSPR_GTX_OPERAND5'].address & 0x3FF   # 0x005
# L1 operand anchors — fusion only collapses frames whose vec reads/writes
# the same banks the load/store DMA targets (see _execute_fused guard).
_ADDRA = LSPR['SPM_ADDRA'].address
_ADDRB = LSPR['SPM_ADDRB'].address
_ADDRC = LSPR['SPM_ADDRC'].address
_ADDRR = LSPR['SPM_ADDRR'].address
# Scope-masked offsets for direct ``npu.lspr.tensor[nest, spu, off]`` access —
# avoids the RegisterFile-narrowing allocation in the buffer hot path.
_ADDRA_T = _ADDRA & 0x3FF
_ADDRB_T = _ADDRB & 0x3FF
_ADDRC_T = _ADDRC & 0x3FF
_ADDRR_T = _ADDRR & 0x3FF

# ── Fusion diagnostics (GTX_DEBUG_FUSION=1) — counts which path each
# buffered frame takes + a one-shot sample of the contiguity operands.
import os as _os
_FUSE_DBG = bool(_os.environ.get("GTX_DEBUG_FUSION"))
_FSTAT: dict = {}
_FSAMPLE: dict = {}


def _fs(key: str, n: int = 1) -> None:
    _FSTAT[key] = _FSTAT.get(key, 0) + n


if _FUSE_DBG:
    import atexit as _atexit
    import sys as _sys

    @_atexit.register
    def _dump_fstat() -> None:
        _sys.stderr.write("[FUSION] " + " ".join(
            f"{k}={v}" for k, v in sorted(_FSTAT.items())) + "\n")
        if _FSAMPLE:
            _sys.stderr.write("[FUSION] sample " + " ".join(
                f"{k}={v}" for k, v in _FSAMPLE.items()) + "\n")


# ----------------------------------------------------------------------
# Bufferable mnemonic set — per user selection (4 op families).
#
# Anything NOT in this set runs eagerly through the FSM. The two
# categories of eager ops are split by whether they drain the buffer:
#   - TRANSPARENT_MNEMONICS (below) run eagerly but leave the buffer
#     intact, so opset/wrspr can interleave with bufferable ops without
#     fragmenting the batch.
#   - Anything else (warp markers, tpose/fill, MM, etc.) is a hard
#     flush boundary — buffer drains in firmware-emitted order first.
# ----------------------------------------------------------------------
BUFFERABLE_MNEMONICS = frozenset({
    # DMA L2 ↔ L1 (T-loop branch)
    'load', 'store', 'copy',
    # Vec unary — MATH (0x1C) / SIGN (0x1D) / ROUND (0x1E)
    'abs.v', 'neg.v', 'sign.v', 'step.v',
    'ceil.v', 'trunc.v', 'floor.v', 'rne.v',
    'sqrt.v', 'exp.v', 'log.v',
    # Vec arith VV / II (0x18)
    'add.vv', 'sub.vv', 'mul.vv', 'div.vv',
    'add.ii', 'sub.ii', 'mul.ii', 'div.ii',
    # Vec SASMD VS / IS (0x10)
    'add.vs', 'sub.vs', 'mul.vs', 'div.vs',
    'add.is', 'sub.is', 'mul.is', 'div.is',
    # credit_ld / credit_st — functional-model counter inc/dec
    'credit.ld', 'credit.st',
})


# ----------------------------------------------------------------------
# Transparent mnemonics — run eagerly inside the FSM but do NOT drain the
# buffer. They mutate state that the NEXT bufferable op snapshots, so the
# buffer keeps growing across them instead of getting fragmented into
# tiny mid-thread flushes:
#
#   - ``opset``           stages GSPR_OPERAND3/5 for the next load/store
#     — buffering ``opset`` itself would lose ordering on the staging
#     write, so it stays eager but does not force a flush.
#   - ``credit_*_chk``    credit-gated dequeue point (260517-s9k): runs
#     eagerly in the FSM; the handler internally drains the T-loop
#     buffer at the chk boundary (TMU side, ``unit.context.dma.
#     _credit_ld_chk``) or the S-loop buffer (SMU side, see
#     :mod:`gtx.sloop_buffer` + ``_credit_st_chk``). Kept in the
#     TRANSPARENT set so the FSM does not force an additional hard
#     flush before the handler runs — the handler owns the drain.
# ----------------------------------------------------------------------
# NOTE: ``wrspr`` is intentionally NOT transparent. Composite intrinsics
# (__layernorm/__rmsnorm/__var) call __set_spm_addr (a wrspr) AND wrspr SGPR_0
# *inside* the per-row loop with buffered .vs/.is ops pending. If wrspr ran
# eagerly without draining, it would mutate SVR/SPM state ahead of the buffered
# ops that depend on the prior state (e.g. wrspr SGPR_0=x_num clobbers SVR_0=sum
# before the buffered div.is reads it). It must flush first. For the ABS hot
# loop wrspr only fires at thread-start with an empty buffer, so flush() is a
# cheap no-op there — zero perf impact.
TRANSPARENT_MNEMONICS = frozenset({
    'opset',
    'credit.chk', 'credit.ld.chk', 'credit.st.chk',
})


# Mnemonics handled by :func:`unit.ins.ops.vec._apply_unary` — the element-
# wise unaries that share the (read addr_a → torch op → write addr_r) path.
# These are the fusion candidates: a run of (load, vec_unary, store) frames
# with matching params collapses into a single bulk torch op.
_VEC_UNARY_MNEMONICS = frozenset({
    'abs.v', 'neg.v', 'sign.v', 'step.v',
    'ceil.v', 'trunc.v', 'floor.v', 'rne.v',
    'sqrt.v', 'exp.v', 'log.v',
})


# ----------------------------------------------------------------------
# Buffer entry — namedtuple keeps per-row allocation small (≤ ~80 B vs
# ~300 B for an equivalent dict) since the ABS hot path buffers ~1280
# entries per (tile, SPU).
#
# ``rs1`` / ``rs2`` are the *values* read from XPR at buffer time, NOT
# register indices — :class:`_InsnShim` always exposes ``rs1=0, rs2=1``
# and :class:`_XPRShim` maps those sentinels to the snapshotted values
# so handlers can do ``state.XPR[insn.rs1]`` unchanged.
# ----------------------------------------------------------------------
TLoopEntry = namedtuple(
    'TLoopEntry',
    ('handler', 'mnemonic',
     'rs1', 'rs2', 'op3', 'op5',
     'funct', 'xd', 'xs1_bit', 'xs2_bit', 'rd',
     # Per-frame SPM operand anchors (set via __set_spm_addr, which mutates
     # between buffered ops — e.g. SILU's neg→exp→add_vs→div each rebinds
     # ADDRA/ADDRB/ADDRR). Replay must restore these or every frame reads the
     # last frame's banks. Snapshotted with the same resolver the handler uses.
     'a_addr', 'b_addr', 'c_addr', 'r_addr'),
)


class _XPRShim:
    """Two-slot integer register file stand-in for replay.

    Handlers index XPR by ``insn.rs1`` / ``insn.rs2``. :class:`_InsnShim`
    pins those to 0 and 1, so this dict-free shim just returns the right
    snapshotted scalar per index.
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
        # Bufferable handlers never write to XPR (they return 0 / void).
        # Defensive no-op for future ops that opt into buffering.
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

    ``rs1`` / ``rs2`` are pinned to 0 / 1 so :class:`_XPRShim` can hand
    out the snapshotted values via a 2-slot lookup. The remaining fields
    mirror the live encoding so handlers that decode ``funct`` / ``xd``
    / ``xs1`` / ``xs2`` / ``rd`` (DMA, vec) behave identically.
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
# Public API — invoked from :mod:`execute` and :mod:`unit.context.control`
# ----------------------------------------------------------------------
def try_buffer(npu: 'GtxNpu', handler, proc, insn, mnemonic: str) -> None:
    """Snapshot one bufferable instruction into the T-loop buffer.

    ``handler`` is the registered ``(npu, proc, inst, cxt)`` callable;
    ``insn`` is the live pybind rocc_insn_t. Caller (``GtxNpu.custom0``)
    has already verified ``mnemonic in BUFFERABLE_MNEMONICS`` and that
    the buffer is armed, so no re-check here.

    Snapshots, not deferred ``proc``/``insn`` refs: ``XPR[i]`` and the
    OPSET staging GSPRs mutate between RoCC instructions, so values are
    captured now (see module docstring).
    """
    state = proc.state
    nest, spu = _resolve_nest_spu(npu)
    lt = npu.lspr.tensor   # (NEST, SPU, 1024) — direct index, no narrowing
    # Positional construction skips namedtuple's kwarg dispatch — the
    # difference is small per call (~200 ns) but multiplied by 1.18 M
    # buffered ops on the ABS hot path it adds up to ~0.3 s.
    npu._tloop_buf.append(TLoopEntry(
        handler,
        mnemonic,
        int(state.XPR[insn.rs1]),
        int(state.XPR[insn.rs2]),
        int(npu.gspr.tensor[_OPERAND3_ADDR]),
        int(npu.gspr.tensor[_OPERAND5_ADDR]),
        insn.funct,
        insn.xd,
        insn.xs1,
        insn.xs2,
        insn.rd,
        int(lt[nest, spu, _ADDRA_T]),
        int(lt[nest, spu, _ADDRB_T]),
        int(lt[nest, spu, _ADDRC_T]),
        int(lt[nest, spu, _ADDRR_T]),
    ))
    # OPSET staging lives for the NEXT instruction only — the vendor clears it
    # once an instruction consumes it. The snapshot above preserves the value
    # for replay, so clear the live GSPRs now (matching GtxNpu.custom0's eager
    # clear) or a buffered op's staged length leaks into the next eager op
    # (e.g. __load's OPERAND3=TOTAL_BYTES landing as mm.o's result-SVR slot).
    npu.gspr.tensor[_OPERAND3_ADDR] = 0
    npu.gspr.tensor[_OPERAND5_ADDR] = 0


def flush(npu: 'GtxNpu') -> None:
    """Drain the buffer with pattern fusion where possible.

    Walks the buffer once, trying :func:`_try_fuse_unary` at each
    position to collapse a run of ``(load, vec_unary, store)`` frames
    (with optional ``credit_ld`` / ``credit_st`` for ``__load_cr`` /
    ``__store_cr``) into a single bulk torch op. Anything that doesn't
    match falls through to :func:`_replay` — same shim-based handler
    invocation as the pre-fusion path, so non-fused mnemonics keep their
    eager-mode semantics.

    Re-buffering is blocked during the drain by swapping
    ``_tloop_buf`` to ``None``; ``_do_endt`` re-arms it.
    """
    buf = npu._tloop_buf
    if not buf:
        return
    npu._tloop_buf = None
    # _replay overwrites GSPR_OPERAND3/5 with each frame's snapshot and leaves
    # the last frame's values behind. A non-bufferable op (e.g. mm.v / max.vs)
    # triggers this drain *after* its own opset has staged OPERAND3/5, then
    # reads them eagerly once the drain returns — so save and restore the
    # staging words around the drain or the trigger op sees replay leftovers.
    saved_op3 = int(npu.gspr.tensor[_OPERAND3_ADDR])
    saved_op5 = int(npu.gspr.tensor[_OPERAND5_ADDR])
    _rn, _rs = _resolve_nest_spu(npu)
    _lt = npu.lspr.tensor
    saved_a = int(_lt[_rn, _rs, _ADDRA_T]); saved_b = int(_lt[_rn, _rs, _ADDRB_T])
    saved_c = int(_lt[_rn, _rs, _ADDRC_T]); saved_r = int(_lt[_rn, _rs, _ADDRR_T])
    try:
        _drain(npu, buf)
    finally:
        npu._tloop_buf = []
        npu.gspr.tensor[_OPERAND3_ADDR] = saved_op3
        npu.gspr.tensor[_OPERAND5_ADDR] = saved_op5
        _lt[_rn, _rs, _ADDRA_T] = saved_a; _lt[_rn, _rs, _ADDRB_T] = saved_b
        _lt[_rn, _rs, _ADDRC_T] = saved_c; _lt[_rn, _rs, _ADDRR_T] = saved_r


def _drain(npu: 'GtxNpu', buf) -> None:
    """Walk the buffer, fuse where possible, sequential-replay otherwise."""
    i = 0
    n = len(buf)
    while i < n:
        consumed = _try_fuse_unary(npu, buf, i)
        if consumed > 0:
            i += consumed
        else:
            _replay(npu, buf[i])
            i += 1


# ----------------------------------------------------------------------
# Frame fusion — collapse (load, [credit_ld], vec_unary, store, [credit_st])
# runs into one bulk torch op. Targets the ABS-style inner loop:
#
#   for r in 0..N-1:
#     opset(write_stride)                        # TRANSPARENT (eager)
#     [load_cr or load] L2[A_off+r*length] →
#                          L1[BANK_A]            # buffered
#     [credit_ld]  (only on r == N-1)            # buffered
#     abs.v / neg.v / ... rs1=vec_size           # buffered
#     opset(read_stride)                         # TRANSPARENT
#     [store_cr or store] L1[BANK_R] →
#                            L2[R_off+r*length]  # buffered
#     [credit_st]  (only on r == N-1)            # buffered
#
# Each row is one Python ``np.abs(view)``-class call, dominated by
# PyTorch dispatch overhead. Detecting N identical-shape frames lets us
# read the N-row L2 slab once, run the unary on the full ``(N, vec_size)``
# tensor, and write the slab back — one torch op per kernel instead of
# N micro-ops. Credit counters still replay individually (state parity).
# ----------------------------------------------------------------------
class _Frame:
    """One inner-loop frame with its DMA decode results cached.

    ``load_args`` / ``store_args`` hold the 6-tuple returned by
    :func:`_decode_dma` — :func:`_frame_signature` and
    :func:`_execute_fused` both peek at the L2 offset, so caching once
    at parse time saves the second decode per call.
    """

    __slots__ = ('load', 'cred_ld', 'vec', 'store', 'cred_st', 'end',
                 'load_args', 'store_args')

    def __init__(self, load, cred_ld, vec, store, cred_st, end,
                  load_args, store_args):
        self.load = load
        self.cred_ld = cred_ld
        self.vec = vec
        self.store = store
        self.cred_st = cred_st
        self.end = end
        self.load_args = load_args
        self.store_args = store_args


def _parse_frame(buf, i):
    """Try to parse one ``load[, cred_ld], vec_unary, store[, cred_st]``
    frame starting at ``i``. Returns ``_Frame`` or ``None``.
    """
    n = len(buf)
    if i >= n or buf[i].mnemonic != 'load':
        return None
    load = buf[i]
    j = i + 1

    cred_ld = None
    if j < n and buf[j].mnemonic == 'credit.ld':
        cred_ld = buf[j]
        j += 1

    if j >= n or buf[j].mnemonic not in _VEC_UNARY_MNEMONICS:
        return None
    vec = buf[j]
    j += 1

    if j >= n or buf[j].mnemonic != 'store':
        return None
    store = buf[j]
    j += 1

    cred_st = None
    if j < n and buf[j].mnemonic == 'credit.st':
        cred_st = buf[j]
        j += 1

    return _Frame(load, cred_ld, vec, store, cred_st, j,
                   _decode_dma(load), _decode_dma(store))


def _decode_dma(entry: TLoopEntry):
    """Mirror of :func:`dma_engine.decode_firmware_dma_args` working off a
    snapshotted entry. Returns the 6-tuple
    ``(addr_hi, addr_lo, height, length, rd_stride, wr_stride)``.
    """
    f3 = (entry.xd << 2) | (entry.xs1_bit << 1) | entry.xs2_bit
    is_store = bool(f3 & 1)
    is_copy = (not is_store) and bool(f3 & 2)
    addr_hi = (entry.rs1 >> 32) if is_copy else ((entry.rs1 >> 27) & 0x1FFFFFFFFF)
    addr_lo = entry.rs1 & 0x7FFFFFF
    h_raw = (entry.rs2 >> 48) & 0xFFFF
    l_raw = (entry.rs2 >> 32) & 0xFFFF
    rs2_low = entry.rs2 & 0xFFFFFFFF
    rs3_low = entry.op3 & 0xFFFFFFFF
    height = 1 if h_raw == 0 else h_raw
    length = 0x10000 if l_raw == 0 else l_raw
    if is_store:
        wr_stride, rd_stride = rs2_low, rs3_low
    else:
        rd_stride, wr_stride = rs2_low, rs3_low
    return addr_hi, addr_lo, height, length, rd_stride, wr_stride


def _frame_signature(frame: _Frame):
    """Hashable tuple capturing everything fusion needs to match across
    frames. Excludes the L2 offset (``addr_hi``) — that one is allowed to
    progress by a uniform stride between frames.

    Reuses the ``load_args`` / ``store_args`` decode results that
    :func:`_parse_frame` already computed, so the comparison costs are
    one tuple build + the vec field reads, no second decode pass.
    """
    _, l_lo, l_h, l_len, l_rds, l_wrs = frame.load_args
    _, s_lo, s_h, s_len, s_rds, s_wrs = frame.store_args
    v = frame.vec
    return (
        l_lo, l_h, l_len, l_rds, l_wrs,
        s_lo, s_h, s_len, s_rds, s_wrs,
        v.funct, v.xd, v.xs1_bit, v.xs2_bit, v.rs1, v.rs2,
    )


def _try_fuse_unary(npu, buf, start):
    """Match the longest run of compatible frames starting at ``start``
    and execute it as a single bulk op. Returns the number of buffer
    entries consumed, or 0 if no fusion was applied.
    """
    f0 = _parse_frame(buf, start)
    if f0 is None:
        if _FUSE_DBG:
            _fs('parse_fail')
        return 0

    sig0 = _frame_signature(f0)
    frames = [f0]
    cursor = f0.end
    while cursor < len(buf):
        nxt = _parse_frame(buf, cursor)
        if nxt is None or _frame_signature(nxt) != sig0:
            break
        frames.append(nxt)
        cursor = nxt.end

    if len(frames) < 2:
        # Single frame — torch op overhead is identical to sequential
        # replay, so don't bother with the bulk path. Let the caller fall
        # through to plain replay.
        if _FUSE_DBG:
            _fs('single_frame')
        return 0

    if _FUSE_DBG:
        _fs('groups')
        _fs('grouped_frames', len(frames))
    _execute_fused(npu, frames)
    return cursor - start


def _execute_fused(npu, frames) -> None:
    """Bulk-execute N identical-shape frames as one torch op.

    Fast path requires uniform L2 stride and ``length == vec_size *
    MX_EXT_BYTES`` (single contiguous row per frame, ABS firmware shape).
    Anything else falls back to a per-frame :func:`_replay` so we never
    miscompile an unusual stride layout.
    """
    import numpy as np  # local to keep module import cycle-free at top level
    from ..custom0.MX.vector import _apply_unary
    from ...config_params import (
        MX_IO_DTYPE, MX_IO_BYTES, MX_EXT_DTYPE, MX_EXT_BYTES)

    n = len(frames)
    f0 = frames[0]
    l_hi0, l_lo, l_h, l_len, _, _ = f0.load_args
    s_hi0, s_lo, _, _, _, _ = f0.store_args

    # Cached decodes from :func:`_parse_frame` — saves a second decode
    # pass across all N frames (was ~0.5 s on ABS, 1.58 M decode calls
    # vs 0.78 M after the cache).
    src_offs = [f.load_args[0] for f in frames]
    dst_offs = [f.store_args[0] for f in frames]
    src_step = src_offs[1] - src_offs[0]
    dst_step = dst_offs[1] - dst_offs[0]

    vec0 = f0.vec
    funct7 = vec0.funct
    sub_op = ((vec0.xd << 2) | (vec0.xs1_bit << 1) | vec0.xs2_bit) & 3
    vec_size = (vec0.rs1 & 0xFFFF) or 0x10000

    nest = npu.warp.current_nest if npu.warp.is_ploop else 0
    spu = npu.warp.current_spu
    lspr = npu.lspr[nest][spu]

    # Fast path requires: height-1 single contiguous row per frame,
    # length == vec_size * MX_IO_BYTES, uniform L2 stride, AND the vec's
    # L1 banks (SPM_ADDRA/ADDRR) coincide with the load dest / store src
    # so the L2→abs→L2 shortcut equals the eager L2→L1→abs→L1→L2 chain.
    # Anything else falls back to per-frame replay (never miscompiles).
    contiguous = (
        l_h == 1 and
        l_len == vec_size * MX_EXT_BYTES and
        src_step == l_len and dst_step == l_len and
        lspr.get(_ADDRA, 0) == l_lo and lspr.get(_ADDRR, 0) == s_lo and
        all(src_offs[i] == src_offs[0] + i * l_len for i in range(n)) and
        all(dst_offs[i] == dst_offs[0] + i * l_len for i in range(n))
    )

    if _FUSE_DBG:
        _fs('exec_fused')
        if l_h != 1: _fs('fail_height')
        if l_len != vec_size * MX_EXT_BYTES: _fs('fail_len')
        if src_step != l_len: _fs('fail_srcstep')
        if dst_step != l_len: _fs('fail_dststep')
        if lspr.get(_ADDRA, 0) != l_lo: _fs('fail_addra')
        if lspr.get(_ADDRR, 0) != s_lo: _fs('fail_addrr')
        if 'l_h' not in _FSAMPLE:
            _FSAMPLE.update(dict(
                l_h=l_h, l_len=l_len, vec_size=vec_size, mxb=MX_IO_BYTES,
                src_step=src_step, dst_step=dst_step,
                l_lo=l_lo, addr_a=int(lspr.get(_ADDRA, 0)),
                s_lo=s_lo, addr_r=int(lspr.get(_ADDRR, 0))))
        if contiguous: _fs('bulk')

    if not contiguous:
        _replay_frames(npu, frames)
        return

    l2 = npu.mem.l2_byte(nest)
    src_base = src_offs[0]
    dst_base = dst_offs[0]
    total_bytes = n * l_len               # L2 EXTERNAL bytes (fp16)

    # Read N rows from L2 (MX_EXT_DTYPE) and widen to the compute dtype —
    # mirrors the dtype-converting T-loop load (dma_imp). When ext==io the
    # .to() is a no-op so this stays the original zero-copy fp16 path.
    src_io = (
        l2[src_base:src_base + total_bytes]
        .view(MX_EXT_DTYPE)
        .reshape(n, vec_size)
        .astype(MX_IO_DTYPE)
    )

    # One torch op for all rows — pyTorch dispatch cost is amortised
    # across the whole tile. ``mode`` = vec rs2[1:0] (exp/ln base select).
    result_io = _apply_unary(funct7, sub_op, src_io, vec0.rs2 & 0x3)

    # Write N rows back to L2 dst slab, narrowing io→ext (mirrors store).
    dst_view = l2[dst_base:dst_base + total_bytes]
    dst_view[...] = result_io.astype(MX_EXT_DTYPE).reshape(-1).view(np.uint8)

    # Maintain L1 invariant: the non-fused path leaves the LAST row's input
    # at ``BANK_A`` and output at ``BANK_R`` — both in the L1 IO dtype
    # (vec_size * MX_IO_BYTES per row), matching the eager load/abs writes.
    l1 = npu.mem.l1_byte(nest, spu)
    io_row = vec_size * MX_IO_BYTES
    l1[l_lo:l_lo + io_row][...] = src_io[-1].view(np.uint8)
    l1[s_lo:s_lo + io_row][...] = result_io[-1].view(np.uint8)

    # Credit counters are independent state — replay only the entries
    # firmware actually emitted (last iter for __load_cr / __store_cr).
    for f in frames:
        if f.cred_ld is not None:
            _replay(npu, f.cred_ld)
        if f.cred_st is not None:
            _replay(npu, f.cred_st)


def _replay_frames(npu, frames) -> None:
    """Fallback path: sequential replay of every entry in every frame."""
    for f in frames:
        _replay(npu, f.load)
        if f.cred_ld is not None:
            _replay(npu, f.cred_ld)
        _replay(npu, f.vec)
        _replay(npu, f.store)
        if f.cred_st is not None:
            _replay(npu, f.cred_st)


def _replay(npu: 'GtxNpu', entry: TLoopEntry) -> None:
    """Invoke ``entry.handler`` with shims that return snapshotted values.

    Restores the OPSET staging GSPRs (OPERAND3/5) to the values captured
    when the instruction was buffered, then runs the handler with the
    ``(npu, proc, inst, cxt)`` signature the registry uses. No clear
    afterwards: OPERAND3/5 persist in this architecture (set by opset,
    live until the next opset), so the last replayed entry leaves the
    same staging value the eager last-opset would — post-flush state
    stays byte-identical.
    """
    npu.gspr.tensor[_OPERAND3_ADDR] = entry.op3
    npu.gspr.tensor[_OPERAND5_ADDR] = entry.op5
    # Restore this frame's SPM operand anchors — handlers read ADDRA/B/C/R
    # live from LSPR, and __set_spm_addr may have rebound them between frames.
    nest, spu = _resolve_nest_spu(npu)
    lt = npu.lspr.tensor
    lt[nest, spu, _ADDRA_T] = entry.a_addr
    lt[nest, spu, _ADDRB_T] = entry.b_addr
    lt[nest, spu, _ADDRC_T] = entry.c_addr
    lt[nest, spu, _ADDRR_T] = entry.r_addr

    insn = Custom0_Insn(entry.mnemonic, _InsnShim(
        entry.funct, entry.xd, entry.xs1_bit, entry.xs2_bit, entry.rd,
    ))
    proc = _ProcShim(_StateShim(_XPRShim(entry.rs1, entry.rs2)))
    entry.handler(npu, proc, insn, npu.CONTEXT)
