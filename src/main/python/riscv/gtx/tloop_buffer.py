"""T-loop instruction buffering — scaffolding for thread-block fusion.

Inside ``__start_thread(tid)`` ... ``__end_thread(tid)`` the firmware
emits a tight ``(opset, load, abs.v, opset, store)`` cadence per row,
hundreds of times per SPU. Each row pays Python dispatch + 3 micro
xp ops; the actual ABS work is trivial in comparison.

This module sets up the *infrastructure* to capture those instructions
without changing per-handler code: when T-loop buffering is enabled
(by ``_do_startt`` in :mod:`unit.context.control`), :mod:`execute`
calls :func:`try_buffer` for bufferable mnemonics, which snapshots all
register-file values the handler will need and skips the handler call.
At ``__end_thread`` (or any non-bufferable boundary), :func:`flush`
replays each entry in order through the same handler the FSM would
have resolved.

Fusion (turning the replay loop into one bulk ``xp.abs`` over the
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

from .config_params import xp
from .unit.csr import GSPR

if TYPE_CHECKING:
    from .npu import GtxNpu


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
    'abs_v', 'neg_v', 'sign_v', 'step_v',
    'ceil_v', 'trunc_v', 'floor_v', 'rne_v',
    'sqrt_v', 'exp_v', 'log_v',
    # Vec arith VV / II (0x18)
    'add_vv', 'sub_vv', 'mul_vv', 'div_vv',
    'add_ii', 'sub_ii', 'mul_ii', 'div_ii',
    # Vec SASMD VS / IS (0x10)
    'add_vs', 'sub_vs', 'mul_vs', 'div_vs',
    'add_is', 'sub_is', 'mul_is', 'div_is',
    # credit_ld / credit_st — functional-model counter inc/dec
    'credit_ld', 'credit_st',
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
#   - ``wrspr``           writes LSPR for the current (NEST, SPU); fires
#     a handful of times at thread-start (``__set_spm_addr``) and never
#     inside the inner loop, so the buffer is empty when these run.
#   - ``credit_*_chk``    credit-gated dequeue point (260517-s9k): runs
#     eagerly in the FSM; the handler internally drains the T-loop
#     buffer at the chk boundary (TMU side, ``unit.context.dma.
#     _credit_ld_chk``) or the S-loop buffer (SMU side, see
#     :mod:`gtx.sloop_buffer` + ``_credit_st_chk``). Kept in the
#     TRANSPARENT set so the FSM does not force an additional hard
#     flush before the handler runs — the handler owns the drain.
# ----------------------------------------------------------------------
TRANSPARENT_MNEMONICS = frozenset({
    'opset',
    'wrspr',
    'credit_chk', 'credit_ld_chk', 'credit_st_chk',
})


# Mnemonics handled by :func:`unit.ins.ops.vec._apply_unary` — the element-
# wise unaries that share the (read addr_a → xp op → write addr_r) path.
# These are the fusion candidates: a run of (load, vec_unary, store) frames
# with matching params collapses into a single bulk xp op.
_VEC_UNARY_MNEMONICS = frozenset({
    'abs_v', 'neg_v', 'sign_v', 'step_v',
    'ceil_v', 'trunc_v', 'floor_v', 'rne_v',
    'sqrt_v', 'exp_v', 'log_v',
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
     'funct', 'xd', 'xs1_bit', 'xs2_bit', 'rd'),
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
def try_buffer(npu: 'GtxNpu') -> bool:
    """Snapshot the current FSM instruction into the T-loop buffer.

    Returns True iff the instruction was buffered (caller must skip the
    handler call). False means the caller should flush any pending
    buffer and run the handler eagerly.

    Preconditions: ``npu._tloop_buf is not None`` and
    ``npu.warp.is_tloop`` are both true (checked by caller for hot-path
    speed).
    """
    mnemonic = npu._ctx.get("mnemonic")
    if mnemonic not in BUFFERABLE_MNEMONICS:
        return False

    handler = npu._ctx["handler"]
    proc = npu._ctx["proc"]
    insn = npu._ctx["insn"]
    state = proc.state

    # Positional construction skips namedtuple's kwarg dispatch — the
    # difference is small per call (~200 ns) but multiplied by 1.18 M
    # buffered ops on the ABS hot path it adds up to ~0.3 s.
    npu._tloop_buf.append(TLoopEntry(
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
    """Drain the buffer with pattern fusion where possible.

    Walks the buffer once, trying :func:`_try_fuse_unary` at each
    position to collapse a run of ``(load, vec_unary, store)`` frames
    (with optional ``credit_ld`` / ``credit_st`` for ``__load_cr`` /
    ``__store_cr``) into a single bulk xp op. Anything that doesn't
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
    try:
        _drain(npu, buf)
    finally:
        npu._tloop_buf = []


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
# runs into one bulk xp op. Targets the ABS-style inner loop:
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
# Each row is one Python ``xp.abs(view)``-class call, dominated by
# Python-level dispatch overhead. Detecting N identical-shape frames lets us
# read the N-row L2 slab once, run the unary on the full ``(N, vec_size)``
# ndarray, and write the slab back — one xp op per kernel instead of
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
    if j < n and buf[j].mnemonic == 'credit_ld':
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
    if j < n and buf[j].mnemonic == 'credit_st':
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
        v.funct, v.xd, v.xs1_bit, v.xs2_bit, v.rs1,
    )


def _try_fuse_unary(npu, buf, start):
    """Match the longest run of compatible frames starting at ``start``
    and execute it as a single bulk op. Returns the number of buffer
    entries consumed, or 0 if no fusion was applied.
    """
    f0 = _parse_frame(buf, start)
    if f0 is None:
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
        # Single frame — bulk path has no advantage over sequential replay;
        # let the caller fall through to plain replay.
        return 0

    _execute_fused(npu, frames)
    return cursor - start


def _execute_fused(npu, frames) -> None:
    """Bulk-execute N identical-shape frames as one xp op.

    Fast path requires uniform L2 stride and ``length == vec_size * 2``
    (single contiguous row per frame, exactly the ABS firmware shape).
    Anything else falls back to a per-frame :func:`_replay` so we never
    miscompile an unusual stride layout.
    """
    from .unit.ins.ops.vec import _apply_unary

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

    contiguous = (
        l_h == 1 and
        l_len == vec_size * 2 and
        src_step == l_len and dst_step == l_len and
        all(src_offs[i] == src_offs[0] + i * l_len for i in range(n)) and
        all(dst_offs[i] == dst_offs[0] + i * l_len for i in range(n))
    )

    if not contiguous:
        _replay_frames(npu, frames)
        return

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id

    # Read raw xp L2 storage directly (skip the byte/f16 accessor wrappers).
    l2 = npu.mem.l2[nest]
    src_base = src_offs[0]
    dst_base = dst_offs[0]
    total_bytes = n * l_len

    # Read N rows from L2 as a single (N, vec_size) fp16 ndarray — no copy,
    # just a strided view. xp `.view(dtype)` reinterprets; use `.reshape`
    # for shape changes.
    src_f16 = (
        l2[src_base:src_base + total_bytes]
        .view(xp.float16)
        .reshape(n, vec_size)
    )

    # One xp op for all rows — Python-level dispatch cost is amortised
    # across the whole tile.
    result_f16 = _apply_unary(funct7, sub_op, src_f16)

    # Write N rows back to L2 dst slab. Reshape src to 1D uint8 view to
    # match the dst's flat byte layout, then in-place copy.
    dst_view = l2[dst_base:dst_base + total_bytes]
    xp.copyto(dst_view, result_f16.reshape(-1).view(xp.uint8))

    # Maintain L1 invariant: the non-fused path would leave the LAST
    # row's input at ``BANK_A`` and the last row's output at ``BANK_R``.
    # Keep that observable, so end-of-thread debug dumps match.
    l1 = npu.mem.l1[nest, spu]
    last_src = src_offs[-1]
    xp.copyto(l1[l_lo:l_lo + l_len], l2[last_src:last_src + l_len])
    xp.copyto(l1[s_lo:s_lo + l_len], result_f16[-1].view(xp.uint8))

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

    Staging GSPRs (OPERAND3 / OPERAND5) are restored just before the
    handler call and re-cleared after, matching the OPSET-aware clear
    in :mod:`writeback` for non-OPSET custom0 — every bufferable
    mnemonic is non-OPSET so the clear is unconditional here.
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
