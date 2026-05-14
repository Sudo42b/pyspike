"""GtxNpu -- ``riscv.isa.ROCC`` subclass registered as ``"gtx"``.

FSM-driven dispatch (see :mod:`fsm`) with NPU context awareness
(C1/C2/C3/C4 — see :mod:`unit.context`). SPR storage uses
:class:`~unit.register_file.RegisterFile`, so addresses are indexed by
typed name (``self.gspr.GTX_OPERAND3``) wherever the source
register is declared in :mod:`unit.csr`.
"""
import os
from typing import List, Optional

import torch
# pylint: disable=import-error,no-name-in-module
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

from .config_params import GTX_NEST_NUM, GTX_SPU_NUM, DEVICE
from .dispatch import build_custom0_table, build_custom1_table, resolve_for_context
from .tloop_buffer import (
    BUFFERABLE_MNEMONICS as _TLOOP_BUFFERABLE,
    TRANSPARENT_MNEMONICS as _TLOOP_TRANSPARENT,
    TLoopEntry as _TLoopEntry,
    flush as _tloop_flush,
)
from .unit.ins.encoding import (
    GTX_ISS_F7_OPSET,
    GSPR_GTX_OPERAND3 as _GSPR_OP3,
    GSPR_GTX_OPERAND5 as _GSPR_OP5,
)
from .fsm import NpuState, run_pipeline
from .unit.context import INITIAL_CONTEXT, NpuContext
from .unit.context.warp_state import WarpState
from .unit.csr import GSPR, LSPR, NSPR
from .unit.memory import GtxMemory
from .unit.register_file import RegisterFile

from . import _registry  # noqa: F401  -- imported for completeness
from .unit.ins import ops as _ops  # noqa: F401  -- triggers @handler decorators


# =========================================================================
# Vendor reset defaults — sourced from gtx_npu_core.cc:80-109.
# =========================================================================

# CORE-02: initial stack pointer (firmware ABI).
_SP_INIT_VALUE: int = 0x80100000

# RISC-V architectural CSRs touched at reset (NOT GTX SPRs).
_CSR_MSTATUS: int = 0x300
_MSTATUS_FS_MASK: int = 0x6000   # mstatus.FS [14:13]
_MSTATUS_FS_INITIAL: int = 0x2000   # FS = 01 (Initial)

_GSPR_RESET_DEFAULTS = {
    0x010: 0,   # STACK_INFO
    0x011: 0,   # STACK_SAVE
}


# ============================================================================
# GTX Instruction Encoding (in Spike RoCC custom0 opcode 0x0b)
# 
# RoCC rocc_insn_t bitfield: [funct7|rs2|rs1|xd|xs1|xs2|rd|opcode]
#   opcode[6:0]  = 0x0b (custom-0)
#   rd[11:7]     = destination register
#   xs2[12]      = 1 if rs2 used  ─┐
#   xs1[13]      = 1 if rs1 used  ─┤ these control register value passing!
#   xd[14]       = 1 if rd written ┘
#   rs1[19:15]   = source register 1
#   rs2[24:20]   = source register 2
#   funct[31:25] = function code (funct7) — used for sub-command dispatch

@isa.register("gtx")
class GtxNpu(isa.ROCC):
    """GTX NPU functional model — RoCC ``custom0``/``custom1`` dispatch."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.mem = GtxMemory()
        self.warp = WarpState()
        # Deferred S-loop L2->DDR store queue.
        self.deferred_ddr_stores: list = []
        
        # Layered SPR storage — Tensor-backed via RegisterFile.
        #   gspr: single instance               (shape: [1024])
        #   nspr: per-NEST                       (shape: [NEST, 1024])
        #   lspr: per-(NEST, SPU)                (shape: [NEST, SPU, 1024])
        self.gspr = RegisterFile(GSPR, shape=(1024,), device=DEVICE)
        self.nspr = RegisterFile(NSPR, shape=(GTX_NEST_NUM, 1024), device=DEVICE)
        self.lspr = RegisterFile(LSPR, shape=(GTX_NEST_NUM, GTX_SPU_NUM, 1024), device=DEVICE)

        self._mxe_accum: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.float32,
            device=DEVICE)
        self._credit_ld: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32,
            device=DEVICE)
        self._credit_st: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32,
            device=DEVICE)

        self._disasm_entries: List[disasm_insn_t] = []

        # Apply initial vendor defaults
        self._apply_vendor_defaults()

        self._custom0 = build_custom0_table(self)
        self._custom1 = build_custom1_table(self)
        
        # Initial dispatch cache
        self.refresh_dispatch_cache(INITIAL_CONTEXT)

        # FSM scaffold
        self._state: NpuState = NpuState.IDLE
        self._ctx: dict = {}
        # T-loop instruction buffer.
        self._tloop_buf: list | None = None
        # NPU execution context.
        self._context: NpuContext = INITIAL_CONTEXT

        self.mem.load_via_env()

        import atexit as _atexit
        _atexit.register(self.mem.dump_via_env)

        if os.environ.get("GTX_PROFILE"):
            import cProfile, atexit, pstats, sys
            self._profiler = cProfile.Profile()
            self._profiler.enable()

            def _dump_profile() -> None:
                try:
                    self._profiler.disable()
                except Exception:
                    pass
                ps = pstats.Stats(self._profiler).sort_stats("cumulative")
                print("\n========== GTX_PROFILE (cumulative, top 40) ==========",
                      file=sys.stderr)
                ps.stream = sys.stderr
                ps.print_stats(40)
                print("========== GTX_PROFILE (tottime, top 30) ==========",
                      file=sys.stderr)
                ps2 = pstats.Stats(self._profiler).sort_stats("tottime")
                ps2.stream = sys.stderr
                ps2.print_stats(30)

            atexit.register(_dump_profile)

    def _apply_vendor_defaults(self) -> None:
        """Apply vendor-specific reset values (Broadcasting)."""
        # GSPR defaults via raw map
        self.gspr.reset(_GSPR_RESET_DEFAULTS)
        
        # NSPR defaults (Broadcasting across all nests)
        self.nspr.reset()
        self.nspr.THREAD_MASK.mask = 0xFFFF
        self.nspr.SHARED_MASK.mask = 0
        self.nspr.DATA_FORMAT.data = 1 # FP16
        self.nspr.OP_MODE.double_buffer = 0
        self.nspr.OP_MODE.load_credit_en = 0
        self.nspr.OP_MODE.store_credit_en = 0
        
        # LSPR defaults (Broadcasting across all nests and SPUs)
        self.lspr.reset()
        # All SPM addresses default to 0 (already zeroed by reset)

    def refresh_dispatch_cache(self, context: Optional[NpuContext] = None) -> None:
        """Re-flatten the dispatch tables for the given context."""
        ctx = context or self._context
        self._custom0_resolved, self._custom1_resolved = resolve_for_context(
            self._custom0, self._custom1, ctx
        )

    # ------------------------------------------------------------------
    # ROCC virtual methods
    # ------------------------------------------------------------------
    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]:
        if not self._disasm_entries:
            self._disasm_entries = list(_registry.collect_disasms())
        return list(self._disasm_entries)

    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []   # SPRs are NOT CSRs (project convention)

    def reset(self, proc: processor_t) -> None:
        # CORE-02: stack-pointer init.
        proc.state.XPR.write(2, _SP_INIT_VALUE)
        # FPU enable
        try:
            mstatus = proc.get_csr(_CSR_MSTATUS)
            mstatus = (mstatus & ~_MSTATUS_FS_MASK) | _MSTATUS_FS_INITIAL
            proc.put_csr(_CSR_MSTATUS, mstatus)
        except Exception:
            pass

        # Per-(NEST, SPU) state reset
        self._mxe_accum.fill_(0.0)
        self._credit_ld.fill_(0)
        self._credit_st.fill_(0)

        # Memory hierarchy reset
        self.mem.reset_scratchpads()

        # SPR reset with defaults
        self._apply_vendor_defaults()

        # Warp and FSM reset
        self.deferred_ddr_stores.clear()
        self.warp.reset()
        self._state = NpuState.IDLE
        self._ctx = {}
        self._tloop_buf = None
        
        # Context reset
        self._context = INITIAL_CONTEXT
        self.refresh_dispatch_cache(INITIAL_CONTEXT)


    # ------------------------------------------------------------------
    # RoCC entry points — both kinds drive the same one-instruction FSM
    # ------------------------------------------------------------------
    def custom0(self, proc, insn, xs1, xs2) -> int:
        """RoCC ``custom0`` entry. Hot-path optimized for T-loop."""
        buf = self._tloop_buf
        if buf is not None and self.warp.is_tloop:
            funct7 = insn.funct

            # Hoist hot tensor and state to locals
            gspr_tensor = self.gspr.tensor
            state = proc.state
            xpr = state.XPR

            # OPSET fast-path
            if funct7 == GTX_ISS_F7_OPSET:
                if (int(xpr[insn.rs1]) & 1) == 0:
                    gspr_tensor[_GSPR_OP3] = int(xpr[insn.rs2])
                else:
                    gspr_tensor[_GSPR_OP5] = int(xpr[insn.rs2])
                return 0

            xd = insn.xd
            xs1_bit = insn.xs1
            xs2_bit = insn.xs2
            funct3 = (xd << 2) | (xs1_bit << 1) | xs2_bit

            inner = self._custom0_resolved.get(funct7)
            if inner is not None:
                handler = inner.get(None)
                if handler is None:
                    handler = inner.get(funct3)
            else:
                handler = None

            if handler is not None:
                mnemonic = handler.gtx_mnemonic
                if mnemonic in _TLOOP_BUFFERABLE:
                    # Inline snapshot
                    buf.append(_TLoopEntry(
                        handler, mnemonic,
                        int(xpr[insn.rs1]),
                        int(xpr[insn.rs2]),
                        int(gspr_tensor[_GSPR_OP3]),
                        int(gspr_tensor[_GSPR_OP5]),
                        funct7, xd, xs1_bit, xs2_bit, insn.rd,
                    ))
                    # Mirror WRITEBACK's OPSET-aware clear
                    gspr_tensor[_GSPR_OP3] = 0
                    gspr_tensor[_GSPR_OP5] = 0
                    return 0
                
                if buf and mnemonic not in _TLOOP_TRANSPARENT:
                    _tloop_flush(self)

        return run_pipeline(self, "custom0", proc, insn, xs1, xs2)

    def custom1(self, proc, insn, xs1, xs2) -> int:
        """RoCC ``custom1`` entry. Drives the one-instruction FSM."""
        return run_pipeline(self, "custom1", proc, insn, xs1, xs2)

    # ------------------------------------------------------------------
    # Deferred-store flush — wired from end_p / credit_st_chk handlers.
    # ------------------------------------------------------------------
    def flush_deferred_ddr_stores(self) -> None:
        """Drain the S-loop deferred L2→DDR store queue."""
        if not self.deferred_ddr_stores:
            return
        from .config_params import GTX_L2_SIZE_BYTES
        for req in self.deferred_ddr_stores:
            l2_src = self.mem.l2_byte(req.nest).cpu()
            max_off = req.ddr_off + (req.height - 1) * req.ddr_stride + req.length
            self.mem.ensure_ddr(max_off)
            cap = self.mem.ddr.capacity()
            for row in range(req.height):
                ddr_off = req.ddr_off + row * req.ddr_stride
                l2_off = (req.l2_off + row * req.l2_stride) % GTX_L2_SIZE_BYTES
                copy_len = min(req.length, cap - ddr_off, GTX_L2_SIZE_BYTES - l2_off)
                if copy_len > 0:
                    self.mem.ddr.write(
                        ddr_off, l2_src[l2_off:l2_off + copy_len]
                    )
        self.deferred_ddr_stores.clear()
