"""GtxNpu -- ``riscv.isa.ROCC`` subclass registered as ``"gtx"``.

FSM-driven dispatch (see :mod:`fsm`) with NPU context awareness
(C1/C2/C3/C4 — see :mod:`unit.context`). SPR storage uses
:class:`~unit.register_file.RegisterFile`, so addresses are indexed by
typed name (``self.gspr["GSPR_GTX_OPERAND3"]``) wherever the source
register is declared in :mod:`unit.csr`.
"""
import os
from typing import List

import torch
# pylint: disable=import-error,no-name-in-module
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

from .config_params import GTX_NEST_NUM, GTX_SPU_NUM, DEVICE
from .dispatch import build_custom0_table, build_custom1_table
from .fsm import NpuState, run_pipeline
from .unit.context import INITIAL_CONTEXT, NpuContext
from .unit.context.warp_state import WarpState
from .unit.csr import GSPR as _GSPR_DEFS, LSPR as _LSPR_DEFS, NSPR as _NSPR_DEFS
from .unit.ins.encoding import (
    GSPR_GTX_OPERAND0, GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2,
    GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND4, GSPR_GTX_OPERAND5,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
    NSPR_THREAD_MASK, NSPR_SHARED_MASK, NSPR_TYPE, NSPR_OP_MODE,
    NSPR_CLEAR, NSPR_SDLE_STATUS, NSPR_SMU_DEBUG, NSPR_CREDIT_COUNT,
)
from .unit.memory import GtxMemory
from .unit.register_file import RegisterFile

from . import _registry  # noqa: F401  -- imported for completeness
from .unit.ins import ops as _ops  # noqa: F401  -- triggers @handler decorators


# =========================================================================
# Vendor reset defaults — sourced from gtx_npu_core.cc:80-109.
# Address keys come from encoding constants so callers can grep them.
# =========================================================================

# CORE-02: initial stack pointer (firmware ABI).
_SP_INIT_VALUE: int = 0x80100000

# RISC-V architectural CSRs touched at reset (NOT GTX SPRs).
_CSR_MSTATUS: int = 0x300
_MSTATUS_FS_MASK: int = 0x6000   # mstatus.FS [14:13]
_MSTATUS_FS_INITIAL: int = 0x2000   # FS = 01 (Initial)

_GSPR_RESET_DEFAULTS = {
    GSPR_GTX_OPERAND0: 0,
    GSPR_GTX_OPERAND1: 0,
    GSPR_GTX_OPERAND2: 0,
    GSPR_GTX_OPERAND3: 0,
    GSPR_GTX_OPERAND4: 0,
    0x010: 0,   # STACK_INFO
    0x011: 0,   # STACK_SAVE
}

_NSPR_RESET_DEFAULTS = {
    NSPR_THREAD_MASK: 0xFFFF,   # all SPUs active
    NSPR_SHARED_MASK: 0,
    NSPR_TYPE: 1,               # FP16
    NSPR_OP_MODE: 0,
    NSPR_CLEAR: 0,
    NSPR_SDLE_STATUS: 0,
    NSPR_SMU_DEBUG: 0,
    NSPR_CREDIT_COUNT: 0,
}

_LSPR_RESET_DEFAULTS = {
    LSPR_SPM_ADDRA: 0,
    LSPR_SPM_ADDRB: 0,
    LSPR_SPM_ADDRC: 0,
    LSPR_SPM_ADDRR: 0,
}

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
        # Deferred S-loop L2->DDR store queue. Pushed by
        # unit/context/dma.py @handler firmware_dma_store (S-loop branch);
        # flushed by unit/context/control.py end_p (when !wsplit_seen) or
        # unit/context/dma.py credit_st_chk (when is_sloop).
        self.deferred_ddr_stores: list = []
        # Layered SPR storage — name-indexed via RegisterFile.
        #   gspr: single instance               (RegisterFile)
        #   nspr: per-NEST                       (list[RegisterFile])
        #   lspr: per-(NEST, SPU)                (list[list[RegisterFile]])
        self.gspr: RegisterFile = RegisterFile(_GSPR_DEFS)
        self.nspr: List[RegisterFile] = [
            RegisterFile(_NSPR_DEFS) for _ in range(GTX_NEST_NUM)
        ]
        self.lspr: List[List[RegisterFile]] = [
            [RegisterFile(_LSPR_DEFS) for _ in range(GTX_SPU_NUM)]
            for _ in range(GTX_NEST_NUM)
        ]
        #!TODO: 굳이 없어도 될 것 같은데? 없자피 l1, l0에 저장하는건데.
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

        self._custom0 = build_custom0_table(self)
        self._custom1 = build_custom1_table(self)

        # FSM scaffold — `_state` is the current pipeline state; `_ctx`
        # holds transient per-instruction values produced/consumed by
        # the state functions in decode/dispatch_state/execute/writeback.
        self._state: NpuState = NpuState.IDLE
        self._ctx: dict = {}
        # NPU execution context (persistent across instructions; warp
        # markers mutate via apply_transition in WRITEBACK).
        self._context: NpuContext = INITIAL_CONTEXT

        self.mem.load_via_env()

        # WJOIN no longer dumps DDR (control.wjoin_with_exit) — register
        # the dump as an atexit hook so multi-tile firmware pays the
        # cost once at process teardown instead of per-tile.
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
        # `super().reset()` is a no-op in vendor/spike/riscv/extension.h:18
        # and rejects MockProcessor under strict pybind11 types — skip it.
        # CORE-02: stack-pointer init.
        proc.state.XPR.write(2, _SP_INIT_VALUE)
        # FPU enable (forward-compat for P4 GEMM, mstatus.FS = Initial).
        try:
            mstatus = proc.get_csr(_CSR_MSTATUS)
            mstatus = (mstatus & ~_MSTATUS_FS_MASK) | _MSTATUS_FS_INITIAL
            proc.put_csr(_CSR_MSTATUS, mstatus)
        except Exception:
            pass

        # Per-SPU MXE accumulator + per-(NEST, SPU) credit counters
        # (vendor parity, P8 fix).
        self._mxe_accum.fill_(0.0)
        self._credit_ld.fill_(0)
        self._credit_st.fill_(0)

        # Scratchpad zero-init — DDR is preserved (firmware-loaded data
        # must survive a hart reset). reset_scratchpads is the OOP entry
        # point that recurses through nests→(L2, SPUs→(L0, L1)).clear().
        self.mem.reset_scratchpads()

        # SPR zero-init + vendor defaults (gtx_npu_core.cc:80-109 verbatim,
        # routed through RegisterFile.reset for name-based readability).
        self.gspr.reset(_GSPR_RESET_DEFAULTS)
        for n in range(GTX_NEST_NUM):
            self.nspr[n].reset(_NSPR_RESET_DEFAULTS)
            for s in range(GTX_SPU_NUM):
                self.lspr[n][s].reset(_LSPR_RESET_DEFAULTS)

        # P3 D-05: clear deferred queue on reset (`wsplit_seen` NOT cleared
        # — see WarpState.reset() field comment + 03-RESEARCH Pitfall 7).
        self.deferred_ddr_stores.clear()
        # Warp state reset.
        self.warp.reset()
        # FSM scaffold reset.
        self._state = NpuState.IDLE
        self._ctx = {}
        # Context reset — back to C1 (plan outside).
        self._context = INITIAL_CONTEXT

    # ------------------------------------------------------------------
    # RoCC entry points — both kinds drive the same one-instruction FSM
    # ------------------------------------------------------------------
    def custom0(self, proc, insn, xs1, xs2) -> int:
        """RoCC ``custom0`` entry. Drives the one-instruction FSM.

        Dispatch semantics (preserved verbatim from pre-FSM port; see
        vendor ``gtx_npu_custom0.cc:1042-1058``):

        - 3-level lookup: ``funct7 → context → {funct3-or-None: handler}``
        - ``sub_table[None]`` tried first (P2 back-compat, no funct3 decomp)
        - fallback: ``funct3`` synthesised from RoCC R-type flags
        - unmapped routes return 0 (silent NOP)
        - OPSET (``funct7 = GTX_ISS_F7_OPSET``) leaves OPERAND3/5 staging
          populated; every other instruction clears them post-dispatch.
        """
        return run_pipeline(self, "custom0", proc, insn, xs1, xs2)

    def custom1(self, proc, insn, xs1, xs2) -> int:
        """RoCC ``custom1`` entry. Drives the one-instruction FSM.

        Dispatch semantics: single-level ``funct3`` lookup; miss returns
        0. No OPSET post-clear (custom0-only invariant).
        """
        return run_pipeline(self, "custom1", proc, insn, xs1, xs2)

    # ------------------------------------------------------------------
    # Deferred-store flush — wired from end_p / credit_st_chk handlers.
    # ------------------------------------------------------------------
    def flush_deferred_ddr_stores(self) -> None:
        """Drain the S-loop deferred L2→DDR store queue.

        Direct port of ``gtx_npu_dma.cc:415-435``. Triggers (see
        ``unit/context/control.py:_do_endp`` and
        ``unit/context/dma.py:_credit_*_chk``):

        - end_p when not ``wsplit_seen`` (simple firmware)
        - credit_{ld,st}_chk when ``is_sloop`` (plan-style firmware)
        """
        if not self.deferred_ddr_stores:
            return
        from .config_params import GTX_L2_SIZE_BYTES
        # atexit-safe: import torch locally so its module ref stays alive
        # for the duration of this call (avoids _C dtor races during
        # interpreter shutdown).
        import torch  # noqa: F401  -- pin module reference
        for req in self.deferred_ddr_stores:
            # Hierarchy contract: L2 is on DEVICE (CUDA), DDR is on CPU.
            # Per-request single H→D snapshot of L2; all row writes then
            # stay CPU↔CPU (no per-row sync). Single ``ensure_ddr`` grow
            # up-front so capacity is stable across the inner loop.
            l2_src = self.mem.l2_byte(req.nest).cpu()
            max_off = req.ddr_off + (req.height - 1) * req.ddr_stride + req.length
            self.mem.ensure_ddr(max_off)
            cap = self.mem.ddr.capacity()
            for row in range(req.height):
                ddr_off = req.ddr_off + row * req.ddr_stride
                l2_off = (req.l2_off + row * req.l2_stride) % GTX_L2_SIZE_BYTES
                copy_len = req.length
                copy_len = min(copy_len, cap - ddr_off)
                copy_len = min(copy_len, GTX_L2_SIZE_BYTES - l2_off)
                if copy_len > 0:
                    self.mem.ddr.write(
                        ddr_off, l2_src[l2_off:l2_off + copy_len]
                    )
        self.deferred_ddr_stores.clear()
