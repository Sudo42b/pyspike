#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""GtxNpu -- riscv.isa.ROCC subclass, registered as 'gtx' (CORE-01, D-14).

FSM-driven dispatch with NPU context awareness — see ORDER.md.
"""
import enum
from typing import List
import torch
# pylint: disable=import-error,no-name-in-module
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

from .memory import GtxMemory
from .context.warp_state import WarpState
from .params import (GTX_NEST_NUM, GTX_SPU_NUM)
from .context import (
    NpuContext, INITIAL_CONTEXT
)

from . import _registry  # noqa: F401  -- imported for completeness
from .ins import ops as _ops  # noqa: F401  -- triggers @handler decorators
from .dispatch import build_custom0_table, build_custom1_table

# FSM state transition functions — one module per state, pure functions.
from .decode import state_decode
from .dispatch_state import state_dispatch
from .execute import state_execute
from .writeback import state_writeback


class _NpuState(enum.Enum):
    """One-instruction FSM states for GtxNpu.custom0/custom1 dispatch.

    Pipeline (per pyspike functional model — not cycle-accurate):

        IDLE → DECODE → DISPATCH → EXECUTE → WRITEBACK → IDLE

    C1/C2/C3/C4 are NPU *contexts* (persistent across instructions, see
    npu_context.py), not sub-states of EXECUTE. Context validity and
    transitions are handled in DISPATCH (context-aware handler lookup)
    and WRITEBACK (warp-marker → apply_transition).
    """
    IDLE = enum.auto()
    DECODE = enum.auto()
    DISPATCH = enum.auto()
    EXECUTE = enum.auto()
    WRITEBACK = enum.auto()


# FSM state → transition-function table. Each function takes the npu
# instance and returns the next _NpuState. IDLE has no transition (loop
# exit sentinel) and is therefore absent from this table.
_STATE_DISPATCH_TABLE = {
    _NpuState.DECODE:    state_decode,
    _NpuState.DISPATCH:  state_dispatch,
    _NpuState.EXECUTE:   state_execute,
    _NpuState.WRITEBACK: state_writeback,
}

# P6 D-04/D-05: single-global NPU instance pointer for atexit dump hook.
# Direct port of vendor gtx_npu_core.cc:59
#   static gtx_npu_t *g_gtx_instance = nullptr;
# The atexit handler in ddr.py reads this at interpreter shutdown.
# Single-global is correct for v1 single-hart scope; v2 multi-hart may
# upgrade to weakref.WeakValueDictionary.
_LAST_NPU = None  # type: ignore[var-annotated]


@isa.register("gtx")
class GtxNpu(isa.ROCC):
    """GTX NPU functional model -- Phase 2 dispatch shell (CORE-01)."""

    def __init__(self):
        super().__init__()
        self.mem = GtxMemory()
        self.warp = WarpState()
        # deferred S-loop L2->DDR store queue. Pushed by ops/dma.py
        # @handler firmware_dma_store (S-loop branch), flushed by ops/control.py
        # end_p (when !wsplit_seen) or ops/dma.py credit_st_chk (when is_sloop).
        self.deferred_ddr_stores: list = []
        # Layered SPR storage
        #   gspr: flat dict (single instance)
        #   nspr: list of dict, length GTX_NEST_NUM
        #   lspr: list of list of dict, [NEST][SPU]
        self.gspr: dict = {}
        self.nspr: list = [dict() for _ in range(GTX_NEST_NUM)]
        self.lspr: list = [
            [dict() for _ in range(GTX_SPU_NUM)] for _ in range(GTX_NEST_NUM)
        ]

        
        self._mxe_accum: torch.Tensor = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.float32)

        self._credit_ld: torch.Tensor = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32)
        self._credit_st: torch.Tensor = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32)
        
        self._disasm_entries: List[disasm_insn_t] = []
        
        self._custom0 = build_custom0_table(self)
        self._custom1 = build_custom1_table(self)

        # FSM scaffold — _state is the current pipeline state, _ctx holds
        # transient per-instruction values produced/consumed by state methods.
        self._state: _NpuState = _NpuState.IDLE
        self._ctx: dict = {}
        # NPU execution context (persistent across instructions; warp markers
        # mutate via apply_transition in WRITEBACK). See npu_context.py.
        self._context: NpuContext = INITIAL_CONTEXT

        from .ddr import _init_ddr_from_env
        _init_ddr_from_env(self)

    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]:
        # Plan 04 populates self._disasm_entries on demand
        if not self._disasm_entries:
            self._disasm_entries = list(_registry.collect_disasms())
        return list(self._disasm_entries)

    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []   # SPRs are NOT CSRs (project convention)

    def reset(self, proc: processor_t) -> None:
        # super().reset() is a no-op in vendor/spike/riscv/extension.h:18
        # and rejects MockProcessor under strict pybind11 types — skip it.
        # CORE-02: sp init
        proc.state.XPR.write(2, 0x80100000)
        # FPU enable (forward-compat for P4 GEMM, mstatus.FS = 01 Initial)
        try:
            mstatus = proc.get_csr(0x300)
            mstatus = (mstatus & ~0x6000) | 0x2000
            proc.put_csr(0x300, mstatus)
        except Exception:
            pass
        # mxe_accum zero-init
        self._mxe_accum.fill_(0.0)
        # P8: credit counter zero-init (vendor parity)
        self._credit_ld.fill_(0)
        self._credit_st.fill_(0)
        # Memory zero-init
        self.mem._l0_bytes.fill_(0)
        self.mem._l1_bytes.fill_(0)
        self.mem._l2_bytes.fill_(0)
        # SPR zero-init + defaults (gtx_npu_core.cc:80-109 verbatim)
        self.gspr.clear()
        for addr in (0x000, 0x001, 0x002, 0x003, 0x004, 0x010, 0x011):
            self.gspr[addr] = 0
        for n in range(GTX_NEST_NUM):
            self.nspr[n].clear()
            self.nspr[n][0x400] = 0xFFFF   # NSPR_THREAD_MASK = all SPUs active
            self.nspr[n][0x401] = 0
            self.nspr[n][0x402] = 1        # NSPR_TYPE = FP16 default
            self.nspr[n][0x403] = 0
            self.nspr[n][0x700] = 0
            self.nspr[n][0x780] = 0
            self.nspr[n][0x781] = 0
            self.nspr[n][0x782] = 0
            for s in range(GTX_SPU_NUM):
                self.lspr[n][s].clear()
                self.lspr[n][s][0x900] = 0
                self.lspr[n][s][0x901] = 0
                self.lspr[n][s][0x902] = 0
                self.lspr[n][s][0x903] = 0
        # P3 D-05: clear deferred queue on reset (wsplit_seen NOT cleared --
        # see WarpState.reset() field comment + 03-RESEARCH Pitfall 7)
        self.deferred_ddr_stores.clear()
        # Warp state reset
        self.warp.reset()
        # FSM scaffold reset
        self._state = _NpuState.IDLE
        self._ctx = {}
        # Context reset — back to C1 (plan outside).
        self._context = INITIAL_CONTEXT

    def custom0(self, proc, insn, xs1, xs2) -> int:
        """RoCC custom0 entry. Drives the one-instruction FSM pipeline.

        Dispatch semantics (preserved verbatim from pre-FSM port; see
        vendor gtx_npu_custom0.cc:1042-1058):
          - 2-level lookup: funct7 -> {funct3-or-None: handler}
          - sub_table[None] tried first (P2 back-compat, no funct3 decomp)
          - fallback: funct3 synthesized from RoCC R-type flags
          - unmapped routes return 0 (silent NOP)
          - OPSET (funct7=0x4A) leaves OPERAND3/4 staging populated;
            every other instruction clears them post-dispatch.
        """
        return self._run_pipeline("custom0", proc, insn, xs1, xs2)

    def flush_deferred_ddr_stores(self) -> None:
        """Direct port of gtx_npu_dma.cc:415-435.

        Empties self.deferred_ddr_stores by performing each requested L2->DDR
        per-row copy. Plan 05 wires the triggers (end_p when !wsplit_seen +
        credit_st_chk when is_sloop); this plan only registers the API.
        """
        if not self.deferred_ddr_stores:
            return
        # Lazy-import to avoid circular ddr.py <- dma_engine.py <- this
        from .ddr import ensure_ddr
        from .params import GTX_L2_SIZE_BYTES
        # atexit-safe: import torch locally so its module ref stays alive
        # for the duration of this call (avoids _C dtor races during shutdown).
        import torch  # noqa: F401  -- pin module reference
        for req in self.deferred_ddr_stores:
            l2_src = self.mem.l2_byte(req.nest).detach().cpu().contiguous()
            for row in range(req.height):
                ddr_off = req.ddr_off + row * req.ddr_stride
                l2_off = (req.l2_off + row * req.l2_stride) % GTX_L2_SIZE_BYTES
                copy_len = req.length
                ensure_ddr(self.mem, ddr_off + copy_len)
                ddr_buf = self.mem._ddr_bytes
                copy_len = min(copy_len, ddr_buf.numel() - ddr_off)
                copy_len = min(copy_len, GTX_L2_SIZE_BYTES - l2_off)
                if copy_len > 0:
                    ddr_buf[ddr_off:ddr_off + copy_len] = (
                        l2_src[l2_off:l2_off + copy_len]
                    )
        self.deferred_ddr_stores.clear()

    def custom1(self, proc, insn, xs1, xs2) -> int:
        """RoCC custom1 entry. Drives the one-instruction FSM pipeline.

        Dispatch semantics: single-level funct3 lookup; miss returns 0.
        No OPSET post-clear (custom0-only invariant).
        """
        return self._run_pipeline("custom1", proc, insn, xs1, xs2)

    # ------------------------------------------------------------------
    # FSM driver -- class-based pipeline (states are methods)
    # ------------------------------------------------------------------
    def _run_pipeline(self, kind: str, proc, insn, xs1, xs2) -> int:
        """Single-instruction FSM driver.

        Per pyspike functional-model contract the pipeline completes within
        one Python call (no cycle-accurate latency). The FSM exists for
        structural clarity (state-by-state debugging, future cycle-accurate
        hook insertion); behavior is identical to the pre-FSM 2-level
        dispatch.
        """
        self._ctx = {
            "kind": kind,
            "proc": proc,
            "insn": insn,
            "xs1": xs1,
            "xs2": xs2,
            "rd": 0,
        }
        self._state = _NpuState.DECODE
        while self._state is not _NpuState.IDLE:
            self._step()
        return self._ctx["rd"]

    def _step(self) -> None:
        """Run one state transition.

        Looks up the current state's transition function in
        `_STATE_DISPATCH_TABLE` and applies its return value to
        `self._state`. Each state function lives in its own module:

          DECODE     -> decode.state_decode
          DISPATCH   -> dispatch_state.state_dispatch
          EXECUTE    -> execute.state_execute
          WRITEBACK  -> writeback.state_writeback

        IDLE is not in the table — `_run_pipeline` exits the loop before
        reaching IDLE inside `_step`.
        """
        fn = _STATE_DISPATCH_TABLE.get(self._state)
        if fn is None:
            raise RuntimeError(f"GtxNpu FSM: unreachable state {self._state!r}")
        self._state = fn(self)
