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
"""GtxNpu -- riscv.isa.ROCC subclass, registered as 'gtx' (CORE-01, D-14)."""
from typing import List
import torch
# pylint: disable=import-error,no-name-in-module
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

from .memory import GtxMemory
from .warp_state import WarpState
from .params import (GTX_NEST_NUM, GTX_SPU_NUM)
from . import _registry  # noqa: F401  -- imported for completeness
from . import ops as _ops  # noqa: F401  -- triggers @handler decorators
from .dispatch import build_custom0_table, build_custom1_table

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
        # P3 D-05: deferred S-loop L2->DDR store queue. Pushed by ops/dma.py
        # @handler firmware_dma_store (S-loop branch), flushed by ops/control.py
        # end_p (when !wsplit_seen) or ops/dma.py credit_st_chk (when is_sloop).
        self.deferred_ddr_stores: list = []
        # Layered SPR storage (D-11 + research §390-396 strong recommendation):
        #   gspr: flat dict (single instance)
        #   nspr: list of dict, length GTX_NEST_NUM
        #   lspr: list of list of dict, [NEST][SPU]
        self.gspr: dict = {}
        self.nspr: list = [dict() for _ in range(GTX_NEST_NUM)]
        self.lspr: list = [
            [dict() for _ in range(GTX_SPU_NUM)] for _ in range(GTX_NEST_NUM)
        ]
        # mxe_accum: 2D scalar accumulator per (NEST, SPU). FP32. CORRECTED
        # from CONTEXT.md D-06 (4D was wrong -- see C++ gtx_npu.h:1254).
        self._mxe_accum: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.float32
        )
        # P8 (2026-05-11) — per-NEST/per-SPU credit counters (vendor parity).
        # gtx_npu.h:624-625 declares these on the nest struct; pyspike collapses
        # to a single [NEST][SPU] 2D array per type. Functional model NOPs the
        # check variants ("always true — DMA is instantaneous") so the counters
        # are stat-only; kept for vendor 1:1 diff invariance and to surface any
        # future check-path coupling without re-architecting.
        self._credit_ld: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32
        )
        self._credit_st: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM), dtype=torch.int32
        )
        # Disasm cache (plan 04 fills via _registry.collect_disasms)
        self._disasm_entries: List[disasm_insn_t] = []
        # Dispatch tables (plan 02-03 fill _registry.HANDLERS)
        self._custom0 = build_custom0_table(self)
        self._custom1 = build_custom1_table(self)
        # P6 follow-up (vendor gtx_npu_core.cc:120): GTX_DDR_INIT pre-stage —
        # symmetric pair to atexit dump. Must run BEFORE _LAST_NPU registration
        # so atexit hook never sees a half-initialized DDR. P5 Plan 02 added
        # only the dump half; the init half was missing until this commit.
        from .ddr import _init_ddr_from_env
        _init_ddr_from_env(self)

        # P6 D-04/D-05: register self as the latest GtxNpu (vendor gtx_npu_core.cc:59
        # `g_gtx_instance = this;` direct port). Last-instance-wins. Single-hart
        # invariant; tests use subprocess isolation for per-instance lookup.
        global _LAST_NPU
        _LAST_NPU = self

    # NOTE: do NOT override get_instructions(). The C++ rocc_t base
    # (vendor/spike/riscv/rocc.cc:34-42) returns the c0/c1/c2/c3 dispatch
    # entries for opcodes 0x0b/0x2b/0x5b/0x7b. Returning [] from Python
    # blanks that out and makes the decoder reject custom1 as illegal —
    # see tests/test_extension.py::MyDummyROCC for the canonical pattern
    # (no override, n_insn==4).

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

    def custom0(self, proc, insn, xs1, xs2) -> int:
        """2-level dispatch: funct7 -> {funct3-or-None: handler}.

        Tries the no-sub-decomposition entry first (sentinel inner key None for
        P2 backwards-compat); if absent, synthesizes funct3 from RoCC R-type
        flags and tries the integer-keyed sub-table (P3+ mask_funct3=True path).
        Unmapped routes return 0 (silent NOP, P5/P6 may upgrade to illegal).

        P8 (2026-05-11) — port of vendor outer wrapper `gtx_npu_t::custom0`
        (~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:1042-1058):
        OPSET (funct7=0x4A) is the only instruction that LEAVES OPERAND3
        (0x003) and OPERAND4 (0x005) staging slots populated for the next
        instruction to consume. Every OTHER instruction must CLEAR both
        slots AFTER its handler returns, so stale stage values do not leak
        into subsequent unrelated instructions. Without this clear, vendor
        single-tile sweep ops (NEG/EXP/EXPM1/CUMSUM) inherited stale OPSET
        state from prior multi-tile ops and dumped zero / garbage output.
        """
        funct7 = insn.funct
        sub_table = self._custom0.get(funct7)
        result = 0
        if sub_table is not None:
            # P2 backwards-compat: try the non-decomposed entry first
            handler = sub_table.get(None)
            if handler is None:
                funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
                handler = sub_table.get(funct3)
            if handler is not None:
                result = handler(proc, insn, xs1, xs2)
        # Clear OPSET staging slots after non-OPSET dispatch (vendor parity).
        # OPSET funct7 = 0x4A; literal kept here to avoid import cycle.
        if funct7 != 0x4A:
            self.gspr[0x003] = 0   # GSPR_GTX_OPERAND3
            self.gspr[0x005] = 0   # GSPR_GTX_OPERAND4 (vendor literal)
        return result

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
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        handler = self._custom1.get(funct3)
        if handler is None:
            return 0
        return handler(proc, insn, xs1, xs2)
