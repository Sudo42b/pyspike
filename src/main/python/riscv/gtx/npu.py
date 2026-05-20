import os
import sys
from typing import Any, List
import torch

from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import processor_t

from .config_params import (
    _GSPR_RESET_DEFAULTS, _SP_INIT_VALUE, _CSR_MSTATUS, _MSTATUS_FS_MASK, 
    _MSTATUS_FS_INITIAL, NEST_NUM, SPU_NUM, DEVICE
)
from .csr import GSPR, LSPR, NSPR
from .memory import GtxMemory
from .csr.register_file import RegisterFile
from .context.exec_st import CXT
from .context import WarpState
from .context.disasm import Custom0_Insn, Custom1_Insn, inst_register

# Register handlers — imported for side-effect (decorators populate
# inst_register). Done AFTER inst_register is defined to avoid a circular
# import through disasm. Only ported modules are listed; the rest land as
# they are migrated to the new API.
from .context.custom1 import control as _control      # noqa: F401,E402
from .context.custom0.DL import spr as _spr, dma as _dma, credit as _credit  # noqa: F401,E402
from .context.custom0.MX import (                     # noqa: F401,E402
    matmul as _matmul, vector as _vector, pooling as _pooling,
    act as _act, softmax as _softmax, type_cvt as _type_cvt,
    mem_op as _mem_op, conv as _conv,
)
from .context.custom0.MC import ucode as _ucode       # noqa: F401,E402
from .context.custom0.SN import sync as _sync         # noqa: F401,E402

@isa.register("gtx")
class GtxNpu(isa.ROCC):
    """GTX NPU functional model — RoCC ``custom0``/``custom1`` dispatch."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.mem = GtxMemory()
        
        # Deferred S-loop L2->DDR store queue.
        self.deferred_ddr_stores: list = []
        # 현재 실행 context (C1/C2/C3/C4) — affects memory access behavior and dispatch.
        self.CONTEXT: CXT = CXT.C1  # Initial context (reset state)
        # Warp routing state (current_nest/current_spu); is_* flags derive from CONTEXT.
        self.warp = WarpState(self)
        # T/S-loop instruction buffers (eager for now — buffering disabled).
        self._tloop_buf = None
        self._sloop_buf = None

        # Layered SPR storage — Tensor-backed via RegisterFile.
        #   gspr: single instance               (shape: [1024])
        #   nspr: per-NEST                       (shape: [NEST, 1024])
        #   lspr: per-(NEST, SPU)                (shape: [NEST, SPU, 1024])
        self.gspr = RegisterFile(GSPR, shape=(1024,), device=DEVICE)
        self.nspr = RegisterFile(NSPR, shape=(NEST_NUM, 1024), device=DEVICE)
        self.lspr = RegisterFile(LSPR, shape=(NEST_NUM, SPU_NUM, 1024), device=DEVICE)

        self._mxe_accum: torch.Tensor = torch.zeros(
            (NEST_NUM, SPU_NUM), dtype=torch.float32,
            device=DEVICE)
        self._credit_ld: torch.Tensor = torch.zeros(
            (NEST_NUM, SPU_NUM), dtype=torch.int32,
            device=DEVICE)
        self._credit_st: torch.Tensor = torch.zeros(
            (NEST_NUM, SPU_NUM), dtype=torch.int32,
            device=DEVICE)

        self._disasm_entries: List[disasm_insn_t] = []

        self.reset_register()
        self.mem.load_via_env()

        import atexit as _atexit
        _atexit.register(self.mem.dump_via_env)

        if os.environ.get("GTX_PROFILE"):
            # 데이터 저장하고 싶을때.
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

        
    def reset_register(self) -> None:
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
        self.lspr.reset()

    # ------------------------------------------------------------------
    # ROCC virtual methods
    # ------------------------------------------------------------------
    def get_disasms(self, proc: processor_t) -> List[Any]:
        if not self._disasm_entries:
            self._disasm_entries = inst_register.collect_disasms()
        return self._disasm_entries

    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []   # SPRs are NOT CSRs (project convention)

    def reset(self, proc: processor_t) -> None:
        proc.state.XPR.write(2, _SP_INIT_VALUE)
        try:
            mstatus = proc.get_csr(_CSR_MSTATUS)
            proc.put_csr(_CSR_MSTATUS, (mstatus & ~_MSTATUS_FS_MASK) | _MSTATUS_FS_INITIAL)
        except Exception:
            pass

        self._mxe_accum.fill_(0.0)
        self._credit_ld.fill_(0)
        self._credit_st.fill_(0)
        self.mem.reset_scratchpads()
        self.reset_register()
        self.deferred_ddr_stores.clear()

        self.CONTEXT = CXT.C1
        self.warp.reset()

    def custom0(self, proc, insn, xs1, xs2) -> int:
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        func = inst_register._c0_funcs.get((insn.funct, funct3))
        if func:
            nemonic = getattr(func, 'mnemonic', 'unknown')
            c0_insn = Custom0_Insn(nemonic, insn)
            return func(self, proc, c0_insn, self.CONTEXT)
        return 0

    def custom1(self, proc, insn, xs1, xs2) -> int:
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        func = inst_register._c1_funcs.get((funct3))
        if func:
            nemonic = getattr(func, 'mnemonic', 'unknown')
            c1_insn = Custom1_Insn(nemonic, insn)
            return func(self, proc, c1_insn, self.CONTEXT)
        return 0

    # ------------------------------------------------------------------
    # Deferred-store flush — wired from end_p / credit_st_chk handlers.
    # ------------------------------------------------------------------
    def flush_deferred_ddr_stores(self) -> None:
        """Drain the S-loop deferred L2→DDR store queue."""
        _dbg = os.environ.get("GTX_DEBUG_FLUSH")
        if _dbg:
            print(f"[FLUSH] queue len={len(self.deferred_ddr_stores)}",
                  file=sys.stderr, flush=True)
        if not self.deferred_ddr_stores:
            return
        from .config_params import L2_SIZE_BYTES
        for req in self.deferred_ddr_stores:
            l2_src = self.mem.l2_byte(req.nest).cpu()
            if _dbg:
                _samp = l2_src[req.l2_off:req.l2_off + 8].tolist()
                print(f"[FLUSH] nest={req.nest} l2_off={req.l2_off:#x} "
                      f"ddr_off={req.ddr_off:#x} len={req.length} h={req.height} "
                      f"l2_sample={_samp}", file=sys.stderr, flush=True)
            max_off = req.ddr_off + (req.height - 1) * req.ddr_stride + req.length
            self.mem.ensure_ddr(max_off)
            cap = self.mem.ddr.capacity()
            for row in range(req.height):
                ddr_off = req.ddr_off + row * req.ddr_stride
                l2_off = (req.l2_off + row * req.l2_stride) % L2_SIZE_BYTES
                copy_len = min(req.length, cap - ddr_off, L2_SIZE_BYTES - l2_off)
                if copy_len > 0:
                    self.mem.ddr.write(
                        ddr_off, l2_src[l2_off:l2_off + copy_len]
                    )
        self.deferred_ddr_stores.clear()
