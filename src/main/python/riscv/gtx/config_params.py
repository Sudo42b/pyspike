import torch


# --------------------------------------------------------------------------
# Torch backend device — single source of truth.
# --------------------------------------------------------------------------
# Every GTX tensor (L0/L1/L2/DDR scratchpads, MXE accumulators, kernel
# temporaries) lives on ``DEVICE``. CUDA is the default whenever it is
# available; otherwise CPU.
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# NEST x SPU topology
GTX_NEST_NUM: int = 4
GTX_SPU_NUM: int = 16          # SPUs per NEST
GTX_SPUS_PER_NEST: int = GTX_SPU_NUM   # alias for clarity
# D-02 default: 4 GiB
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024
# D-13: floor for doubling-grow first allocation. 1 MiB picked because:
#   - covers 32-byte bus-word minimum with ample headroom
#   - small enough that CI per-test allocations are cheap
#   - large enough that "single grow per test" is the common case
INITIAL_FLOOR: int = 1 * 1024 * 1024

# Memory sizes (bytes)
GTX_L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
GTX_L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
GTX_L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

# DDR (D-02: capped by GTX_DDR_SIZE env var; default below)
GTX_DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

# DDR I/O (D-03)
GTX_DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for GTX_DDR_REVERSED reversal

# DDR base physical address (firmware GTX_MAIN_BASE -- gtx_params.h:24)
GTX_DDR_BASE: int = 0x370000000

# SPR address ranges — source of truth is `unit/csr/__init__.py:42-47`.
# Don't redefine the GSPR_BASE / NSPR_BASE / LSPR_BASE constants here;
# import them from `riscv.gtx.unit.csr` instead.
