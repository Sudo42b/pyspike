import os

import numpy as np

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

# NumPy backend is CPU-only; DEVICE kept as a sentinel for call-site compat.
DEVICE: str = "cpu"

# GTX ISA spec revision this backend targets (SMM_ISA_v2.0.0d.xlsx). 2.0.0d is
# the FP32-native revision: matmul renamed mm/mmt/mmc/mmct (was mm.o/mm.v/...),
# sum/dot moved from the matrix unit to the vector unit (sum.v/dot.vv/sum.i/
# dot.ii), and the conversion family re-encoded one op per (funct7, funct3) slot
# (scvt.qs/hq/is/bi/sn/bs/hs/hf/si). Bump this whenever the decode tables track a
# new SMM_ISA spreadsheet revision.
GTX_ISA_VERSION: str = "2.0.0d"


# NEST x SPU topology
NEST_NUM: int = 4
SPU_NUM: int = 16          # SPUs per NEST
SPUS_PER_NEST: int = SPU_NUM   # alias for clarity
# D-02 default: 4 GiB
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024
# D-13: floor for doubling-grow first allocation. 1 MiB picked because:
#   - covers 32-byte bus-word minimum with ample headroom
#   - small enough that CI per-test allocations are cheap
#   - large enough that "single grow per test" is the common case
INITIAL_FLOOR: int = 1 * 1024 * 1024

# Memory sizes (bytes)
L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

# MX numeric I/O width at the L1/L0 read/write boundaries. Internal compute is
# always FP32; widening the I/O to FP32 removes the FP16-cast precision loss at
# the boundaries (a deliberate divergence from vendor). Set MX_IO_DTYPE to
# np.float16 to restore vendor-exact FP16 I/O — every MX op routes its
# operand decode / L1 / L0 / SVR access through these two constants.
# Env override GTX_MX_IO_DTYPE=float16 selects vendor-exact FP16 I/O (the
# baseline that matches the FP16 SystemC-ISS goldens, e.g. ggml_ops_c).
MX_IO_DTYPE: type = (
    np.float16
    if os.environ.get("GTX_MX_IO_DTYPE", "float32").lower() in ("float16", "fp16", "16")
    else np.float32
)
MX_IO_BYTES: int = np.finfo(MX_IO_DTYPE).bits // 8

# External numeric width in DDR/L2. SMM_ISA v2.0.0d unifies the whole hierarchy
# to fp32 — DDR, L2, L1, L0 all carry MX_IO_DTYPE, so MX_EXT tracks MX_IO and the
# T-loop L2↔L1 DMA is a raw byte copy (MX_EXT_DTYPE == MX_IO_DTYPE ⇒ no convert).
# (Flipping MX_IO_DTYPE to float16 for vendor-parity carries DDR/L2 along, so the
# two sides always match and the DMA never needs a width conversion.)
MX_EXT_DTYPE: type = MX_IO_DTYPE
MX_EXT_BYTES: int = np.finfo(MX_EXT_DTYPE).bits // 8

# DDR (D-02: capped by DDR_SIZE env var; default below)
DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

# DDR I/O (D-03)
DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for DDR_REVERSED reversal

# DDR base physical address (firmware MAIN_BASE -- gtx_params.h:24)
DDR_BASE: int = 0x370000000
