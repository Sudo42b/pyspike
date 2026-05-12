"""GTX NPU functional model.

Subpackages and modules
    config_params           hardware constants (NEST/SPU counts, sizes, DDR base)
    fsm                     one-instruction FSM enum + driver
    decode / dispatch_state / execute / writeback / idle
                            per-state transition functions (see :mod:`fsm`)
    dispatch                table builders that wire @handler entries into the
                            FSM's ``_custom0`` / ``_custom1`` lookup tables
    _registry               ``@handler`` decorator + collectors
    unit/                   HW topology — NSU / NEST / SPU / memory / CSR / FSM
    unit/csr                typed CSR register definitions
                            (gspr.py / nspr.py / lspr.py — name-indexed via
                            :class:`~unit.register_file.RegisterFile`)
    unit/ins                instruction subpackage (encoding, engines, ops)
    unit/context            NPU context FSM (C1..C4), warp state, DMA ops

Registering the RoCC extension ``"gtx"`` (via ``@isa.register`` in :mod:`npu`)
is conditional on the optional ``ddr`` helper being available; when it is
not, importing this package still succeeds so the FSM / topology / CSR
modules remain usable for inspection and unit tests.
"""
import faulthandler
import os
import sys

# Enable faulthandler in the embedded interpreter so SIGSEGV produces a
# traceback (Python's ``-X faulthandler`` does not reach pyspike's
# embedded interpreter).
if os.environ.get('PYSPIKE_FAULTHANDLER', '1') != '0':
    faulthandler.enable()

# D-09 / RESEARCH.md "Anti-Patterns": ``np.float16`` view semantics
# assume little-endian host. ``manylinux2014_x86_64`` is always LE; this
# tripwire defends against accidental non-LE host (theoretical — not in
# v1 platform target).
if sys.byteorder != "little":
    raise RuntimeError(
        f"riscv.gtx requires little-endian host (sys.byteorder='little'); "
        f"got '{sys.byteorder}'. NumPy float16 view semantics assume LE byte "
        f"order."
    )

from . import config_params
from . import fsm
from .unit import memory
from .unit.ins import encoding

# Pitfall 6 (research): pyspike --extlib=riscv.gtx imports this module;
# importing ``npu`` triggers ``@isa.register("gtx")`` which makes the
# extension findable by Spike's PythonBridge. Wrapped in try/except so
# missing optional helpers (e.g. ``ddr.py`` in WIP refactor states) do
# not block import of the FSM / CSR / topology surface.
try:
    from . import npu   # noqa: F401
    from .npu import GtxNpu
    _NPU_AVAILABLE = True
except ImportError as _exc:
    npu = None  # type: ignore[assignment]
    GtxNpu = None  # type: ignore[assignment]
    _NPU_AVAILABLE = False
    import warnings
    warnings.warn(
        f"riscv.gtx.npu unavailable ({_exc}); FSM / CSR / topology surface "
        f"still importable.",
        ImportWarning,
        stacklevel=2,
    )

__all__ = [
    "config_params",
    "encoding",
    "fsm",
    "memory",
    "npu",
    "GtxNpu",
]

try:
    import torch  # noqa: F401 -- imported for early ImportError surface
except ImportError:
    raise ImportError(
        "PyTorch is required for riscv.gtx. Please install PyTorch to use "
        "this module.")

# Re-export the single source of truth for the compute device so callers can
# write `from riscv.gtx import DEVICE` if they prefer the package root path.
from .config_params import DEVICE  # noqa: E402,F401
