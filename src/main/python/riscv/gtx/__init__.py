"""
    GTX NPU functional model
        FP16 helpers (`fp`), 
        memory layer (`memory`), 
        HW parameter constants (`params`)
        funct7 encoding (`encoding`)
        DDR helpers (`ddr`).

The ROCC subclass is added in Phase 2 (D-14).
"""
import faulthandler
import os
import sys

# Enable faulthandler in embedded interpreter so SIGSEGV produces a traceback
# (Python's -X faulthandler doesn't reach pyspike's embedded interpreter).
if os.environ.get('PYSPIKE_FAULTHANDLER', '1') != '0':
    faulthandler.enable()

# D-09 / RESEARCH.md "Anti-Patterns": np.float16 view assumes little-endian host.
# manylinux2014_x86_64 is always LE; this tripwire defends against accidental
# non-LE host (theoretical -- not in v1 platform target).
if sys.byteorder != "little":
    raise RuntimeError(
        f"riscv.gtx requires little-endian host (sys.byteorder='little'); "
        f"got '{sys.byteorder}'. NumPy float16 view semantics assume LE byte order."
    )

from . import encoding
from . import fp
from . import memory
from . import params
from . import ddr

# Pitfall 6 (research): pyspike --extlib=riscv.gtx imports this module;
# importing npu triggers @isa.register("gtx") which makes the extension
# findable by Spike's PythonBridge.
try:
    from . import npu   # noqa: F401
    from .npu import GtxNpu
except ImportError as _exc:

    npu = None  # type: ignore[assignment]
    GtxNpu = None  # type: ignore[assignment]
    raise ImportError(
        "riscv.gtx.npu is not available. This may be expected if you are running "
        "Phase 1 tests without a built wheel. If you need npu functionality, please "
        "build the wheel or install from source with 'pip install .'."
    )

__all__ = ["encoding", "fp", "memory", "params", "ddr", "npu", "GtxNpu"]

try:
    import torch
    DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    raise ImportError(
        "PyTorch is required for riscv.gtx. Please install PyTorch to use this module.")