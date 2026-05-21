"""GTX NPU functional model — pyspike RoCC extension package.

Importing this package registers the ``gtx`` extension (via ``@isa.register``
on :class:`~riscv.gtx.npu.GtxNpu`) and populates the custom0/custom1 handler
registry as a side-effect of importing :mod:`riscv.gtx.npu`.
"""
import sys

if sys.byteorder != "little":
    raise RuntimeError(
        f"riscv.gtx requires a little-endian host; got '{sys.byteorder}'. "
        f"NumPy/torch float16 view semantics assume LE byte order."
    )

from . import config_params
from . import memory
from . import npu
from . import devices  # noqa: F401  — registers the sifive_exit MMIO device
from .npu import GtxNpu

import os as _os
if _os.environ.get("GTX_PROF"):
    from . import _prof
    _prof.install()

__all__ = ["config_params", "memory", "npu", "devices", "GtxNpu"]
