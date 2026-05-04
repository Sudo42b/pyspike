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
"""GTX NPU functional model -- Phase 1 skeleton.

Phase 1 exposes FP16 helpers (`fp`), memory layer (`memory`), HW parameter
constants (`params`), funct7 encoding (`encoding`), and DDR helpers (`ddr`).
The ROCC subclass is added in Phase 2 (D-14).
"""
import sys

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
    # _riscv.so not available -- Phase 1 tests still pass; npu is unavailable
    # until a wheel build (or in-tree build_ext) lands. Plans 02-05 unit tests
    # use mocks via tests/gtx/conftest.py; integration test plan 05 gates
    # on _RISCV_AVAILABLE.
    npu = None  # type: ignore[assignment]
    GtxNpu = None  # type: ignore[assignment]

__all__ = ["encoding", "fp", "memory", "params", "ddr", "npu", "GtxNpu"]
