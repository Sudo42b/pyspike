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
"""Dispatch dict builders -- wires _registry handlers into custom0/custom1 tables.

Called from GtxNpu.__init__. Creates closures binding `npu` (the GtxNpu instance)
so handlers can read npu.warp / npu.gspr / npu.spr_router etc.
"""
from typing import Callable, Dict
from . import _registry


def build_custom0_table(npu) -> Dict[int, Dict]:
    """Build funct7 → context → {funct3-or-None: bound-handler} 3-level dict.

    Closure-binds npu so handlers can read npu.warp / npu.gspr / npu.mem.

    Levels:
      L1 (outer)  funct7 (int)
      L2 (middle) NpuContext or None
                  None = universal (handler valid in any context, matches
                  legacy @handler calls without context=).
                  NpuContext.Cx = per-context override.
      L3 (inner)  funct3 (int) when mask_funct3=True, else None (P2 back-compat
                  sentinel — dispatcher tries None first, then funct3).
    """
    raw = _registry.collect_for_kind("custom0")
    return {
        f7: {
            ctx_key: {f3: _bind(fn, npu) for f3, fn in inner.items()}
            for ctx_key, inner in ctx_table.items()
        }
        for f7, ctx_table in raw.items()
    }


def build_custom1_table(npu) -> Dict[int, Dict]:
    """Build funct3 → context → bound-handler 2-level dict.

    Levels:
      L1 (outer)  funct3 (int)
      L2 (inner)  NpuContext or None (universal).
    """
    raw = _registry.collect_for_kind("custom1")
    return {
        f3: {ctx_key: _bind(fn, npu) for ctx_key, fn in inner.items()}
        for f3, inner in raw.items()
    }


def _bind(fn: Callable, npu) -> Callable:
    def wrapped(proc, insn, xs1, xs2):
        return fn(npu, proc, insn, xs1, xs2)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    # Propagate mnemonic (set by _registry.handler decorator on fn) so
    # npu._state_dispatch can extract it for warp-marker detection.
    wrapped.gtx_mnemonic = getattr(fn, "gtx_mnemonic", None)  # type: ignore[attr-defined]
    return wrapped


# NOTE: the legacy `dispatch_4mode` router has been retired — its warp-state
# broadcast logic is subsumed by the FSM's DISPATCH stage plus the per-handler
# warp routing helpers in unit/context/{control,dma}.py.
