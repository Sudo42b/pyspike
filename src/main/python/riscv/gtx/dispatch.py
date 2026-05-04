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


def build_custom0_table(npu) -> Dict[int, Callable]:
    """Build funct7 -> handler dict (closure-binding npu)."""
    raw = _registry.collect_for_kind("custom0")
    return {f7: _bind(fn, npu) for f7, fn in raw.items()}


def build_custom1_table(npu) -> Dict[int, Callable]:
    """Build funct3 -> handler dict (closure-binding npu)."""
    raw = _registry.collect_for_kind("custom1")
    return {f3: _bind(fn, npu) for f3, fn in raw.items()}


def _bind(fn: Callable, npu) -> Callable:
    def wrapped(proc, insn, xs1, xs2):
        return fn(npu, proc, insn, xs1, xs2)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped
