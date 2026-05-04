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
"""Tests for CORE-01 -- @isa.register('gtx') registration.

Two-tier validation per D-21:
  Tier 1 (always run, even without _riscv.so): module imports, attribute exposure.
  Tier 2 (skipif _riscv.so missing): subclass + register_extension factory.

Note: module-level _RISCV_AVAILABLE detection (NOT the conftest fixture) so
that the acceptance command `pytest ... --noconftest -o "addopts="` still
selects the correct branch. The conftest fixture path is exercised when
running without --noconftest.
"""
import pytest

import riscv.gtx


# Module-level detection (mirrors tests/gtx/conftest.py D-17 pattern but is
# self-contained so it survives --noconftest).
try:  # pragma: no cover -- branch depends on whether _riscv.so was built
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


# ----------------- Tier 1: always-run design contract -----------------

def test_gtx_module_imports_without_error():
    """riscv.gtx must always import (Phase 1 contract preserved)."""
    assert hasattr(riscv.gtx, "GtxNpu")
    # GtxNpu may be None when _riscv.so is missing; that is expected.
    # The attribute must exist either way.


def test_gtx_exports_match_all():
    """riscv.gtx.__all__ contains the documented names."""
    for name in ("encoding", "fp", "memory", "params", "ddr", "npu", "GtxNpu"):
        assert name in riscv.gtx.__all__, f"{name!r} not in __all__"


# ----------------- Tier 2: requires _riscv.so -----------------

def test_gtxnpu_is_rocc_subclass():
    """When _riscv.so is built, GtxNpu must inherit from isa.ROCC."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built -- GtxNpu is None")
    # pylint: disable=import-error,no-name-in-module
    from riscv import isa
    from riscv.gtx import GtxNpu
    assert GtxNpu is not None
    assert issubclass(GtxNpu, isa.ROCC), (
        f"GtxNpu must inherit from isa.ROCC, MRO: {GtxNpu.__mro__}"
    )


def test_gtxnpu_name_property():
    """@isa.register('gtx') sets .name property to 'gtx'."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built")
    from riscv.gtx import GtxNpu
    npu = GtxNpu()
    # `name` is a property (not an instance attribute)
    assert npu.name == "gtx", f"expected 'gtx', got {npu.name!r}"


def test_register_extension_factory_finds_gtx():
    """The factory bound by @isa.register must be discoverable via Spike's
    registry. This is what `pyspike --extlib=riscv.gtx ...` relies on
    (research §1054-1062 Pitfall 6)."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built")
    try:
        # pylint: disable=import-error,no-name-in-module
        from riscv.extension import find_extension
    except ImportError:
        pytest.skip("riscv.extension.find_extension not exposed in this build")
    # Importing riscv.gtx already triggered @isa.register('gtx').
    ext_factory = find_extension("gtx")
    assert ext_factory is not None
