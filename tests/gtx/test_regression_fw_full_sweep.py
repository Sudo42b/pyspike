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
"""P7 NJIT-04 Tier 2: vendor 84-op directory full sweep (strict-mode).

Plan 05 GREEN-fills: parametrize over 84 vendor op directories,
invoke pyspike subprocess + compare_hex(strict=True). Wave 0 leaves
all parametrize invocations as `pytest.skip(...)`.
"""
from __future__ import annotations
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VENDOR_TEST_DIR = REPO_ROOT / "vendor" / "gtx_cpp_reference" / "test"

# Auto-discover 84 vendor op directories at collection time
if VENDOR_TEST_DIR.exists():
    VENDOR_OP_DIRS = sorted(
        p.name for p in VENDOR_TEST_DIR.iterdir()
        if p.is_dir() and p.name != "__pycache__" and p.name[:1].isupper()
    )
else:
    VENDOR_OP_DIRS = []


@pytest.mark.parametrize("op_dir", VENDOR_OP_DIRS or ["__no_vendor__"], ids=lambda x: x)
def test_vendor_op_sweep_strict(op_dir: str) -> None:
    """Strict-mode regression for one vendor op directory.

    Plan 05 GREEN-fills: looks up `tests/gtx/data/firmware/<op>.elf` +
    `tests/gtx/data/golden/<op>.hex`; subprocess pyspike with GTX_DDR_DUMP
    env vars; compare_hex(strict=True) -> assert PASS or graceful skip
    if assets missing.
    """
    pytest.skip(f"Plan 05 GREEN-fills vendor 84-op sweep (op={op_dir})")
