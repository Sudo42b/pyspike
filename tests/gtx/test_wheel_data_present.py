"""PKG-01 RED scaffold: wheel-bundled firmware/golden assets smoke (Plan 05 GREEN-fills).

This file is a RED scaffold landed by Plan 01 (Wave 1a). Plan 05 (Wave 2) will
GREEN-fill these tests after the build_py hook copies tests/gtx/data/{elf,golden}/
to riscv/gtx/data/{firmware,golden}/ and pyproject.toml's
[tool.setuptools.package-data] is extended to include riscv.gtx data.

Cross-wave safety (D-17 + D-18): Plan 05 is in Wave 2 (sequential after Wave 1b),
not Wave 1a, so no parallel-execution conflict with this RED scaffold.
"""
import importlib.resources as r

import pytest


def test_firmware_data_dir_present_in_wheel():
    """Plan 05 GREEN-fills: assert riscv.gtx.data.firmware contains at least one .elf."""
    pytest.skip(
        "Plan 05 (Wave 2) build_py hook copies tests/gtx/data/elf/ "
        "to riscv/gtx/data/firmware/"
    )


def test_golden_data_dir_present_in_wheel():
    """Plan 05 GREEN-fills: assert riscv.gtx.data.golden contains at least one .hex."""
    pytest.skip(
        "Plan 05 (Wave 2) build_py hook copies tests/gtx/data/golden/ "
        "to riscv/gtx/data/golden/"
    )


def test_bundled_elfs_helper_returns_list():
    """Plan 05 GREEN-fills: assert riscv.gtx._verify.bundled_elfs() returns non-empty list."""
    pytest.skip(
        "Plan 05 (Wave 2) wheel-install context required: "
        "after `pip install dist/*.whl`, bundled_elfs() must return non-empty list"
    )
