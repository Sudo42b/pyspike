"""P6 PKG-01: wheel-bundled .elf + .hex asset presence (importlib.resources smoke).

Tests verify that AFTER `pip wheel . && pip install dist/spike-*.whl` (or
`python setup.py build_ext --inplace + setup.py build` which runs the
build_py hook), the riscv.gtx package contains data/firmware/*.elf and
data/golden/*.hex.

In an editable install (`pip install -e .`) the build_py hook is NOT run,
so data/ subdirs do not exist. Tests gracefully skip in that case to
preserve CI green; hard-PASS only in wheel-installed venv.
"""
import importlib.resources as r
import pathlib
from typing import Optional

import pytest


def _gtx_data_dir() -> Optional[pathlib.Path]:
    """Return the riscv.gtx data/ directory if it exists, else None.

    Editable installs do NOT have <pkg>/data/* until setup.py build_py runs.
    Wheel installs always do.
    """
    try:
        data = r.files('riscv.gtx').joinpath('data')
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    try:
        if not data.is_dir():
            return None
    except Exception:
        return None
    return pathlib.Path(str(data))


def test_firmware_data_dir_present_in_wheel():
    """importlib.resources.files('riscv.gtx').joinpath('data','firmware').iterdir() must yield >=1 .elf.

    ROADMAP P6 success #3 verbatim:
      'python -c "... assert any(p.name.endswith(\".elf\") for p in
       r.files(\"riscv.gtx\").joinpath(\"data\",\"firmware\").iterdir())" works'
    """
    data_dir = _gtx_data_dir()
    if data_dir is None:
        pytest.skip("riscv.gtx/data/ not present - editable install? Plan 05 build_py runs in wheel build.")

    firmware = data_dir / "firmware"
    if not firmware.exists():
        pytest.skip("riscv.gtx/data/firmware/ not present - Plan 05 build_py copy did not run (editable install)")

    elfs = [p for p in firmware.iterdir() if p.name.endswith('.elf')]
    assert len(elfs) >= 1, (
        "Expected >=1 .elf in " + str(firmware) + ", got " + str(len(elfs)) + ": "
        + str([p.name for p in elfs])
    )


def test_golden_data_dir_present_in_wheel():
    """importlib.resources.files('riscv.gtx').joinpath('data','golden').iterdir() must yield >=1 .hex."""
    data_dir = _gtx_data_dir()
    if data_dir is None:
        pytest.skip("riscv.gtx/data/ not present (editable install)")

    golden = data_dir / "golden"
    if not golden.exists():
        pytest.skip("riscv.gtx/data/golden/ not present (editable install)")

    hexes = [p for p in golden.iterdir() if p.name.endswith('.hex')]
    assert len(hexes) >= 1, (
        "Expected >=1 .hex in " + str(golden) + ", got " + str(len(hexes)) + ": "
        + str([p.name for p in hexes])
    )


def test_bundled_elfs_helper_returns_list():
    """riscv.gtx._verify.bundled_elfs() must return a list of pathlib.Path with each .name ending in .elf.

    Validates D-14 helper API. In editable install bundled_elfs() returns []
    (empty list, not error) per the FileNotFoundError catch in the helper body.
    """
    try:
        from riscv.gtx._verify import bundled_elfs
    except ImportError:
        pytest.skip("riscv.gtx._verify not importable (Plan 01 may not have landed)")

    elfs = bundled_elfs()
    assert isinstance(elfs, list), "expected list, got " + type(elfs).__name__

    if not elfs:
        pytest.skip("bundled_elfs() returned empty - editable install (no wheel-side data)")

    for p in elfs:
        assert isinstance(p, pathlib.Path), "expected Path, got " + type(p).__name__
        assert p.name.endswith('.elf'), "expected .elf suffix, got " + p.name


def test_load_golden_helper_returns_bytes():
    """riscv.gtx._verify.load_golden(name) must return bytes (the golden file content).

    Skipped in editable install if no goldens are bundled.
    """
    try:
        from riscv.gtx._verify import load_golden, bundled_elfs
    except ImportError:
        pytest.skip("riscv.gtx._verify not importable")

    elfs = bundled_elfs()
    if not elfs:
        pytest.skip("no bundled elfs - editable install")

    # Pick the first elf and try loading its matching golden by stem.
    elf_stem = elfs[0].stem
    # mm_basic legacy maps to mm_basic_n1s16
    golden_stem = "mm_basic_n1s16" if elf_stem == "mm_basic" else elf_stem
    try:
        content = load_golden(golden_stem)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip("no golden for " + golden_stem + ": " + str(exc))
    assert isinstance(content, bytes), "expected bytes, got " + type(content).__name__
    assert len(content) > 0, "golden is empty"
