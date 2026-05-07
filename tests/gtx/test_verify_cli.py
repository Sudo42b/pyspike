"""VRF-01: riscv.gtx._verify CLI smoke tests (Plan 01 GREEN against Task 1).

Validates:
- `python -m riscv.gtx._verify --help` exits 0 with --strict / --ulp / --atol / --fp16 visible
- `pyspike-verify --help` exits 0 (skip if not on PATH; hard PASS after Plan 05 wheel install)
- main([result, golden, --strict, --fp16]) returns 0 for self-compare

P5 plan-05 D-1: module-local _VERIFY_AVAILABLE detection.
"""
import pathlib
import shutil
import subprocess
import sys

import pytest

try:  # pragma: no cover
    from riscv.gtx._verify import main as _verify_main
    _VERIFY_AVAILABLE = True
except ImportError:
    _VERIFY_AVAILABLE = False


def _write_hex(tmp_path: pathlib.Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_python_m_verify_help_exits_zero():
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    result = subprocess.run(
        [sys.executable, "-m", "riscv.gtx._verify", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}; stderr={result.stderr}"
    assert "--strict" in result.stdout, "--strict missing from help"
    assert "--ulp" in result.stdout, "--ulp missing from help"
    assert "--atol" in result.stdout, "--atol missing from help"
    assert "--fp16" in result.stdout, "--fp16 missing from help"


def test_pyspike_verify_console_script_help():
    """Hard PASS only after Plan 05 wheel install or `pip install -e .`."""
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    cli = shutil.which("pyspike-verify")
    if cli is None:
        pytest.skip("pyspike-verify not on PATH; Plan 05 wheel install or `pip install -e .` required")
    result = subprocess.run(
        [cli, "--help"], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--strict" in result.stdout


def test_main_returns_zero_for_self_compare(tmp_path):
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    hex_body = "00" * 32 + "\n"
    a = _write_hex(tmp_path, "a.hex", hex_body)
    g = _write_hex(tmp_path, "g.hex", hex_body)
    rc = _verify_main([a, g, "--strict", "--fp16"])
    assert rc == 0
