"""VRF-01: riscv.gtx._verify.compare_hex unit tests (Plan 01 GREEN against Task 1).

Validates production compare_hex (D-01 hybrid base) for:
- strict zeros self-compare
- stats dict mini-port + vendor-alias keys both present
- BE bit-pair compare (verify.py:235)
- tolerant within-ULP pass
- strict rejects within-tolerance
- @-line and #-line skipping

P5 plan-05 D-1: module-local _VERIFY_AVAILABLE detection (NOT conftest fixture)
because acceptance commands run with `--noconftest`.
"""
import pathlib

import pytest

try:  # pragma: no cover
    from riscv.gtx._verify import compare_hex
    _VERIFY_AVAILABLE = True
except ImportError:
    _VERIFY_AVAILABLE = False


def _write_hex(tmp_path: pathlib.Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_self_compare_zeros_strict_passes(tmp_path):
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    hex_body = "00" * 32 + "\n"
    a = _write_hex(tmp_path, "a.hex", hex_body)
    g = _write_hex(tmp_path, "g.hex", hex_body)
    passed, stats = compare_hex(a, g, strict=True)
    assert passed is True
    assert stats["exact_matches"] == 16
    assert stats["total_fp16"] == 16
    assert stats["within_tolerance"] == 0
    assert stats["failures"] == 0


def test_compare_hex_strict_keys_present(tmp_path):
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    hex_body = "00" * 32 + "\n"
    a = _write_hex(tmp_path, "a.hex", hex_body)
    g = _write_hex(tmp_path, "g.hex", hex_body)
    _, stats = compare_hex(a, g, strict=True)
    mini_port_keys = {"exact_matches", "within_tolerance", "failures",
                      "total_fp16", "first_failure"}
    vendor_alias_keys = {"mismatches", "first_mismatch", "size_result",
                         "size_golden", "total_bytes"}
    assert mini_port_keys.issubset(stats.keys()), \
        f"missing mini-port keys: {mini_port_keys - stats.keys()}"
    assert vendor_alias_keys.issubset(stats.keys()), \
        f"missing vendor-alias keys: {vendor_alias_keys - stats.keys()}"
    assert stats["failures"] == stats["mismatches"]


def test_compare_hex_be_bit_pair(tmp_path):
    """BE bit-pair: high byte 0x3C first, low byte 0x00 second decodes 0x3C00 = FP16(1.0)."""
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    hex_body = "3C00" * 16 + "\n"
    a = _write_hex(tmp_path, "a.hex", hex_body)
    g = _write_hex(tmp_path, "g.hex", hex_body)
    passed, stats = compare_hex(a, g, strict=True)
    assert passed is True
    assert stats["exact_matches"] == 16


def test_compare_hex_tolerant_within_ulp_passes(tmp_path):
    """Last FP16 differs by 1 ULP (0x3C00 vs 0x3C01); strict=False ulp=1 passes."""
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    a_body = "3C00" * 15 + "3C01" + "\n"
    g_body = "3C00" * 16 + "\n"
    a = _write_hex(tmp_path, "a.hex", a_body)
    g = _write_hex(tmp_path, "g.hex", g_body)
    passed, stats = compare_hex(a, g, strict=False, ulp=1)
    assert passed is True
    assert stats["within_tolerance"] == 1
    assert stats["failures"] == 0


def test_compare_hex_strict_rejects_within_tolerance(tmp_path):
    """Same input as tolerant test, but strict=True rejects (exact_matches < total_fp16)."""
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    a_body = "3C00" * 15 + "3C01" + "\n"
    g_body = "3C00" * 16 + "\n"
    a = _write_hex(tmp_path, "a.hex", a_body)
    g = _write_hex(tmp_path, "g.hex", g_body)
    passed, stats = compare_hex(a, g, strict=True, ulp=1)
    assert passed is False
    assert stats["within_tolerance"] == 1
    assert stats["exact_matches"] == 15


def test_compare_hex_skips_at_and_hash_lines(tmp_path):
    """@<addr> and # comment lines must be skipped, not parsed as bytes."""
    if not _VERIFY_AVAILABLE:
        pytest.skip("riscv.gtx._verify not importable; Task 1 must complete first")
    a_body = "# comment\n@370000000\n" + "3C00" * 16 + "\n"
    g_body = "3C00" * 16 + "\n"
    a = _write_hex(tmp_path, "a.hex", a_body)
    g = _write_hex(tmp_path, "g.hex", g_body)
    passed, stats = compare_hex(a, g, strict=True)
    assert passed is True
    assert stats["exact_matches"] == 16
