"""VRF-04 RED scaffold: parametrized strict-mode .elf regression matrix (Plan 04 GREEN-fills).

This file is a RED scaffold landed by Plan 01 (Wave 1a). Plan 04 (Wave 1b) will
GREEN-fill the per-elf strict-mode regression body that:
  1. Subprocess `pyspike --extlib=riscv.gtx <elf>` with GTX_DDR_DUMP env vars set
  2. Reads dumped DDR hex
  3. Runs compare_hex(strict=True) against bundled golden
  4. Asserts passed and stats['within_tolerance'] == 0 (every byte exact)

The parametrize anchor `BUNDLED_ELFS = sorted(ELF_DIR.glob("*.elf"))` is the
key point of contact — Plan 04 will tighten it to require non-empty.

Cross-wave safety (D-17 + D-18): Plan 04 is in Wave 1b (sequential after Wave 1a),
not Wave 1a, so no parallel-execution conflict with this RED scaffold.
"""
import pathlib

import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "elf"
GOLDEN_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "golden"
BUNDLED_ELFS = sorted(ELF_DIR.glob("*.elf")) if ELF_DIR.exists() else []


@pytest.mark.parametrize(
    "elf_path",
    BUNDLED_ELFS or [pathlib.Path("placeholder.elf")],
    ids=lambda p: p.stem,
)
def test_regression_fw_full(elf_path, tmp_path):
    """Parametrized strict-mode regression — Plan 04 GREEN-fills body.

    `BUNDLED_ELFS or [placeholder]` ensures parametrize collects at least one
    test ID even if Plan 03 hasn't landed assets yet. Plan 04 will tighten to
    `BUNDLED_ELFS` only and require non-empty.
    """
    pytest.skip(
        f"Plan 04 (Wave 1b) GREEN-fills the strict-mode regression body for "
        f"{elf_path.stem}"
    )
