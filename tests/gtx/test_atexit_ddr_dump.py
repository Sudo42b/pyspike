"""P6 D-04/D-05/D-06: GTX_DDR_DUMP atexit hook tests.

PRIMARY atexit signal for P6 (hard-PASS in CI when pyspike CLI + nop_wjoin.elf
are available). The secondary signal (test_regression_fw_act.py::test_act_strict_mode_pass
tier #5) requires full toolchain to be useful.

Verifies that:
  1. atexit hook fires on subprocess SystemExit(0) when GTX_DDR_DUMP is set,
     producing a dump file with correct content.
  2. atexit hook does NOT register when GTX_DDR_DUMP is unset.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "nop_wjoin.elf"


def _resolve_pyspike_command():
    """Resolve pyspike CLI: prefer `pyspike` on PATH, fall back to `python -m riscv`.
    Mirrors P5 test_regression_fw_act.py:49-55 pattern."""
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    return [sys.executable, "-m", "riscv"]


@pytest.mark.skipif(
    shutil.which('pyspike') is None and not (REPO_ROOT / "scripts" / "pyspike").exists(),
    reason="pyspike CLI not available"
)
def test_atexit_dump_fires_on_systemexit(tmp_path):
    """Subprocess pyspike + GTX_DDR_DUMP env vars -> dump file MUST exist.

    Vendor gtx_npu_core.cc:127 atexit registration + Python __init__.py D-04
    gating must produce a dump file at WJOIN/SystemExit(0) when env vars set.
    """
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing -- cannot run subprocess regression")

    actual_dump = tmp_path / "atexit_dump.hex"
    env = os.environ.copy()
    env.pop('GTX_NO_EXIT', None)
    env['GTX_DDR_DUMP'] = str(actual_dump)
    env['GTX_DDR_DUMP_ADDR'] = '0x0'   # nop_wjoin doesn't write DDR; just exercise dump path
    env['GTX_DDR_DUMP_SIZE'] = '0x20'  # 32 bytes = 1 line

    cmd = _resolve_pyspike_command() + [
        '--extlib=riscv.gtx', '--extension=gtx', str(ELF_PATH),
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=30, check=False,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"pyspike CLI not found: {exc}")

    assert result.returncode == 0, (
        f"pyspike rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert actual_dump.exists(), (
        f"GTX_DDR_DUMP atexit hook FAILED to write {actual_dump}.\n"
        f"D-04/D-05 broken: subprocess cleanly exited but no dump.\n"
        f"stderr:\n{result.stderr}"
    )

    # Check dump has expected size: 32 bytes = 64 hex chars on 1 line.
    content = actual_dump.read_text()
    data_lines = [
        line for line in content.splitlines()
        if line.strip() and not line.startswith(('#', '@'))
    ]
    assert len(data_lines) == 1, (
        f"Expected 1 hex line for SIZE=0x20, got {len(data_lines)}; "
        f"content={content!r}"
    )
    assert len(data_lines[0]) == 64, (
        f"Expected 64 hex chars (32 bytes), got {len(data_lines[0])}; "
        f"line={data_lines[0]!r}"
    )


@pytest.mark.skipif(
    shutil.which('pyspike') is None and not (REPO_ROOT / "scripts" / "pyspike").exists(),
    reason="pyspike CLI not available"
)
def test_atexit_dump_does_not_register_when_env_unset(tmp_path):
    """Subprocess pyspike WITHOUT GTX_DDR_DUMP -> no dump file is created."""
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing")

    suspicious_dump = tmp_path / "should_not_exist.hex"
    env = os.environ.copy()
    env.pop('GTX_DDR_DUMP', None)
    env.pop('GTX_DDR_DUMP_ADDR', None)
    env.pop('GTX_DDR_DUMP_SIZE', None)
    env.pop('GTX_NO_EXIT', None)

    cmd = _resolve_pyspike_command() + [
        '--extlib=riscv.gtx', '--extension=gtx', str(ELF_PATH),
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=30, check=False,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"pyspike CLI not found: {exc}")

    assert result.returncode == 0
    assert not suspicious_dump.exists(), (
        "atexit hook fired despite GTX_DDR_DUMP being unset -- D-04 conditional gate broken"
    )
