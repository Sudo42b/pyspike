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
"""End-to-end byte-exact regression over vendor n1s16 firmware bundle.

Parametrized over every ``test/<OP>/n1s16/n1s16_<op>.elf`` that ships an
input.txt + ref.txt pair. Each case:

  1. Reads ref.txt's leading ``@<addr>`` line for DUMP_ADDR and counts
     remaining lines (32 hex chars = 16 bytes? No -- vendor lines are 64
     chars = 32 bytes) for DUMP_SIZE.
  2. Spawns `pyspike --extlib=riscv.gtx --extension=gtx <elf>` with
     ``GTX_NO_EXIT=1`` (multi-tile firmware exits via SystemExit on the
     first __join() without it -- see control.py:wjoin_with_exit).
  3. Byte-exact diff between dump file and ref.txt with the @-header stripped.

This whole module is OPT-IN. Set ``PYTEST_ELF_REGRESSION=1`` to enable.
Single cases can be selected with ``-k <op_dir_lowercased>`` (e.g. ``-k abs``).

Per-case timeout defaults to 1800s (15 min) -- ABS alone needs ~10-15 min on
the current backend. Override via ``PYTEST_ELF_TIMEOUT``.
"""
from __future__ import annotations
import os
import pathlib
import shutil
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_DIR = REPO_ROOT / "test"

# Per-line byte count in vendor ref.txt: 64 hex chars = 32 bytes.
_BYTES_PER_REF_LINE = 32

# Module-level opt-in gate. Without this, the whole file collects 0 tests
# (skipped at module level). Avoids accidental 80-case ~10-hour CI runs.
_OPT_IN = os.environ.get("PYTEST_ELF_REGRESSION") == "1"
_TIMEOUT = int(os.environ.get("PYTEST_ELF_TIMEOUT", "1800"))


def _discover_cases() -> list[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]]:
    """Return (op_id, elf, input_txt, ref_txt) for every n1s16 firmware that
    has the full input/ref pair. Cases missing any artefact are silently
    dropped (they would skip individually otherwise, but pytest's -k filter
    is cleaner when the case never appears).
    """
    if not TEST_DIR.exists():
        return []
    cases = []
    for op_dir in sorted(TEST_DIR.iterdir()):
        n1s16 = op_dir / "n1s16"
        if not n1s16.is_dir():
            continue
        elfs = sorted(n1s16.glob("n1s16_*.elf"))
        if not elfs:
            continue
        elf = elfs[0]
        stem = elf.stem  # e.g. "n1s16_abs"
        data = n1s16 / "data"
        ref = data / f"{stem}_ref.txt"
        inp = data / f"{stem}_input.txt"
        if not ref.exists():
            continue
        cases.append((op_dir.name.lower(), elf, inp, ref))
    return cases


_CASES = _discover_cases()


def _parse_ref(ref_path: pathlib.Path) -> tuple[int, int, pathlib.Path]:
    """Parse ref.txt header to extract (dump_addr, dump_size, stripped_path).

    ``stripped_path`` is a sibling tempfile with ``@<addr>`` lines removed --
    needed for the byte-exact compare since pyspike's dump file has no header.
    """
    addr = None
    data_lines = 0
    stripped = ref_path.with_suffix(".nohdr.txt")
    with ref_path.open("r") as src, stripped.open("w") as dst:
        for line in src:
            s = line.strip()
            if not s:
                continue
            if s.startswith("@"):
                if addr is None:
                    addr = int(s[1:], 16)
                continue
            dst.write(line)
            data_lines += 1
    if addr is None:
        raise ValueError(f"{ref_path}: missing @<addr> header")
    return addr, data_lines * _BYTES_PER_REF_LINE, stripped


def _pyspike_cmd() -> list[str] | None:
    """Resolve `pyspike` -- prefer `uv run scripts/pyspike` (dev) over a
    PATH-installed binary."""
    script = REPO_ROOT / "scripts" / "pyspike"
    if script.exists() and shutil.which("uv"):
        return ["uv", "run", str(script)]
    pyspike = shutil.which("pyspike")
    if pyspike:
        return [pyspike]
    return None


@pytest.mark.skipif(not _OPT_IN,
                    reason="set PYTEST_ELF_REGRESSION=1 to enable")
@pytest.mark.skipif(not _CASES,
                    reason=f"no firmware cases discovered under {TEST_DIR}")
@pytest.mark.parametrize(
    "op_id,elf,inp,ref",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_elf_byte_exact(tmp_path, op_id, elf, inp, ref):
    """Run firmware end-to-end; assert DDR dump matches vendor ref byte-for-byte."""
    cmd_base = _pyspike_cmd()
    if cmd_base is None:
        pytest.skip("pyspike CLI not found (uv/scripts/pyspike or PATH)")

    dump_addr, dump_size, stripped_ref = _parse_ref(ref)
    if dump_size == 0:
        pytest.skip(f"{ref}: zero-byte ref (no data lines)")

    dump_path = tmp_path / f"{elf.stem}_result.hex"
    env = os.environ.copy()
    env.update({
        "GTX_NO_EXIT": "1",
        "GTX_DDR_REVERSED": "1",
        "GTX_DDR_DUMP": str(dump_path),
        "GTX_DDR_DUMP_ADDR": hex(dump_addr),
        "GTX_DDR_DUMP_SIZE": hex(dump_size),
    })
    if inp.exists() and inp.stat().st_size > 0:
        env["GTX_DDR_INIT"] = str(inp)

    cmd = [*cmd_base, "--extlib=riscv.gtx", "--extension=gtx", str(elf)]
    try:
        subprocess.run(cmd, env=env, timeout=_TIMEOUT, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"{op_id}: pyspike timed out after {_TIMEOUT}s\n"
                    f"stderr tail:\n{(e.stderr or b'')[-2000:].decode(errors='replace')}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{op_id}: pyspike exited {e.returncode}\n"
                    f"stderr tail:\n{(e.stderr or b'')[-2000:].decode(errors='replace')}")

    assert dump_path.exists(), f"{op_id}: dump file not produced at {dump_path}"
    assert dump_path.stat().st_size > 0, f"{op_id}: dump file is empty"

    # Byte-exact compare. On mismatch, surface the first differing line for
    # the failure message -- the full diff is too large to embed (ref files
    # can be > 10 MB).
    with stripped_ref.open("rb") as rf, dump_path.open("rb") as df:
        ref_bytes = rf.read()
        dump_bytes = df.read()
    if ref_bytes != dump_bytes:
        # locate the first divergent line
        ref_lines = ref_bytes.splitlines()
        dump_lines = dump_bytes.splitlines()
        first = None
        for i, (a, b) in enumerate(zip(ref_lines, dump_lines), start=1):
            if a != b:
                first = (i, a, b)
                break
        if first is None and len(ref_lines) != len(dump_lines):
            first = (min(len(ref_lines), len(dump_lines)) + 1, b"", b"")
        i, a, b = first or (0, b"", b"")
        pytest.fail(
            f"{op_id}: byte mismatch vs {ref.name}\n"
            f"  ref_lines={len(ref_lines)} dump_lines={len(dump_lines)}\n"
            f"  first diff at line {i}:\n"
            f"    ref:  {a[:80]!r}\n"
            f"    dump: {b[:80]!r}"
        )
