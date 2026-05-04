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
"""Tests for CORE-03 / D-07 / D-08: WJOIN GTX_NO_EXIT semantics.

Two handlers under test:
  - wjoin_with_exit       (custom1 funct3=0b101) -- raises SystemExit unless
                                                     GTX_NO_EXIT is set.
  - wjoin_custom0_no_exit (custom0 funct7=0x03)  -- NEVER raises (research
                                                     §439).
"""
from types import SimpleNamespace

import pytest

from riscv.gtx.ops.control import wjoin_with_exit, wjoin_custom0_no_exit
from riscv.gtx.warp_state import WarpState


def _fake_npu():
    return SimpleNamespace(warp=WarpState())


class _FakeProc:
    def __init__(self):
        from tests.gtx._mocks import MockProcessor
        self._mp = MockProcessor()

    def get_state(self):
        return self._mp.get_state()


# ----------------- custom1 funct3=0b101 wjoin_with_exit -----------------

def test_wjoin_default_raises_systemexit(monkeypatch):
    """ROADMAP P2 #5: GTX_NO_EXIT unset -> SystemExit(0)."""
    monkeypatch.delenv('GTX_NO_EXIT', raising=False)
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    with pytest.raises(SystemExit) as exc_info:
        wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert exc_info.value.code == 0


def test_wjoin_with_no_exit_set_returns_zero(monkeypatch):
    """ROADMAP P2 #5: GTX_NO_EXIT=1 -> return 0, no SystemExit."""
    monkeypatch.setenv('GTX_NO_EXIT', '1')
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    ret = wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0


def test_wjoin_no_exit_zero_string_is_truthy(monkeypatch):
    """Document our truthiness: '0' is non-empty -> bool('0') is True -> return 0.

    This matches Python's bool() convention. Users who want SystemExit must
    UNSET the variable (or set it to empty string), not set it to '0'.
    """
    monkeypatch.setenv('GTX_NO_EXIT', '0')
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # No SystemExit
    ret = wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0


def test_wjoin_no_exit_empty_string_falls_back_to_raise(monkeypatch):
    """Empty string is falsy -> raises SystemExit (D-07 read-every-call)."""
    monkeypatch.setenv('GTX_NO_EXIT', '')
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    with pytest.raises(SystemExit) as exc_info:
        wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert exc_info.value.code == 0


def test_wjoin_reads_env_each_call(monkeypatch):
    """D-07: env var is read every call, not cached."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()

    # Call 1: unset -> raises
    monkeypatch.delenv('GTX_NO_EXIT', raising=False)
    with pytest.raises(SystemExit):
        wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)

    # Call 2: set -> returns 0 (would fail if cached the unset state)
    monkeypatch.setenv('GTX_NO_EXIT', '1')
    ret = wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0

    # Call 3: unset again -> raises again (would fail if cached truthy state)
    monkeypatch.delenv('GTX_NO_EXIT', raising=False)
    with pytest.raises(SystemExit):
        wjoin_with_exit(npu, proc, MockInsn(), xs1=0, xs2=0)


# ----------------- custom0 funct7=0x03 wjoin_custom0_no_exit -----------------

def test_wjoin_custom0_variant_never_raises_unset(monkeypatch):
    """Research §439: custom0 funct7=0x03 NEVER raises, regardless of env."""
    monkeypatch.delenv('GTX_NO_EXIT', raising=False)
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    ret = wjoin_custom0_no_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0


def test_wjoin_custom0_variant_never_raises_set(monkeypatch):
    """Same: with GTX_NO_EXIT=1, custom0 variant still returns 0."""
    monkeypatch.setenv('GTX_NO_EXIT', '1')
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    ret = wjoin_custom0_no_exit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0
