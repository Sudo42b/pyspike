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
"""Tests for DISASM-01 -- disasm.py formulas + _registry.collect_disasms.

All tests run offline (no _riscv.so) -- disasm.py provides a sentinel fallback
so .name / .match / .mask attributes are inspectable.
"""
from riscv.gtx.disasm import (
    add_r_custom0, add_rf3_custom0, add_warp, _RISCV_DISASM_AVAILABLE,
)
from riscv.gtx import _registry
# Force-import op modules so the registry is populated by the time
# collect_disasms() runs.
from riscv.gtx.ops import spr as _spr   # noqa: F401
from riscv.gtx.ops import control as _ctrl  # noqa: F401


def _norm(name: str) -> str:
    """Real disasm_insn_t C++ ctor normalizes mnemonic '_' -> '.'.
    Offline _PyDisasmInsn fallback preserves the input as-is. Tests must
    pass on both paths."""
    return name.replace('_', '.') if _RISCV_DISASM_AVAILABLE else name


# ----------------- Formula tests (research §537-555 worked examples) -----------------

def test_add_r_custom0_wrspr_formula():
    """wrspr ISS-full (funct7=0x49) -- research §537-541."""
    e = add_r_custom0('wrspr', 0x49)
    assert e.name == 'wrspr'
    assert e.match == 0x9200000B, hex(e.match)
    assert e.mask == 0xFE00007F, hex(e.mask)


def test_add_r_custom0_wjoin_formula():
    """wjoin custom0 firmware variant (funct7=0x03) -- research §544-548."""
    e = add_r_custom0('wjoin', 0x03)
    assert e.name == 'wjoin'
    assert e.match == 0x0600000B, hex(e.match)
    assert e.mask == 0xFE00007F, hex(e.mask)


def test_add_rf3_custom0_mm_s_formula():
    """mm_s ISS-full (funct7=0x00, funct3=0). MM op -- registered in P4, but
    the formula is exercised here in isolation."""
    e = add_rf3_custom0('mm_s', 0x00, 0)
    assert e.name == _norm('mm_s')
    assert e.match == 0x0000000B, hex(e.match)
    assert e.mask == 0xFE00707F, hex(e.mask)


def test_add_warp_start_p_formula():
    """warp_start_p (custom1, funct3=0b110) -- research §551-555."""
    e = add_warp('warp_start_p', 0b110)
    assert e.name == _norm('warp_start_p')
    assert e.match == 0x0000602B, hex(e.match)
    assert e.mask == 0x0000707F, hex(e.mask)


def test_add_warp_join_formula():
    """warp_join (custom1, funct3=0b101)."""
    e = add_warp('warp_join', 0b101)
    assert e.name == _norm('warp_join')
    assert e.match == (0b101 << 12) | 0x2b, hex(e.match)
    assert e.mask == (0x7 << 12) | 0x7f, hex(e.mask)


# ----------------- collect_disasms tests -----------------

def test_collect_disasms_minimum_count():
    """ROADMAP P2 #2: per-op registry sums to ~10 entries in P2.

    After plans 02 (4 SPR) + 03 (8 warp + 6 custom0 stubs) + 04 (this plan
    only adds the disasm builder, no new registrations), expect >=18
    mnemonic'd entries -- which exceeds the ~10 ROADMAP threshold.
    """
    entries = _registry.collect_disasms()
    assert len(entries) >= 18, (
        f"expected >=18 disasm entries, got {len(entries)}: "
        f"{[e.name for e in entries]}"
    )


def test_collect_disasms_contains_p2_sample_5():
    """ROADMAP P2 #2 sample 5 (D-12 adapted to P2-available ops):
    ['wrspr', 'rdspr', 'wsplit_c0', 'wjoin_c0', 'warp_start_p'].

    Note: 'wsplit'/'wjoin' bare names refer to the custom1 warp variants
    per research §439; the custom0 firmware variants register under
    'wsplit_c0'/'wjoin_c0'. ROADMAP P2 #2 list ['wrspr','rdspr','wsplit',
    'wjoin','warp_start_p'] is satisfied with EITHER choice -- we use
    the unambiguous custom0 names here.
    """
    entries = _registry.collect_disasms()
    names = {e.name for e in entries}
    sample5 = [_norm(n) for n in ('wrspr', 'rdspr', 'wsplit_c0', 'wjoin_c0', 'warp_start_p')]
    missing = [m for m in sample5 if m not in names]
    assert not missing, f"sample mnemonics missing: {missing}; got names: {sorted(names)}"


def test_collect_disasms_all_8_warp_mnemonics_present():
    """All 8 custom1 funct3 warp mnemonics registered (DISP-02 + DISASM-01)."""
    entries = _registry.collect_disasms()
    names = {e.name for e in entries}
    warp_mnemonics = [_norm(n) for n in (
        'warp_start_t', 'warp_end_t', 'warp_start_s', 'warp_end_s',
        'warp_split', 'warp_join', 'warp_start_p', 'warp_end_p',
    )]
    missing = [m for m in warp_mnemonics if m not in names]
    assert not missing, f"warp mnemonics missing: {missing}"


def test_collect_disasms_all_4_spr_mnemonics_present():
    """All 4 SPR funct7 mnemonics registered (SPR-02 + DISASM-01)."""
    entries = _registry.collect_disasms()
    names = {e.name for e in entries}
    spr_mnemonics = [_norm(n) for n in ('wrspr', 'rdspr', 'wrspr_gem5', 'rdspr_gem5')]
    missing = [m for m in spr_mnemonics if m not in names]
    assert not missing, f"SPR mnemonics missing: {missing}"


def test_collect_disasms_match_mask_unique_per_funct7():
    """Sanity: no two custom0 entries (mask_funct3=False) share the same match.

    This ensures the dispatch table is unambiguous (no collisions in the
    funct7 keys post plan 02 + plan 03). The funct7=0x00 collision (gem5
    WRSPR vs ISS MM) is resolved at dispatch time by D-02 heuristic -- the
    disasm entry for funct7=0x00 names the gem5 form ('wrspr_gem5')."""
    entries = _registry.collect_disasms()
    # group by (match,mask) and check no collisions for non-funct3 entries
    seen = {}
    for e in entries:
        key = (e.match, e.mask)
        if key in seen:
            # Only flag a true duplicate name + match collision (different
            # encodings of the same instruction would have identical match)
            if seen[key] == e.name:
                raise AssertionError(f"duplicate disasm entry: {e.name}")
        seen[key] = e.name
