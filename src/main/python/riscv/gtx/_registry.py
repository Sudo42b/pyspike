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
"""Per-op decorator registry (D-13) -- internal API for op modules.

Op modules use @handler(...) at module-load time to register a function
plus optional disasm entry. dispatch.build_custom0_table / build_custom1_table
consume the registry at GtxNpu.__init__ time.
"""
from typing import Callable, Dict, List, Optional

_HANDLER_REGISTRY: List[dict] = []


def handler(*, kind: str, funct7: Optional[int] = None,
            funct3: Optional[int] = None, mnemonic: Optional[str] = None,
            mask_funct3: bool = False):
    """Register a handler function + (optional) disasm entry.

    kind: 'custom0' or 'custom1'
    funct7: required for custom0 dispatch key
    funct3: required for custom1 dispatch key (and custom0 mask_funct3 case)
    mnemonic: if provided, contributes a disasm_insn_t entry (built in plan 04)
    mask_funct3: custom0 with funct3 sub-variant (P4 MM only -- false in P2)
    """
    if kind not in ("custom0", "custom1"):
        raise ValueError(f"@handler kind must be 'custom0' or 'custom1', got {kind!r}")

    def decorator(fn: Callable) -> Callable:
        _HANDLER_REGISTRY.append({
            "fn": fn,
            "kind": kind,
            "funct7": funct7,
            "funct3": funct3,
            "mnemonic": mnemonic,
            "mask_funct3": mask_funct3,
        })
        return fn
    return decorator


def collect_for_kind(kind: str):
    """Build dispatch dict for a given kind.

    For 'custom0': returns 2-level Dict[int, Dict[Optional[int], Callable]].
        Outer key = funct7. Inner key = funct3 (when mask_funct3=True) or None
        (when mask_funct3=False). The sentinel None key keeps P2 backwards
        compatibility: the dispatcher tries None first, then synthesized funct3
        if None is not present (see GtxNpu.custom0).
    For 'custom1': returns flat Dict[int, Callable] keyed by funct3 (unchanged).
    """
    if kind == "custom1":
        out_flat: Dict[int, Callable] = {}
        for entry in _HANDLER_REGISTRY:
            if entry["kind"] != "custom1":
                continue
            key = entry["funct3"]
            if key is None:
                raise ValueError(
                    f"@handler custom1 missing funct3: mnemonic={entry['mnemonic']}"
                )
            out_flat[key] = entry["fn"]
        return out_flat

    if kind != "custom0":
        raise ValueError(f"unknown kind: {kind!r}")

    # 2-level for custom0: dict[funct7, dict[Optional[int], Callable]]
    out_2level: Dict[int, Dict] = {}
    for entry in _HANDLER_REGISTRY:
        if entry["kind"] != "custom0":
            continue
        funct7 = entry["funct7"]
        if funct7 is None:
            raise ValueError(
                f"@handler custom0 missing funct7: mnemonic={entry['mnemonic']}"
            )
        inner_key: Optional[int] = (
            entry["funct3"] if entry.get("mask_funct3") else None
        )
        sub = out_2level.setdefault(funct7, {})
        if inner_key in sub:
            raise ValueError(
                f"duplicate handler: funct7=0x{funct7:02x} funct3={inner_key}"
            )
        sub[inner_key] = entry["fn"]
    return out_2level


def collect_disasms() -> list:
    """Build disasm_insn_t list from all entries with mnemonic != None.

    Walks _HANDLER_REGISTRY once, dispatching each mnemonic'd entry to the
    appropriate helper in disasm.py:
        - kind='custom0', mask_funct3=False -> add_r_custom0(mnemonic, funct7)
        - kind='custom0', mask_funct3=True  -> add_rf3_custom0(mnemonic, funct7, funct3)
        - kind='custom1'                    -> add_warp(mnemonic, funct3)

    Lazy-imports add_r_custom0 / add_rf3_custom0 / add_warp from .disasm to
    avoid load-order issues (disasm.py imports encoding.py; _registry.py is
    imported earlier in the dependency chain via dispatch.py).
    """
    from .disasm import add_r_custom0, add_rf3_custom0, add_warp

    out: list = []
    for entry in _HANDLER_REGISTRY:
        mnemonic = entry.get("mnemonic")
        if not mnemonic:
            continue
        kind = entry["kind"]
        if kind == "custom0" and not entry.get("mask_funct3"):
            out.append(add_r_custom0(mnemonic, entry["funct7"]))
        elif kind == "custom0" and entry.get("mask_funct3"):
            out.append(add_rf3_custom0(mnemonic, entry["funct7"], entry["funct3"]))
        elif kind == "custom1":
            out.append(add_warp(mnemonic, entry["funct3"]))
        # else: silently skip (defensive -- handler() already validates kind)
    return out
