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
"""GTX RoCC instruction encoding constants.

Phase 1 scope: funct7 constants (used by P2 dispatch + disasm).
Full disasm_insn_t table moves to disasm.py in Phase 2 (D-13 scope).
"""
# RoCC funct7 -- gem5 simplified (operand staging via GSPR):
GTX_F7_WRSPR: int = 0x00       # WRSPR (gem5) / MM ISS-full (rs1!=0 disambiguation in P4)
GTX_F7_RDSPR: int = 0x01
GTX_F7_WSPLIT: int = 0x02      # custom1 (warp split)
GTX_F7_WJOIN: int = 0x03       # custom1 (warp join -- exit semantics in P2)
GTX_F7_DISPATCH_MM: int = 0x04
GTX_F7_DISPATCH_VEC: int = 0x05
GTX_F7_DISPATCH_ACT: int = 0x06
GTX_F7_DISPATCH_DMA: int = 0x07

# ISS full (per-op funct7) -- selected; full table P2:
# GTX_F7_MM = 0x00 (collides with WRSPR; resolved by insn.rs1 != 0 -- P4)
# GTX_F7_MMC = 0x01
# GTX_F7_DMA_LOAD = 0x40
# GTX_F7_OPSET = 0x4A
# (...remaining 70+ constants in Phase 2)
