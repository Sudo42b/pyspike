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

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu.h:266-353.
Phase 2 (D-11): full funct7 set + custom1 funct3 + opcode constants.
"""
# ----- RoCC custom opcodes -----
CUSTOM0_OPCODE: int = 0x0b      # custom-0
CUSTOM1_OPCODE: int = 0x2b      # custom-1

# ----- gem5-simplified funct7 (custom0) -- gtx_npu.h:266-273 -----
GTX_F7_WRSPR: int = 0x00        # WRSPR (gem5) / MM ISS-full (rs1!=0 -> P4 MM)
GTX_F7_RDSPR: int = 0x01        # RDSPR (gem5) / MMC ISS-full (rs1!=0 -> P4 MMC)
GTX_F7_WSPLIT: int = 0x02
GTX_F7_WJOIN: int = 0x03        # WJOIN custom0 firmware variant (no exit)
GTX_F7_DISPATCH_MM: int = 0x04
GTX_F7_DISPATCH_VEC: int = 0x05
GTX_F7_DISPATCH_ACT: int = 0x06
GTX_F7_DISPATCH_DMA: int = 0x07

# ----- ISS-full funct7 (custom0) -- gtx_npu.h:276-282 (P2 subset) -----
GTX_ISS_F7_RDSPR_ISS: int = 0x48        # 0b1001000
GTX_ISS_F7_WRSPR_ISS: int = 0x49        # 0b1001001
GTX_ISS_F7_OPSET: int = 0x4A            # P3 (DMA staging)
GTX_ISS_F7_DEBUG_WR: int = 0x7D
GTX_ISS_F7_DEBUG_RD: int = 0x7E

# ----- custom1 funct3 (warp control) -- gtx_npu_custom1.cc:21-26 -----
WARP_F3_START_T: int = 0b000
WARP_F3_END_T: int = 0b001
WARP_F3_START_S: int = 0b010
WARP_F3_END_S: int = 0b011
WARP_F3_SPLIT: int = 0b100
WARP_F3_JOIN: int = 0b101
WARP_F3_START_P: int = 0b110
WARP_F3_END_P: int = 0b111

# ----- Mode constants -- gtx_npu.h:289-292 -----
GTX_OP_MM: int = 0
GTX_OP_VECTOR: int = 1
GTX_OP_ACTIVATION: int = 2
GTX_OP_DMA: int = 3

# ----- Loop-control GSPR addresses (used by SPR router in plan 02) -----
# Provided here for cross-module use; addresses 0x100..0x105.
GSPR_STARTP: int = 0x100
GSPR_ENDP: int = 0x101
GSPR_STARTS: int = 0x102
GSPR_ENDS: int = 0x103
GSPR_STARTT: int = 0x104
GSPR_ENDT: int = 0x105
