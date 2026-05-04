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
"""Warp/control ops -- start_p/end_p/start_t/end_t/wsplit/wjoin.

Plan 01: stub. Plan 03 implements all 8 custom1 funct3 handlers
+ the custom0 wjoin/wsplit/dispatch_* stubs.
"""
# Plan 03 will register:
# @handler(kind='custom1', funct3=0b110, mnemonic='warp_start_p')
# @handler(kind='custom1', funct3=0b111, mnemonic='warp_end_p')
# @handler(kind='custom1', funct3=0b000, mnemonic='warp_start_t')
# @handler(kind='custom1', funct3=0b001, mnemonic='warp_end_t')
# @handler(kind='custom1', funct3=0b010, mnemonic='warp_start_s')   # NOP P2
# @handler(kind='custom1', funct3=0b011, mnemonic='warp_end_s')     # NOP P2
# @handler(kind='custom1', funct3=0b100, mnemonic='warp_split')
# @handler(kind='custom1', funct3=0b101, mnemonic='warp_join')      # SystemExit
# @handler(kind='custom0', funct7=0x02, mnemonic='wsplit')
# @handler(kind='custom0', funct7=0x03, mnemonic='wjoin')           # NO exit
# @handler(kind='custom0', funct7=0x04, mnemonic='dispatch_mm')     # P3+ stub
# ...
