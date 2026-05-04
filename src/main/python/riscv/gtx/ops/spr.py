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
"""SPR ops -- WRSPR/RDSPR (gem5 + ISS encodings).

Plan 01: stub. Plan 02 implements wrspr_iss/rdspr_iss/wrspr_gem5/rdspr_gem5.
"""
# Plan 02 will register:
# @handler(kind='custom0', funct7=0x49, mnemonic='wrspr')
# @handler(kind='custom0', funct7=0x48, mnemonic='rdspr')
# @handler(kind='custom0', funct7=0x00, mnemonic='wrspr_gem5')   # collision-aware
# @handler(kind='custom0', funct7=0x01, mnemonic='rdspr_gem5')
