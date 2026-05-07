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
"""VEC op @handler entries -- spike-bound shim layer (D-04 mirror of ops/mm.py).

Plan 01 ships an empty module with the docstring documenting the 8 SASMD-VS +
8 SASMD-VV @handlers + 4 CLAMP-family @handlers + math/sign/round @handlers
that Plan 02 wave 1b will register. NO @handler calls in plan 01 -- that lets
the import be a pure no-op (no registry mutations) until Plan 02 lands.

Plan 02 will append:
  - 8 SASMD-family @handlers at funct7=0x10 (add_vs, sub_vs, mul_vs, div_vs at
    funct3=0..3; add_is, sub_is, mul_is, div_is at funct3=4..7)
  - 8 SASMD-vector @handlers at funct7=0x18 (add_vv, ... + add_ii, ...)
  - 1 vsum + 1 dot @handler at funct7=0x13 (funct3=0/1)
  - 4 CLAMP-family @handlers at funct7=0x1F (clamp_min_v/clamp_max_v/accum_v/arange_v
    at funct3=0..3; bitwise variants at funct3=4..7 are Plan 02 too)
  - sqrt/exp/log/abs/neg/sgn/step/ceil/floor/trunc/rne @handlers at funct7=0x1C/0x1D/0x1E

Phase 5 plan 02 task 1.
"""
# Intentionally empty in Plan 01. @handler calls land in Plan 02.
