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
"""ACT op @handler entries -- spike-bound shim layer.

Plan 01 ships an empty module with the docstring documenting the 16 ISS
activation @handlers + 7 format_cvt @handlers + 2 pool @handlers that
Plans 03 + 04 will register.

Plan 03 will append (8 forward+reversed x 2 paths = 16 activations):
  - prelu (0x28/funct3=3, reversed), prelu_i (0x28/funct3=7, reversed)
  - gelu (0x2A/funct3=0, reversed), gelu_i (0x2A/funct3=4, reversed)
  - tanh (0x2C/funct3=0, reversed), tanh_i (0x2C/funct3=4, reversed)
  - sigm (0x2D/funct3=0, reversed), sigm_i (0x2D/funct3=4, reversed)
  - softmax (0x2F/funct3=2, FORWARD), softmax_i (0x2F/funct3=6)
  - esum (0x2F/funct3=1, FORWARD scalar to L0), esum_i (0x2F/funct3=5)
  - relu via firmware DISPATCH_ACT funct7=0x06 (no dedicated funct7 -- Plan 03
    decides whether to register a synthetic L1 entry or rely on dispatch_4mode)

Plan 04 will append (7 cvt directions + 2 pool):
  - scvt_qh / scvt_hq at funct7=0x20 (sub_op&1 selects direction)
  - scvt_ih / scvt_hi at funct7=0x21
  - scvt_hn at funct7=0x22 (1-direction only -- INT32->FP16)
  - fcvt_sh / fcvt_hs at funct7=0x24
  - fcvt_dh / fcvt_hd at funct7=0x25 (RESEARCH Adjustment 1)
  - pool_m at funct7=0x30 (max), pool_a at funct7=0x31 (avg)

Phase 5 plan 03/04 task 1.
"""
# Intentionally empty in Plan 01. @handler calls land in Plans 03 + 04.
