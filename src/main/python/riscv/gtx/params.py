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
"""Hardware parameter constants -- direct port of vendor/gtx_cpp_reference/gtx/gtx_params.h.

Naming follows the C++ macro convention verbatim (per CONTEXT.md Claude's Discretion).
These values are referenced by tests/gtx/test_memory_layout.py and by Phase 2-5 op handlers.
"""
# NEST x SPU topology
GTX_NEST_NUM: int = 4
GTX_SPU_NUM: int = 16          # SPUs per NEST
GTX_SPUS_PER_NEST: int = GTX_SPU_NUM   # alias for clarity

# Memory sizes (bytes)
GTX_L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
GTX_L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
GTX_L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

# DDR (D-02: capped by GTX_DDR_SIZE env var; default below)
GTX_DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

# DDR I/O (D-03)
GTX_DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for GTX_DDR_REVERSED reversal

# DDR base physical address (firmware GTX_MAIN_BASE -- gtx_params.h:24)
GTX_DDR_BASE: int = 0x370000000

# SPR address ranges (D-11)
GSPR_BASE: int = 0x000
GSPR_END: int = 0x3FF
NSPR_BASE: int = 0x400
NSPR_END: int = 0x7FF
LSPR_BASE: int = 0x800
LSPR_END: int = 0xBFF
