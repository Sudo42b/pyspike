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
"""WarpState -- P/S/T loop state machine, port of gtx_npu_t loop fields.

D-04: bool flags + tmu_id (NEST) + curr_id (SPU/GDMAC).
Reset in reset() and at WJOIN.
"""
from dataclasses import dataclass


@dataclass
class WarpState:
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False  # P3+ DMA paths only
    tmu_id: int = 0   # NEST id selected by start_p
    curr_id: int = 0  # SPU id (T-loop) or GDMAC id (S-loop)
    # P3: process-lifetime sentinel -- set True by WSPLIT, NOT cleared by reset()
    # (matches C++ gtx_npu.h:1251 field initializer; see 03-RESEARCH Pitfall 7)
    wsplit_seen: bool = False

    def reset(self) -> None:
        self.is_ploop = False
        self.is_tloop = False
        self.is_sloop = False
        self.tmu_id = 0
        self.curr_id = 0
        # NOTE: wsplit_seen intentionally NOT reset -- see field comment above.
