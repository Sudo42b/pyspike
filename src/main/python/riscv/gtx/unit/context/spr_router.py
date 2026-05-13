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
"""SPR routing -- port of vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc.

GSPR (0x000-0x3FF) flat single-instance.
NSPR (0x400-0x7FF) per-NEST -- routed by tmu_id when is_ploop, else NEST 0.
LSPR (0x800-0xBFF) per-(NEST,SPU) -- routed by (tmu_id, curr_id) when is_tloop,
broadcast across SPUs when is_ploop, fallback to (0,0) otherwise.

Loop-control GSPR addresses 0x100..0x105 trigger startp/endp/starts/ends/startt/endt
side-effect handlers from ops.control (lazy imported to avoid plan 02 -> plan 03
circular import; plan 03 provides the _do_* helpers).
"""
from ...config_params import (GSPR_BASE, GSPR_END, NSPR_BASE, NSPR_END,
                     LSPR_BASE, LSPR_END, GTX_NEST_NUM, GTX_SPU_NUM)
# from ..ins.encoding import (GSPR_STARTP, GSPR_ENDP, GSPR_STARTS,
#                        GSPR_ENDS, GSPR_STARTT, GSPR_ENDT)


def _in_range(addr: int, base: int, end: int) -> bool:
    return base <= addr <= end


def wr_spr(npu, addr: int, value: int) -> None:
    """Write SPR. Port of gtx_npu_t::wr_spr (gtx_npu_spr.cc:16-78)."""
    addr &= 0xFFFF
    # Loop control side-effects -- lazy import to avoid
    # plan 02 / plan 03 circular dependency; plan 03 fills ops.control._do_*.
    # if addr == GSPR_STARTP:
    #     from . import control as _ctrl
    #     _ctrl._do_startp(npu, value, 0)
    #     return
    # if addr == GSPR_ENDP:
    #     from . import control as _ctrl
    #     _ctrl._do_endp(npu, value, 0)
    #     return
    # if addr == GSPR_STARTS:
    #     from . import control as _ctrl
    #     _ctrl._do_starts(npu, value, 0)
    #     return
    # if addr == GSPR_ENDS:
    #     from . import control as _ctrl
    #     _ctrl._do_ends(npu, value, 0)
    #     return
    # if addr == GSPR_STARTT:
    #     from . import control as _ctrl
    #     _ctrl._do_startt(npu, value, 0)
    #     return
    # if addr == GSPR_ENDT:
    #     from . import control as _ctrl
    #     _ctrl._do_endt(npu, value, 0)
    #     return

    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM
                and npu.warp.curr_id < GTX_SPU_NUM):
            npu.lspr[npu.warp.tmu_id][npu.warp.curr_id][addr] = value
        elif npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            # P-loop: same value into every SPU's LSPR within the active
            # nest. C++ vendor writes each SPU RF separately
            # (gtx_npu_spr.cc:24-25) — semantically equivalent to the
            # docstring's "broadcast across SPUs in the NEST", so a
            # tight per-SPU loop matches both.
            nest_lsprs = npu.lspr[npu.warp.tmu_id]
            for spu_rf in nest_lsprs:
                spu_rf[addr] = value
        else:
            npu.lspr[0][0][addr] = value   # fallback NEST 0, SPU 0
        return

    if _in_range(addr, NSPR_BASE, NSPR_END):
        if npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            npu.nspr[npu.warp.tmu_id][addr] = value
        else:
            npu.nspr[0][addr] = value
        return

    if _in_range(addr, GSPR_BASE, GSPR_END):
        npu.gspr[addr] = value
        return

    # Out-of-range: silently drop (matches C++ behavior -- log only).


def rd_spr(npu, addr: int) -> int:
    """Read SPR. Port of gtx_npu_t::rd_spr (gtx_npu_spr.cc:83-107)."""
    addr &= 0xFFFF

    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM
                and npu.warp.curr_id < GTX_SPU_NUM):
            return npu.lspr[npu.warp.tmu_id][npu.warp.curr_id].get(addr, 0)
        return npu.lspr[0][0].get(addr, 0)

    if _in_range(addr, NSPR_BASE, NSPR_END):
        nid = npu.warp.tmu_id if (npu.warp.is_ploop and
                                  npu.warp.tmu_id < GTX_NEST_NUM) else 0
        return npu.nspr[nid].get(addr, 0)

    if _in_range(addr, GSPR_BASE, GSPR_END):
        return npu.gspr.get(addr, 0)

    return 0
