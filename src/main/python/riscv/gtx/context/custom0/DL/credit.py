"""Credit counters + checks — port of gtx_npu_custom0.cc:646-694.

Per-(NEST, SPU) load/store credit counters that gate the DMA<->compute
hand-off. In pyspike's eager functional model the *_chk spins always pass
(DMA is instantaneous), so the counters are tracked for vendor 1:1 parity
but never stall control flow.

The load-bearing behavior is ``credit.st.chk``'s deferred-DDR-store flush:
plan-style (WSPLIT) firmware suppresses the ``end.p`` flush
(``control.py:endp`` checks ``wsplit_seen``) and relies on ``credit.st.chk``
to commit the S-loop deferred queue mid-execution. Multi-tile firmware needs
this per-tile flush — the queue snapshots L2 at flush time, so deferring all
tiles to the atexit flush would make every entry read the final tile's L2.
"""
from ...inst_handler import inst_register
from ....config_params import NEST_NUM, SPU_NUM


@inst_register.custom0(name='credit.ld', funct7=0b1010000, funct3=0)
def credit_ld(npu, proc, inst, cxt) -> int:
    """credit.ld (0x50): S-loop DMA-load done → inc all SPUs; T-loop consume → dec."""
    warp = npu.warp
    nest = warp.current_nest if warp.is_ploop else 0
    if nest < NEST_NUM:
        if warp.is_sloop:
            npu._credit_ld[nest, :] += 1
        elif warp.is_tloop and warp.current_spu < SPU_NUM:
            npu._credit_ld[nest, warp.current_spu] -= 1
    return 0


@inst_register.custom0(name='credit.st', funct7=0b1010001, funct3=0)
def credit_st(npu, proc, inst, cxt) -> int:
    """credit.st (0x51): T-loop compute done → inc; S-loop DMA-store consume → dec all."""
    warp = npu.warp
    nest = warp.current_nest if warp.is_ploop else 0
    if nest < NEST_NUM:
        if warp.is_tloop and warp.current_spu < SPU_NUM:
            npu._credit_st[nest, warp.current_spu] += 1
        elif warp.is_sloop:
            npu._credit_st[nest, :] -= 1
    return 0


@inst_register.custom0(name='credit.ld.chk', funct7=0b1010010, funct3=0)
def credit_ld_chk(npu, proc, inst, cxt) -> int:
    """credit.ld.chk (0x52): NOP — spin passes unconditionally (DMA instantaneous)."""
    return 0


@inst_register.custom0(name='credit.st.chk', funct7=0b1010011, funct3=0)
def credit_st_chk(npu, proc, inst, cxt) -> int:
    """credit.st.chk (0x53): S-loop commits the deferred L2→DDR store queue.

    Sole flush trigger for plan-style (WSPLIT) firmware — see module docstring.
    """
    if npu.warp.is_sloop:
        npu.flush_deferred_ddr_stores()
    return 0
