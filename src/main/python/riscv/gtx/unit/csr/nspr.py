"""NSPR — Nest SPRs (per-NEST scope, [11:10]=01, PIPE range 0x400-0x7FF).

Both PIPE NSPR_* and APB versions of nest-scoped registers (APB_NSPR_*)
live here so the @csr declarations stay co-located with their semantic
scope.
"""
from typing import Dict
from .register import make_csr, bits, BusType, Register

# ===========================================================================
# NSPR registry (scope-local; @csr decorator below is bound to this dict)
# ===========================================================================

NSPR: Dict[str, Register] = {}
csr = make_csr(NSPR)


# -----------------------------------------------------------------------
# 64-bit PIPE Registers: NSPR
# -----------------------------------------------------------------------

@csr(name="NSPR_THREAD_MASK", address=0x400, width=64, rw_type="RW")
class NSPR_THREAD_MASK:
    mask = bits(0, 15)


@csr(name="NSPR_SHARED_MASK", address=0x401, width=64, rw_type="RW")
class NSPR_SHARED_MASK:
    mask = bits(0)


@csr(name="NSPR_TYPE", address=0x402, width=64, rw_type="RW")
class NSPR_TYPE:
    data = bits(0, 1)


@csr(name="NSPR_OP_MODE", address=0x403, width=64, rw_type="RW")
class NSPR_OP_MODE:
    double_buffer  = bits(0)
    load_credit_en = bits(8)
    store_credit_en = bits(9)


@csr(name="NSPR_CLEAR", address=0x700, width=64, rw_type="WO")
class NSPR_CLEAR:
    load_credit      = bits(0)
    store_credit     = bits(1)
    load_credit_err  = bits(2)
    store_credit_err = bits(3)
    smu_err_clr      = bits(4)
    illegal_opcode   = bits(8)


@csr(name="NSPR_NEST_CREDIT", address=0x701, width=64, rw_type="RW")
class NSPR_NEST_CREDIT:
    spu_load_credit_bypass   = bits(0)
    smu_load_credit_bypass   = bits(1)
    plan_load_credit_bypass  = bits(2)
    spu_store_credit_bypass  = bits(4)
    smu_store_credit_bypass  = bits(5)
    plan_store_credit_bypass = bits(6)


@csr(name="NSPR_MCAST_FAST_MODE", address=0x710, width=64, rw_type="RW")
class NSPR_MCAST_FAST_MODE:
    mcast_fast_mode = bits(0)


@csr(name="NSPR_TMU_STATUS", address=0x750, width=64, rw_type="RO")
class NSPR_TMU_STATUS:
    tmu_tpe_state  = bits(0, 3)
    spu_fifo_empty = bits(32, 47)
    smu_fifo_empty = bits(48)


@csr(name="NSPR_SDLE_STATUS", address=0x780, width=64, rw_type="RO")
class NSPR_SDLE_STATUS:
    sdle_busy = bits(0)
    smu_mode  = bits(1, 4)


@csr(name="NSPR_SMU_DEBUG", address=0x781, width=64, rw_type="RO")
class NSPR_SMU_DEBUG:
    smu_read_pause    = bits(0)
    smu_write_pause   = bits(1)
    smu_err_axi_read  = bits(2)
    smu_err_axi_write = bits(3)
    smu_err_idim      = bits(4)
    smu_err_izero     = bits(5)


@csr(name="NSPR_CREDIT_COUNT", address=0x782, width=64, rw_type="RO")
class NSPR_CREDIT_COUNT:
    load  = bits(0, 15)
    store = bits(16, 31)


@csr(name="NSPR_CREDIT_ERROR", address=0x783, width=64, rw_type="RO")
class NSPR_CREDIT_ERROR:
    load  = bits(0, 15)
    store = bits(16, 31)

# -----------------------------------------------------------------------
# 32-bit APB Registers: NSPR (moved from gspr.py — semantic scope match)
# -----------------------------------------------------------------------

@csr(name="APB_NSPR_THREAD_MASK", address=0x700, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_THREAD_MASK:
    mask = bits(0, 15)


@csr(name="APB_NSPR_SHARED_MASK", address=0x704, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_SHARED_MASK:
    mask = bits(0)


@csr(name="APB_NSPR_TYPE", address=0x708, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_TYPE:
    data = bits(0, 1) # 2'h0: fp8, 2'h1: fp16, 2'h2:int8


@csr(name="APB_NSPR_OP_MODE", address=0x70C, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_OP_MODE:
    double_buffer   = bits(0) # 1'b0
    load_credit_en  = bits(8) #1'b1
    store_credit_en = bits(9) #1'b1


@csr(name="APB_NSPR_CLEAR", address=0x800, width=32, rw_type="WO", bus_type=BusType.APB)
class APB_NSPR_CLEAR:
    load_credit      = bits(0) # 1'b0
    store_credit     = bits(1) # 1'b0
    load_credit_err  = bits(2) # 1'b0
    store_credit_err = bits(3) # 1'b0
    smu_err_clr      = bits(4) # 1'b0
    illegal_opcode   = bits(8) # 1'b0


@csr(name="APB_NSPR_NEST_CREDIT", address=0x804, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_NEST_CREDIT:
    spu_load_credit_bypass   = bits(0) # 1'b1
    smu_load_credit_bypass   = bits(1) # 1'b0
    plan_load_credit_bypass  = bits(2) # 1'b1
    spu_store_credit_bypass  = bits(4) # 1'b0
    smu_store_credit_bypass  = bits(5) # 1'b1
    plan_store_credit_bypass = bits(6) # 1'b1


@csr(name="APB_NSPR_MCAST_FAST_MODE", address=0x810, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_NSPR_MCAST_FAST_MODE:
    mcast_fast_mode = bits(0) # 1'b0


@csr(name="APB_NSPR_TMU_STATUS_L", address=0x850, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_TMU_STATUS_L:
    tmu_tpe_state = bits(0, 3)


@csr(name="APB_NSPR_TMU_STATUS_H", address=0x854, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_TMU_STATUS_H:
    spu_fifo_empty = bits(0, 15)
    smu_fifo_empty = bits(16)


@csr(name="APB_NSPR_SDLE_STATUS", address=0x900, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_SDLE_STATUS:
    sdle_busy = bits(0)
    smu_mode  = bits(1, 4)


@csr(name="APB_NSPR_SMU_DEBUG", address=0x904, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_SMU_DEBUG:
    smu_read_pause    = bits(0)
    smu_write_pause   = bits(1)
    smu_err_axi_read  = bits(2)
    smu_err_axi_write = bits(3)
    smu_err_idim      = bits(4)
    smu_err_izero     = bits(5)


@csr(name="APB_NSPR_CREDIT_COUNT_LOAD", address=0x910, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_CREDIT_COUNT_LOAD:
    load = bits(0, 15)


@csr(name="APB_NSPR_CREDIT_COUNT_STORE", address=0x914, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_CREDIT_COUNT_STORE:
    store = bits(16, 31)


@csr(name="APB_NSPR_CREDIT_ERROR_LOAD", address=0x918, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_CREDIT_ERROR_LOAD:
    load = bits(0, 15)


@csr(name="APB_NSPR_CREDIT_ERROR_STORE", address=0x91C, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSPR_CREDIT_ERROR_STORE:
    store = bits(16, 31)
