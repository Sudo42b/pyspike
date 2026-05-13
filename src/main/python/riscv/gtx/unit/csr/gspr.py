"""GSPR — Global SPRs (address scope [11:10]=00, range 0x000-0x3FF for PIPE).

PIPE registers live in this scope. APB counterparts (with their own flat
APB address space) are also kept here when their *semantic* scope is
global; APB versions of nest-scoped registers belong in nspr.py.
"""
from typing import Dict
from .register import make_csr, bits, BusType, Register

# ===========================================================================
# GSPR registry (scope-local; @csr decorator below is bound to this dict)
# ===========================================================================

GSPR: Dict[str, Register] = {}
csr = make_csr(GSPR)


# ===========================================================================
# CSR declarations
# ===========================================================================
# Order mirrors the original GSPR dict (PIPE block first, then APB block).
# Class names are UPPERCASE_SNAKE; registry keys match.

# -----------------------------------------------------------------------
# 64-bit PIPE Registers
# -----------------------------------------------------------------------

@csr(name="STACK_INFO", address=0x010, width=64, rw_type="RW")
class STACK_INFO:
    pointer = bits(0, 37)
    size    = bits(48, 63)


@csr(name="STACK_SAVE", address=0x011, width=64, rw_type="RW")
class STACK_SAVE:
    addr               = bits(0, 35)
    stack_recovery_en  = bits(48)


@csr(name="CORE_IMEM_OFFSET", address=0x280, width=9, rw_type="RW")
class CORE_IMEM_OFFSET:
    offset = bits(0, 8)


@csr(name="CORE_DMEM_OFFSET", address=0x281, width=9, rw_type="RW")
class CORE_DMEM_OFFSET:
    offset = bits(0, 8)


@csr(name="GTX_NSU_OFFSET", address=0x282, width=9, rw_type="RW")
class GTX_NSU_OFFSET:
    offset = bits(0, 8)


@csr(name="GTX_NEST0_OFFSET", address=0x283, width=9, rw_type="RW")
class GTX_NEST0_OFFSET:
    offset = bits(0, 8)


@csr(name="GTX_NEST1_OFFSET", address=0x284, width=9, rw_type="RW")
class GTX_NEST1_OFFSET:
    offset = bits(0, 8)


@csr(name="GTX_NEST2_OFFSET", address=0x285, width=9, rw_type="RW")
class GTX_NEST2_OFFSET:
    offset = bits(0, 8)


@csr(name="GTX_NEST3_OFFSET", address=0x286, width=9, rw_type="RW")
class GTX_NEST3_OFFSET:
    offset = bits(0, 8)


@csr(name="GSPR_CLEAR", address=0x320, width=1, rw_type="WO")
class GSPR_CLEAR:
    gdle_clear = bits(0)


@csr(name="GDLE_STATUS", address=0x330, width=12, rw_type="RO")
class GDLE_STATUS:
    gdle_busy        = bits(0)
    gdle_mode_2      = bits(1, 2)
    gdle_mode        = bits(3)
    gdle_fifo_empty  = bits(4, 7)
    gdle_fifo_full   = bits(8, 11)


@csr(name="GDLE_ADDR_READ", address=0x331, width=32, rw_type="RO")
class GDLE_ADDR_READ:
    gdle_read_start = bits(0, 15)
    gdle_read_final = bits(16, 31)


@csr(name="GDLE_ADDR_WRITE_S", address=0x332, width=64, rw_type="RO")
class GDLE_ADDR_WRITE_S:
    gdle_write_start = bits(0, 63)


@csr(name="GDLE_ADDR_WRITE_F", address=0x333, width=64, rw_type="RO")
class GDLE_ADDR_WRITE_F:
    gdle_write_final = bits(0, 63)


@csr(name="GDLE_DEBUG", address=0x334, width=13, rw_type="RO")
class GDLE_DEBUG:
    gdle_err_rdata_fifo = bits(0, 1)
    gdle_err_wdata_fifo = bits(2, 9)
    gdle_err_cfg        = bits(10, 12)


@csr(name="FCVT_STATUS", address=0x340, width=2, rw_type="RO")
class FCVT_STATUS:
    fcvt_state = bits(0, 1)


@csr(name="ICACHE_STATUS", address=0x350, width=64, rw_type="RO")
class ICACHE_STATUS:
    icache_busy           = bits(0)
    icache_state          = bits(1, 3)
    icache_hit_num        = bits(4, 7)
    icache_flush_pending  = bits(8)
    icache_axi_read_addr  = bits(32, 63)


@csr(name="ICACHE_FLUSH", address=0x351, width=1, rw_type="WO")
class ICACHE_FLUSH:
    icache_flush_req = bits(0)


@csr(name="RISCV_TIMER", address=0x360, width=37, rw_type="RW")
class RISCV_TIMER:
    timer_base_addr = bits(0, 36)


@csr(name="CDC_CONTROL", address=0x370, width=1, rw_type="RW")
class CDC_CONTROL:
    cdc_level_set = bits(0)


@csr(name="GLOBAL_FIFO_CLEAR", address=0x371, width=1, rw_type="WO")
class GLOBAL_FIFO_CLEAR:
    fifo_clear = bits(0)


@csr(name="SPU_BUSY", address=0x380, width=64, rw_type="RO")
class SPU_BUSY:
    busy = bits(0, 63)


@csr(name="SMU_BUSY", address=0x381, width=4, rw_type="RO")
class SMU_BUSY:
    busy = bits(0, 3)


@csr(name="NSU_IDE_STATUS", address=0x390, width=6, rw_type="RO")
class NSU_IDE_STATUS:
    ide_con_state    = bits(0, 2)
    ide_dec_state    = bits(3, 4)
    illegal_context  = bits(5)


@csr(name="NSU_MSE_STATUS", address=0x391, width=6, rw_type="RO")
class NSU_MSE_STATUS:
    mse_state      = bits(0, 2)
    mse_run_state  = bits(3, 5)


@csr(name="NSU_MPE_STATUS", address=0x392, width=12, rw_type="RO")
class NSU_MPE_STATUS:
    mpe0_run_state = bits(0, 2)
    mpe1_run_state = bits(3, 5)
    mpe2_run_state = bits(6, 8)
    mpe3_run_state = bits(9, 11)


@csr(name="NSU_UCODE_MODE", address=0x393, width=5, rw_type="RW")
class NSU_UCODE_MODE:
    mse_ucode_fast_mode = bits(0)
    mpe_ucode_fast_mode = bits(1, 4)


@csr(name="NSU_MSE_UCODE_CNT", address=0x394, width=32, rw_type="RO")
class NSU_MSE_UCODE_CNT:
    mse_ucode_count = bits(0, 31)


@csr(name="NSU_MPE0_UCODE_CNT", address=0x395, width=32, rw_type="RO")
class NSU_MPE0_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="NSU_MPE1_UCODE_CNT", address=0x396, width=32, rw_type="RO")
class NSU_MPE1_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="NSU_MPE2_UCODE_CNT", address=0x397, width=64, rw_type="RO")
class NSU_MPE2_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="NSU_MPE3_UCODE_CNT", address=0x398, width=32, rw_type="RO")
class NSU_MPE3_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="INFO", address=0x3F0, width=16, rw_type="RO")
class INFO:
    spu_count  = bits(0, 5)
    nest_count = bits(8, 13)


@csr(name="ID", address=0x3F1, width=32, rw_type="RO")
class ID:
    vendor          = bits(0, 7)
    architecture    = bits(8, 15)
    implementation  = bits(16, 23)
    thread          = bits(24, 31)


# -----------------------------------------------------------------------
# 32-bit APB Registers
# -----------------------------------------------------------------------
# H/W calculate stack start address(sp_addr-sp_size)
@csr(name="APB_STACK_INFO_L", address=0x140, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_STACK_INFO_L:
    pointer = bits(0, 31, value=0x0)


@csr(name="APB_STACK_INFO_H", address=0x144, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_STACK_INFO_H:
    pointer = bits(0, 4)   # APB_STACK_INFO_H(32bit)+5bit임 = 37bit로 사용
    reserved = bits(5, 15)
    size    = bits(16, 31, value=0x0)


@csr(name="APB_STACK_SAVE_L", address=0x148, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_STACK_SAVE_L:
    addrL = bits(0, 31, value=0x0)


@csr(name="APB_STACK_SAVE_H", address=0x14C, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_STACK_SAVE_H:
    addrH              = bits(0, 4) # APB_STACK_SAVE_L(32bit)+5bit임 = 37bit로 사용
    reserved          = bits(5, 15)
    stack_recovery_en  = bits(16, value=0x0) 


@csr(name="APB_CORE_IMEM_OFFSET", address=0x180, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_CORE_IMEM_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_CORE_DMEM_OFFSET", address=0x184, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_CORE_DMEM_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GTX_NSU_OFFSET", address=0x188, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_GTX_NSU_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GTX_NEST0_OFFSET", address=0x18C, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_GTX_NEST0_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GTX_NEST1_OFFSET", address=0x190, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_GTX_NEST1_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GTX_NEST2_OFFSET", address=0x194, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_GTX_NEST2_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GTX_NEST3_OFFSET", address=0x198, width=9, rw_type="RW", bus_type=BusType.APB)
class APB_GTX_NEST3_OFFSET:
    offset = bits(0, 8)


@csr(name="APB_GSPR_CLEAR", address=0x220, width=1, rw_type="WO", bus_type=BusType.APB)
class APB_GSPR_CLEAR:
    gdle_clear = bits(0)


@csr(name="APB_GDLE_STATUS", address=0x230, width=12, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_STATUS:
    gdle_busy        = bits(0)
    gdle_mode_2      = bits(1, 2)
    gdle_mode        = bits(3)
    gdle_fifo_empty  = bits(4, 7)
    gdle_fifo_full   = bits(8, 11)


@csr(name="APB_GDLE_ADDR_READ", address=0x234, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_ADDR_READ:
    gdle_read_start = bits(0, 15)
    gdle_read_final = bits(16, 31)


@csr(name="APB_GDLE_ADDR_WRITE_S_L", address=0x238, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_ADDR_WRITE_S_L:
    gdle_write_start_L = bits(0, 31)


@csr(name="APB_GDLE_ADDR_WRITE_S_H", address=0x23C, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_ADDR_WRITE_S_H:
    gdle_write_start_H = bits(0, 31)


@csr(name="APB_GDLE_ADDR_WRITE_F_L", address=0x240, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_ADDR_WRITE_F_L:
    gdle_write_final_L = bits(0, 31)


@csr(name="APB_GDLE_ADDR_WRITE_F_H", address=0x244, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_ADDR_WRITE_F_H:
    gdle_write_final_H = bits(0, 31)


@csr(name="APB_GDLE_DEBUG", address=0x248, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_GDLE_DEBUG:
    gdle_err_rdata_fifo = bits(0, 1)
    gdle_err_wdata_fifo = bits(2, 9)
    gdle_err_cfg        = bits(10, 12)


@csr(name="APB_FCVT_STATUS", address=0x340, width=1, rw_type="RO", bus_type=BusType.APB)
class APB_FCVT_STATUS:
    fcvt_state = bits(0, 1)


@csr(name="APB_ICACHE_STATUS_L", address=0x350, width=13, rw_type="RO", bus_type=BusType.APB)
class APB_ICACHE_STATUS_L:
    icache_busy           = bits(0)
    icache_state          = bits(1, 3)
    icache_hit_num        = bits(4, 7)
    icache_flush_pending  = bits(8)


@csr(name="APB_ICACHE_STATUS_H", address=0x354, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_ICACHE_STATUS_H:
    icache_axi_read_addr = bits(0, 31)


@csr(name="APB_ICACHE_FLUSH", address=0x358, width=1, rw_type="WO", bus_type=BusType.APB)
class APB_ICACHE_FLUSH:
    # single pulse
    icache_flush_req = bits(0)


@csr(name="APB_RISCV_TIMER_L", address=0x360, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_RISCV_TIMER_L:
    timer_base_addr_L = bits(0, 31)


@csr(name="APB_RISCV_TIMER_H", address=0x364, width=5, rw_type="RW", bus_type=BusType.APB)
class APB_RISCV_TIMER_H:
    timer_base_addr_H = bits(0, 4)


@csr(name="APB_CDC_CONTROL", address=0x370, width=1, rw_type="RW", bus_type=BusType.APB)
class APB_CDC_CONTROL:
    cdc_level_set = bits(0)


@csr(name="APB_GLOBAL_FIFO_CLEAR", address=0x374, width=1, rw_type="WO", bus_type=BusType.APB)
class APB_GLOBAL_FIFO_CLEAR:
    # single pulse
    fifo_clear = bits(0)


@csr(name="APB_SPU_BUSY_L", address=0x400, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_SPU_BUSY_L:
    busy_L = bits(0, 31)


@csr(name="APB_SPU_BUSY_H", address=0x404, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_SPU_BUSY_H:
    busy_H = bits(0, 31)


@csr(name="APB_SMU_BUSY", address=0x408, width=4, rw_type="RO", bus_type=BusType.APB)
class APB_SMU_BUSY:
    busy = bits(0, 3)


@csr(name="APB_NSU_IDE_STATUS", address=0x410, width=6, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_IDE_STATUS:
    ide_con_state    = bits(0, 2)
    ide_dec_state    = bits(3, 4)
    illegal_context  = bits(5)


@csr(name="APB_NSU_MSE_STATUS", address=0x414, width=6, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MSE_STATUS:
    mse_state      = bits(0, 2)
    mse_run_state  = bits(3, 5)


@csr(name="APB_NSU_MPE_STATUS", address=0x418, width=12, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MPE_STATUS:
    mpe0_run_state = bits(0, 2)
    mpe1_run_state = bits(3, 5)
    mpe2_run_state = bits(6, 8)
    mpe3_run_state = bits(9, 11)


@csr(name="APB_NSU_UCODE_MODE", address=0x41C, width=5, rw_type="RW", bus_type=BusType.APB)
class APB_NSU_UCODE_MODE:
    mse_ucode_fast_mode = bits(0)
    mpe_ucode_fast_mode = bits(1, 4)


@csr(name="APB_NSU_MSE_UCODE_CNT", address=0x420, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MSE_UCODE_CNT:
    mse_ucode_count = bits(0, 31)


@csr(name="APB_NSU_MPE0_UCODE_CNT", address=0x424, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MPE0_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="APB_NSU_MPE1_UCODE_CNT", address=0x428, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MPE1_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="APB_NSU_MPE2_UCODE_CNT", address=0x42C, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MPE2_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="APB_NSU_MPE3_UCODE_CNT", address=0x430, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_NSU_MPE3_UCODE_CNT:
    mpe_ucode_count = bits(0, 31)


@csr(name="APB_INFO", address=0x600, width=14, rw_type="RO", bus_type=BusType.APB)
class APB_INFO:
    spu  = bits(0, 5)
    reserved = bits(6, 7)
    nest = bits(8, 13)


@csr(name="APB_ID", address=0x610, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_ID:
    vendor          = bits(0, 7)
    architecture    = bits(8, 15)
    implementation  = bits(16, 23)
    thread          = bits(24, 31)
