"""LSPR — Local (per-SPU) SPRs ([11:10]=10, PIPE range 0x800-0xBFF).

Includes the 128 SGPR scratch registers (LSPR_SGPR0..127 + their APB
_L/_H halves) generated programmatically.
"""
from typing import Dict
from .register import make_csr, bits, BusType, Register, _declare_generated_csr

# ===========================================================================
# LSPR registry (scope-local; @csr decorator below is bound to this dict)
# ===========================================================================

LSPR: Dict[str, Register] = {}
csr = make_csr(LSPR)


# -----------------------------------------------------------------------
# 64-bit PIPE Registers: LSPR
# -----------------------------------------------------------------------


for sgpr_index in range(128):
    globals()[f"LSPR_SGPR{sgpr_index}"] = _declare_generated_csr(
        name=f"LSPR_SGPR{sgpr_index}",
        address=0x800 + sgpr_index,
        width=64,
        rw_type="RW",
        fields={"gpr": (0, 63)},
        registry=LSPR,
    )

    apb_base = 0xA00 + (sgpr_index * 8)
    globals()[f"APB_LSPR_SGPR{sgpr_index}_L"] = _declare_generated_csr(
        name=f"APB_LSPR_SGPR{sgpr_index}_L",
        address=apb_base,
        width=32,
        rw_type="RW",
        bus_type=BusType.APB,
        fields={"gpr_l": (0, 31)},
        registry=LSPR,
    )
    globals()[f"APB_LSPR_SGPR{sgpr_index}_H"] = _declare_generated_csr(
        name=f"APB_LSPR_SGPR{sgpr_index}_H",
        address=apb_base + 0x4,
        width=32,
        rw_type="RW",
        bus_type=BusType.APB,
        fields={"gpr_h": (0, 31)},
        registry=LSPR,
    )


@csr(name="LSPR_SPM_ADDRA", address=0x900, width=64, rw_type="RW")
class LSPR_SPM_ADDRA:
    start_a = bits(0, 18)


@csr(name="LSPR_SPM_ADDRB", address=0x901, width=64, rw_type="RW")
class LSPR_SPM_ADDRB:
    start_b = bits(0, 18)


@csr(name="LSPR_SPM_ADDRC", address=0x902, width=64, rw_type="RW")
class LSPR_SPM_ADDRC:
    start_c = bits(0, 18)


@csr(name="LSPR_SPM_ADDRR", address=0x903, width=64, rw_type="RW")
class LSPR_SPM_ADDRR:
    start_r = bits(0, 18)


@csr(name="LSPR_SPU_COUNTER_MODE", address=0xA00, width=64, rw_type="RW")
class LSPR_SPU_COUNTER_MODE:
    inst_count_mode = bits(0, 2)


@csr(name="LSPR_SPU_COUNTER_CLEAR", address=0xA01, width=64, rw_type="WO")
class LSPR_SPU_COUNTER_CLEAR:
    inst_count_clr = bits(0)


@csr(name="LSPR_SPU_COUNTER", address=0xA02, width=64, rw_type="RO")
class LSPR_SPU_COUNTER:
    inst_count = bits(0, 31)


@csr(name="LSPR_CP_CONTROL", address=0xB00, width=64, rw_type="RW")
class LSPR_CP_CONTROL:
    double_buffer = bits(0)


@csr(name="LSPR_MXE_CONTROL", address=0xB30, width=64, rw_type="RW")
class LSPR_MXE_CONTROL:
    mem_slow_mode = bits(0)


@csr(name="LSPR_SVR_TIMING", address=0xB60, width=64, rw_type="RW")
class LSPR_SVR_TIMING:
    svr_read_delay  = bits(0)
    svr_write_delay = bits(1)


@csr(name="LSPR_SPU_ERROR_CLEAR", address=0xB70, width=64, rw_type="WO")
class LSPR_SPU_ERROR_CLEAR:
    dma_err_clr = bits(0)
    mxe_err_clr = bits(1)
    sde_err_clr = bits(2)
    pde_err_clr = bits(3)


@csr(name="LSPR_SPU_STATUS", address=0xB80, width=64, rw_type="RO")
class LSPR_SPU_STATUS:
    idle           = bits(0)
    illegal_opcode = bits(1)
    dmac_busy      = bits(4)
    mxe_busy       = bits(5)
    sde_busy       = bits(6)
    pde_busy       = bits(7)
    cp_state       = bits(8, 10)
    last_command   = bits(16, 25)


@csr(name="LSPR_DMA_DEBUG", address=0xBA1, width=64, rw_type="RO")
class LSPR_DMA_DEBUG:
    dma_cmd_hang   = bits(0, 3)
    dma_cmd_err    = bits(4, 7)
    dma_err_zero   = bits(8)
    dma_err_axi    = bits(9)
    dma_err_oob    = bits(10)
    dma_err_l1addr = bits(11)


@csr(name="LSPR_MXE_DEBUG", address=0xBB1, width=64, rw_type="RO")
class LSPR_MXE_DEBUG:
    mxe_ctrl_fsm = bits(0, 3)
    mxe_err      = bits(4, 5)
    mxe_nan      = bits(6, 7)


@csr(name="LSPR_SDE_DEBUG", address=0xBC1, width=64, rw_type="RO")
class LSPR_SDE_DEBUG:
    sde_err_inst     = bits(0)
    sde_err_state    = bits(1, 6)
    sde_err_data_num = bits(7)


@csr(name="LSPR_PDE_DEBUG", address=0xBD1, width=64, rw_type="RO")
class LSPR_PDE_DEBUG:
    pde_read_req     = bits(0)
    odd_address_err  = bits(1)
    kernel_size_err  = bits(2)
    stride_err       = bits(3)
    imap_size_err    = bits(4)
    omap_size_err    = bits(5)


@csr(name="LSPR_CREDIT", address=0xBE0, width=64, rw_type="RO")
class LSPR_CREDIT:
    load_count     = bits(0, 1)
    load_error_uf  = bits(2)
    load_error_of  = bits(3)
    store_count    = bits(4, 5)
    store_error_uf = bits(6)
    store_error_of = bits(7)


@csr(name="LSPR_THREAD_ID", address=0xBFF, width=64, rw_type="RO")
class LSPR_THREAD_ID:
    value = bits(0, 5)


# -----------------------------------------------------------------------
# 32-bit APB Registers: LSPR
# -----------------------------------------------------------------------

@csr(name="APB_LSPR_SPM_ADDRA", address=0xE00, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SPM_ADDRA:
    start_a = bits(0, 18)


@csr(name="APB_LSPR_SPM_ADDRB", address=0xE04, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SPM_ADDRB:
    start_b = bits(0, 18)


@csr(name="APB_LSPR_SPM_ADDRC", address=0xE08, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SPM_ADDRC:
    start_c = bits(0, 18)


@csr(name="APB_LSPR_SPM_ADDRR", address=0xE0C, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SPM_ADDRR:
    start_r = bits(0, 18)


@csr(name="APB_LSPR_SPU_COUNTER_MODE", address=0xE10, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SPU_COUNTER_MODE:
    inst_count_mode = bits(0, 2)


@csr(name="APB_LSPR_SPU_COUNTER_CLEAR", address=0xE14, width=32, rw_type="WO", bus_type=BusType.APB)
class APB_LSPR_SPU_COUNTER_CLEAR:
    inst_count_clr = bits(0)


@csr(name="APB_LSPR_SPU_COUNTER", address=0xE18, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_SPU_COUNTER:
    inst_count = bits(0, 31)


@csr(name="APB_LSPR_CP_CONTROL", address=0xE40, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_CP_CONTROL:
    double_buffer = bits(0)


@csr(name="APB_LSPR_MXE_CONTROL", address=0xE88, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_MXE_CONTROL:
    mem_slow_mode = bits(0)


@csr(name="APB_LSPR_SVR_TIMING", address=0xEA0, width=32, rw_type="RW", bus_type=BusType.APB)
class APB_LSPR_SVR_TIMING:
    svr_read_delay  = bits(0)
    svr_write_delay = bits(1)


@csr(name="APB_LSPR_SPU_ERROR_CLEAR", address=0xEB0, width=32, rw_type="WO", bus_type=BusType.APB)
class APB_LSPR_SPU_ERROR_CLEAR:
    dma_err_clr = bits(0)
    mxe_err_clr = bits(1)
    sde_err_clr = bits(2)
    pde_err_clr = bits(3)


@csr(name="APB_LSPR_SPU_STATUS", address=0xEC0, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_SPU_STATUS:
    idle           = bits(0)
    illegal_opcode = bits(1)
    dmac_busy      = bits(4)
    mxe_busy       = bits(5)
    sde_busy       = bits(6)
    pde_busy       = bits(7)
    cp_state       = bits(8, 10)
    last_command   = bits(16, 25)


@csr(name="APB_LSPR_DMA_DEBUG", address=0xED4, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_DMA_DEBUG:
    dma_cmd_hang   = bits(0, 3)
    dma_cmd_err    = bits(4, 7)
    dma_err_zero   = bits(8)
    dma_err_axi    = bits(9)
    dma_err_oob    = bits(10)
    dma_err_l1addr = bits(11)


@csr(name="APB_LSPR_MXE_DEBUG", address=0xEE4, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_MXE_DEBUG:
    mxe_ctrl_fsm = bits(0, 3)
    mxe_err      = bits(4, 5)
    mxe_nan      = bits(6, 7)


@csr(name="APB_LSPR_SDE_DEBUG", address=0xEF4, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_SDE_DEBUG:
    sde_err_inst     = bits(0)
    sde_err_state    = bits(1, 6)
    sde_err_data_num = bits(7)


@csr(name="APB_LSPR_PDE_DEBUG", address=0xF04, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_PDE_DEBUG:
    pde_read_req     = bits(0)
    odd_address_err  = bits(1)
    kernel_size_err  = bits(2)
    stride_err       = bits(3)
    imap_size_err    = bits(4)
    omap_size_err    = bits(5)


@csr(name="APB_LSPR_CREDIT", address=0xFD0, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_CREDIT:
    load_count     = bits(0, 1)
    load_error_uf  = bits(2)
    load_error_of  = bits(3)
    store_count    = bits(4, 5)
    store_error_uf = bits(6)
    store_error_of = bits(7)


@csr(name="APB_LSPR_THREAD_ID", address=0xFF0, width=32, rw_type="RO", bus_type=BusType.APB)
class APB_LSPR_THREAD_ID:
    spu_id = bits(0, 5)
