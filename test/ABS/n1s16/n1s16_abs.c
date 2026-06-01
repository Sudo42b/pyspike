//==================================================================
// n1s16_abs — element-wise absolute value with 1 NEST x 16 SPUs
// dst[row] = |src0[row]|, HEIGHT rows x 8 FP16 elements
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP32 (SMM_ISA v2.0.0d: all I/O fp32)

#define WIDTH               8
#define HEIGHT              393217

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define NEST_DATA_BYTES     (ROWS_PER_NEST * ROW_BYTES)
#define MAX_SHARED_DMA_BYTES 65535u
#define SHARED_TILE_MAX_ROWS ((uint32_t)(MAX_SHARED_DMA_BYTES / ROW_BYTES))
#define SHARED_TILE_MAX_BYTES (SHARED_TILE_MAX_ROWS * ROW_BYTES)
#define L2_RESULT           (L2_A + SHARED_TILE_MAX_BYTES)

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t rows_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;
    return (uint32_t)(tid < rows_per_spu_rem ? (rows_per_spu_quot + 1) : rows_per_spu_quot);
}

static inline uint32_t row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;

    return (uint32_t)(tid < rows_per_spu_rem
        ? tid * (rows_per_spu_quot + 1)
        : rows_per_spu_rem * (rows_per_spu_quot + 1) + (tid - rows_per_spu_rem) * rows_per_spu_quot);
}

static inline uint16_t active_tid_mask_for_rows(uint32_t total_rows) {
    uint32_t active_tid_count = min_u32(total_rows, SPU_NUM_PER_NEST);
    return (uint16_t)(active_tid_count >= SPU_NUM_PER_NEST ? 0xFFFFu : ((1u << active_tid_count) - 1u));
}


int main(void) {

    uint8_t nest_id = 0;
    uint32_t nest_off = (uint32_t)nest_id * NEST_DATA_BYTES;

    for (uint32_t tile_row_start = 0; tile_row_start < ROWS_PER_NEST; tile_row_start += SHARED_TILE_MAX_ROWS) {
        uint32_t tile_rows = min_u32(ROWS_PER_NEST - tile_row_start, SHARED_TILE_MAX_ROWS);
        uint32_t tile_bytes = tile_rows * ROW_BYTES;
        uint32_t tile_ddr_off = nest_off + tile_row_start * ROW_BYTES;
        uint16_t active_tid_mask = active_tid_mask_for_rows(tile_rows);

        __split();

        {
            __start_plan(nest_id);

                __start_shared();
                    __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + tile_ddr_off, L2_A,
                        tile_bytes,
                        (uint16_t)tile_bytes,
                        1, (uint16_t)tile_bytes,
                        1, active_tid_mask, 0xBEEF);

                    __credit_chk(active_tid_mask);

                    __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + tile_ddr_off,
                        tile_bytes,
                        (uint16_t)tile_bytes,
                        1, (uint16_t)tile_bytes,
                        1, active_tid_mask);
                __end_shared();

                for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                    uint32_t row_start = row_start_for_tid(tile_rows, tid);
                    uint32_t rows_for_this_tid = rows_for_tid(tile_rows, tid);
                    uint16_t tid_mask = (uint16_t)(0x1u << tid);

                    __start_thread(tid);
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);

                        for (uint32_t r = 0; r < rows_for_this_tid; r++) {
                            uint32_t row_off = (row_start + r) * ROW_BYTES;

                            if (r == rows_for_this_tid - 1) {
                                __load_cr(L2_A + row_off, BANK_A,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                    1, tid_mask, nest_id);
                            } else {
                                __load(L2_A + row_off, BANK_A,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                            }

                            __abs_v(WIDTH);

                            if (r == rows_for_this_tid - 1) {
                                __store_cr(BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                    1, tid_mask);
                            } else {
                                __store(BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                            }
                        }
                    __end_thread(tid);
                }

            __end_plan(nest_id);
        }

        __join();
    }

    return 0;
}
