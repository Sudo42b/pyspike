//==================================================================
// {{OP_NAME}} (generated) — row-wise FP16 SUM_ROWS (dst[row] = sum(row))
// Source: test/SUM_ROWS/n1s16/n1s16_sum_rows.c
// Template name: unary_sum_rows.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_NUM              1u
#define SPU_NUM_PER_NEST      16u
#define DTYPE                 2u

#define WIDTH                 {{WIDTH}}
#define HEIGHT                {{HEIGHT}}
#define BASE_DDR_A            0x1000000u
#define BASE_DDR_RESULT       0xf000000u

#define L2_A                  0x000000u
#define L2_RESULT             0x020000u

#define BANK_A                0x00000u
#define BANK_B                0x20000u
#define BANK_C                0x30000u
#define BANK_R                0x50000u

#define ROW_BYTES             (WIDTH * DTYPE)
#define RESULT_DTYPE          2u
#define ROWS_PER_NEST         (HEIGHT / NEST_NUM)
#define NEST_DATA_BYTES       (ROWS_PER_NEST * ROW_BYTES)
#define MAX_DMA_BYTES         65535u
#define TILE_MAX_ROWS         (MAX_DMA_BYTES / ROW_BYTES)
#define TILE_COUNT            ((ROWS_PER_NEST + TILE_MAX_ROWS - 1u) / \
                               TILE_MAX_ROWS)
#define LOAD_READY_TOKEN_BASE 0xAA00u

#define SVR_SUM               0u

#if WIDTH == 0 || HEIGHT == 0
#error "n1s16_sum_rows requires non-zero WIDTH and HEIGHT"
#endif

#if WIDTH > 65535u
#error "n1s16_sum_rows requires WIDTH <= 65535 for __sum element count"
#endif

#if ROW_BYTES > MAX_DMA_BYTES
#error "n1s16_sum_rows requires one input row to fit in a 16-bit DMA byte count"
#endif

#if TILE_MAX_ROWS == 0
#error "n1s16_sum_rows requires at least one row per streaming tile"
#endif

#if TILE_MAX_ROWS > 65535u
#error "n1s16_sum_rows tile row count must fit in 16 bits"
#endif

#if (TILE_MAX_ROWS * RESULT_DTYPE) > MAX_DMA_BYTES
#error "n1s16_sum_rows output tile must fit in a 16-bit DMA byte count"
#endif

#if (LOAD_READY_TOKEN_BASE + TILE_COUNT) > 65535u
#error "n1s16_sum_rows has too many streaming tiles for 16-bit credit tokens"
#endif

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t rows_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t q = total_rows / SPU_NUM_PER_NEST;
    uint32_t r = total_rows % SPU_NUM_PER_NEST;

    return tid < r ? q + 1u : q;
}

static inline uint32_t row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t q = total_rows / SPU_NUM_PER_NEST;
    uint32_t r = total_rows % SPU_NUM_PER_NEST;

    if (tid < r) {
        return (uint32_t)tid * (q + 1u);
    }

    return r * (q + 1u) + ((uint32_t)tid - r) * q;
}

static inline uint16_t active_tid_mask_for_rows(uint32_t total_rows) {
    uint32_t active_tid_count = min_u32(total_rows, SPU_NUM_PER_NEST);

    if (active_tid_count >= SPU_NUM_PER_NEST) {
        return 0xFFFFu;
    }

    return (uint16_t)((1u << active_tid_count) - 1u);
}

int main(void) {
    uint8_t nest_id = 0;
    uint32_t nest_off = (uint32_t)nest_id * NEST_DATA_BYTES;

    for (uint32_t tile_row = 0; tile_row < ROWS_PER_NEST; tile_row += TILE_MAX_ROWS) {
        uint32_t tile_rows = min_u32(ROWS_PER_NEST - tile_row, TILE_MAX_ROWS);
        uint32_t tile_input_bytes = tile_rows * ROW_BYTES;
        uint32_t tile_output_bytes = tile_rows * RESULT_DTYPE;
        uint32_t tile_input_ddr_off = nest_off + tile_row * ROW_BYTES;
        uint32_t tile_output_ddr_off = tile_row * RESULT_DTYPE;
        uint32_t tile_idx = tile_row / TILE_MAX_ROWS;
        uint16_t active_tid_mask = active_tid_mask_for_rows(tile_rows);
        uint32_t load_token = LOAD_READY_TOKEN_BASE + tile_idx;

        __split();

        {
            __start_plan(nest_id);

                __start_shared();
                    __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + tile_input_ddr_off,
                        L2_A, tile_input_bytes, (uint16_t)tile_input_bytes,
                        1u, (uint16_t)tile_input_bytes, 1u,
                        active_tid_mask, load_token);

                    __credit_chk(active_tid_mask);

                    __store_cr(L2_RESULT,
                        GTX_MAIN_ADDR(BASE_DDR_RESULT) + tile_output_ddr_off,
                        tile_output_bytes, (uint16_t)tile_output_bytes,
                        1u, (uint16_t)tile_output_bytes, 1u,
                        active_tid_mask);
                __end_shared();

                for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                    uint32_t rows_this_tid = rows_for_tid(tile_rows, tid);

                    __start_thread(tid);
                        if (rows_this_tid > 0u) {
                            uint16_t tid_mask = (uint16_t)(1u << tid);
                            uint32_t row_start = row_start_for_tid(tile_rows, tid);
                            uint32_t input_row_off = row_start * ROW_BYTES;
                            uint32_t output_row_off = row_start * RESULT_DTYPE;
                            uint32_t local_result_bytes = rows_this_tid * RESULT_DTYPE;

                            __credit_chk(load_token);

                            __load(L2_A + input_row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                (uint16_t)rows_this_tid, ROW_BYTES);

                            for (uint32_t local_row = 0; local_row < rows_this_tid; local_row++) {
                                uint32_t input_off = local_row * ROW_BYTES;
                                uint32_t output_off = local_row * RESULT_DTYPE;

                                __set_spm_addr(BANK_R + output_off, BANK_C,
                                    BANK_B, BANK_A + input_off);
                                __sum((uint16_t)WIDTH, SVR_SUM);
                                __store_svr(BANK_R + output_off, SVR_SUM);
                            }

                            __store_cr(BANK_R, L2_RESULT + output_row_off,
                                local_result_bytes, (uint16_t)local_result_bytes,
                                1u, (uint16_t)local_result_bytes, 1u,
                                tid_mask);
                        }
                    __end_thread(tid);
                }

            __end_plan(nest_id);
        }

        __join();
    }

    return 0;
}
