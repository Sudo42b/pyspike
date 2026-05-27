//==================================================================
// {{OP_NAME}} (generated) — total reduction; output = 32 bytes (1 fp16 sum + 15 pad)
// Source: test/SUM/n1s16/n1s16_sum.c
// Template name: unary_sum.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_NUM              1u
#define SPU_NUM_PER_NEST      16u
#define DTYPE                 2u

#define WIDTH                 {{WIDTH}}
#define HEIGHT                {{HEIGHT}}
#define TOTAL_ELEMS           (WIDTH * HEIGHT)

#define BASE_DDR_A            0x1000000u
#define BASE_DDR_RESULT       0xf000000u

#define L2_A                  0x000000u
#define L2_RESULT             0x020000u

#define BANK_A                0x00000u
#define BANK_B                0x20000u
#define BANK_C                0x30000u
#define BANK_R                0x50000u

#define ROW_BYTES             (WIDTH * DTYPE)
#define INPUT_BYTES           (TOTAL_ELEMS * DTYPE)
#define OUTPUT_BYTES          32u
#define MAX_DMA_BYTES         65535u
#define DMA_MAX_ROWS          (MAX_DMA_BYTES / ROW_BYTES)
#define MAX_SUM_ELEMS         65535u
#define SUM_MAX_ROWS          (MAX_SUM_ELEMS / WIDTH)
#define TILE_MAX_ROWS         ((DMA_MAX_ROWS < SUM_MAX_ROWS) ? \
                               DMA_MAX_ROWS : SUM_MAX_ROWS)
#define TILE_MAX_BYTES        (TILE_MAX_ROWS * ROW_BYTES)

#define LOAD_READY_TOKEN_BASE 0xAD00u
#define ACTIVE_TID_MASK       0x0001u

#define SVR_ACC               0u
#define SVR_CHUNK             1u

#if WIDTH == 0 || HEIGHT == 0
#error "n1s16_sum requires non-zero WIDTH and HEIGHT"
#endif

#if ROW_BYTES > MAX_DMA_BYTES
#error "n1s16_sum requires one input row to fit in a 16-bit DMA byte count"
#endif

#if TILE_MAX_ROWS == 0
#error "n1s16_sum requires at least one row per streaming tile"
#endif

#if TILE_MAX_BYTES > L2_RESULT
#error "n1s16_sum streaming tile must fit below L2_RESULT scratch"
#endif

static inline __attribute__((always_inline)) uint32_t min_u32(uint32_t a,
        uint32_t b) {
    return a < b ? a : b;
}

int main(void) {
    uint8_t nest_id = 0u;

    for (uint32_t row = 0u; row < HEIGHT; row += TILE_MAX_ROWS) {
        uint32_t rows = min_u32(HEIGHT - row, TILE_MAX_ROWS);
        uint32_t off = row * ROW_BYTES;
        uint32_t bytes = rows * ROW_BYTES;
        uint32_t elems = rows * WIDTH;
        uint32_t tile_idx = row / TILE_MAX_ROWS;
        uint32_t token = LOAD_READY_TOKEN_BASE + tile_idx;
        uint32_t first = tile_idx == 0u;
        uint32_t last = row + rows == HEIGHT;

        __split();

        {
            __start_plan(nest_id);

                __start_shared();
                    __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + off, L2_A,
                        bytes, (uint16_t)bytes, 1u, (uint16_t)bytes, 1u,
                        ACTIVE_TID_MASK, token);

                    if (first) {
                        __fill(L2_RESULT, OUTPUT_BYTES, (uint16_t)OUTPUT_BYTES,
                            1u, 0u, 0u);
                    }

                    __credit_chk(ACTIVE_TID_MASK);

                    if (last) {
                        __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                            OUTPUT_BYTES, (uint16_t)OUTPUT_BYTES, 1u,
                            (uint16_t)OUTPUT_BYTES, 1u, ACTIVE_TID_MASK);
                    } else {
                        __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                            DTYPE, (uint16_t)DTYPE, 1u, (uint16_t)DTYPE, 1u,
                            ACTIVE_TID_MASK);
                    }
                __end_shared();

                for (uint8_t tid = 0u; tid < SPU_NUM_PER_NEST; tid++) {
                    __start_thread(tid);
                        if (tid == 0u) {
                            __credit_chk(token);

                            __load(L2_A, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                (uint16_t)rows, ROW_BYTES);

                            __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                            __sum((uint16_t)elems,
                                first ? SVR_ACC : SVR_CHUNK);
                            if (!first) {
                                __load(L2_RESULT, BANK_C,
                                    OUTPUT_BYTES, (uint16_t)OUTPUT_BYTES, 1u,
                                    (uint16_t)OUTPUT_BYTES);
                                __load_svr(BANK_C, SVR_ACC);
                                __add_ii(SVR_ACC, SVR_CHUNK, SVR_ACC);
                            }

                            __store_svr(BANK_C, SVR_ACC);

                            __store_cr(BANK_C, L2_RESULT,
                                DTYPE, (uint16_t)DTYPE, 1u,
                                (uint16_t)DTYPE, 1u, ACTIVE_TID_MASK);
                        }
                    __end_thread(tid);
                }

            __end_plan(nest_id);
        }

        __join();
    }

    return 0;
}
