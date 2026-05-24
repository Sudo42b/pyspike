//==================================================================
// n1s16_add_vv — element-wise OP (vector + vector), 1 NEST x 16 SPUs
// dst[i] = src0[i] + src1[i], multi-tile (ISS-compatible, ABS-style).
//
// Rewritten from the legacy mh.kim __start_plani kernel (which only
// committed on lenient spike; ISS/pyspike produced 0) to the proven
// ABS/ggml_ops_c structure: shared __load_cr supplies the thread credit,
// __credit_chk gates the deferred L2->DDR __store_cr that flushes at the
// plan boundary — so all three sims (ISS / spike / pyspike) converge.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               8
#define HEIGHT              131072  // 1,048,576 elements / WIDTH

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000
#define BASE_DDR_RESULT     0xf000000

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
// L2 layout: A | B | RESULT, each one shared-tile wide.
#define L2_A                0x000000
#define L2_B                (L2_A + SHARED_TILE_MAX_BYTES)
#define L2_RESULT           (L2_B + SHARED_TILE_MAX_BYTES)

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t rows_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t q = total_rows / SPU_NUM_PER_NEST;
    uint32_t rem = total_rows % SPU_NUM_PER_NEST;
    return (uint32_t)(tid < rem ? (q + 1) : q);
}

static inline uint32_t row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t q = total_rows / SPU_NUM_PER_NEST;
    uint32_t rem = total_rows % SPU_NUM_PER_NEST;
    return (uint32_t)(tid < rem
        ? tid * (q + 1)
        : rem * (q + 1) + (tid - rem) * q);
}

static inline uint16_t active_tid_mask_for_rows(uint32_t total_rows) {
    uint32_t cnt = min_u32(total_rows, SPU_NUM_PER_NEST);
    return (uint16_t)(cnt >= SPU_NUM_PER_NEST ? 0xFFFFu : ((1u << cnt) - 1u));
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
                    // src0 (A) -> L2, no credit
                    __load(GTX_MAIN_ADDR(BASE_DDR_A) + tile_ddr_off, L2_A,
                        tile_bytes, (uint16_t)tile_bytes, 1, (uint16_t)tile_bytes);
                    // src1 (B) -> L2, with credit to the active threads
                    __load_cr(GTX_MAIN_ADDR(BASE_DDR_B) + tile_ddr_off, L2_B,
                        tile_bytes, (uint16_t)tile_bytes, 1, (uint16_t)tile_bytes,
                        1, active_tid_mask, 0xBEEF);

                    __credit_chk(active_tid_mask);

                    __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + tile_ddr_off,
                        tile_bytes, (uint16_t)tile_bytes, 1, (uint16_t)tile_bytes,
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

                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                            __load(L2_B + row_off, BANK_B,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                            if (r == rows_for_this_tid - 1) {
                                __credit_ld(tid, nest_id);
                            }

                            __add_vv(WIDTH);

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
