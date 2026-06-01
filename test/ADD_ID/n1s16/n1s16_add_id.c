//==================================================================
// n1s16_add_id — indexed row addition for GGML_OP_ADD_ID
// ggml semantic contract:
//   dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
// Fixed harness-aligned shape used by this kernel:
//   --shape [189983,9]
//   a   : [189983, 9, 2] FP16
//   b   : [189983, 8]    FP16
//   ids : [9, 2]         I32
//   dst : [189983, 9, 2] FP16
// GTX strategy:
//   1) shared section DMA-loads full A and B tables into L2.
//   2) host-side setup decodes the small ids table into B-row byte offsets.
//   3) each SPU loads one A row and its selected B row into L1.
//   4) each SPU performs __add_vv and stores the result row to L2.
//==================================================================

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4

#define WIDTH               189983
#define SEQ_LEN             9
#define BATCH               2
#define HEIGHT              (SEQ_LEN * BATCH)
#define N_B_ROWS            8

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000
#define BASE_DDR_IDS        0x3000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define NEST_DATA_BYTES     (HEIGHT * ROW_BYTES)
#define B_TABLE_BYTES       (N_B_ROWS * ROW_BYTES)
#define MAX_DMA_BYTES       65504u
#define L2_ALIGN_BYTES      32u
#define ALIGN_UP_32(x)      (((x) + (L2_ALIGN_BYTES - 1u)) & ~(L2_ALIGN_BYTES - 1u))
#define L2_B_TABLE          ALIGN_UP_32(L2_A + NEST_DATA_BYTES)
#define L2_RESULT           ALIGN_UP_32(L2_B_TABLE + B_TABLE_BYTES)

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t rows_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;
    return (uint32_t)(tid < rows_per_spu_rem ? (rows_per_spu_quot + 1u) : rows_per_spu_quot);
}

static inline uint32_t row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;

    return (uint32_t)(tid < rows_per_spu_rem
        ? tid * (rows_per_spu_quot + 1u)
        : rows_per_spu_rem * (rows_per_spu_quot + 1u) + (tid - rows_per_spu_rem) * rows_per_spu_quot);
}

static inline uint16_t active_tid_mask_for_rows(uint32_t total_rows) {
    uint32_t active_tid_count = min_u32(total_rows, SPU_NUM_PER_NEST);
    return (uint16_t)(active_tid_count >= SPU_NUM_PER_NEST ? 0xFFFFu : ((1u << active_tid_count) - 1u));
}

static inline uint16_t tid_mask(uint8_t tid) {
    return (uint16_t)(0x1u << tid);
}

int main(void) {
    volatile int32_t * ids_ddr = (volatile int32_t *)(uintptr_t) GTX_MAIN_ADDR(BASE_DDR_IDS);
    uint32_t selected_b_row_off[HEIGHT];
    uint16_t active_tid_mask = active_tid_mask_for_rows(HEIGHT);

    for (uint32_t row = 0; row < HEIGHT; ++row) {
        selected_b_row_off[row] = (uint32_t) ids_ddr[row] * ROW_BYTES;
    }

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                for (uint32_t row = 0; row < HEIGHT; ++row) {
                    uint32_t row_off = row * ROW_BYTES;

                    for (uint32_t chunk_byte_off = 0; chunk_byte_off < ROW_BYTES; chunk_byte_off += MAX_DMA_BYTES) {
                        uint32_t chunk_bytes = min_u32(ROW_BYTES - chunk_byte_off, MAX_DMA_BYTES);
                        uint32_t l2_chunk_off = row_off + chunk_byte_off;
                        uint32_t is_last_a_chunk = (row + 1u == HEIGHT) && (chunk_byte_off + chunk_bytes == ROW_BYTES);

                        if (is_last_a_chunk && B_TABLE_BYTES == 0u) {
                            __load_cr(
                                GTX_MAIN_ADDR(BASE_DDR_A) + l2_chunk_off, L2_A + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes,
                                1, active_tid_mask, 0xBEEF
                            );
                        } else {
                            __load(
                                GTX_MAIN_ADDR(BASE_DDR_A) + l2_chunk_off, L2_A + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes
                            );
                        }
                    }
                }

                for (uint32_t row = 0; row < N_B_ROWS; ++row) {
                    uint32_t row_off = row * ROW_BYTES;

                    for (uint32_t chunk_byte_off = 0; chunk_byte_off < ROW_BYTES; chunk_byte_off += MAX_DMA_BYTES) {
                        uint32_t chunk_bytes = min_u32(ROW_BYTES - chunk_byte_off, MAX_DMA_BYTES);
                        uint32_t l2_chunk_off = row_off + chunk_byte_off;
                        uint32_t is_last_b_chunk = (row + 1u == N_B_ROWS) && (chunk_byte_off + chunk_bytes == ROW_BYTES);

                        if (is_last_b_chunk) {
                            __load_cr(
                                GTX_MAIN_ADDR(BASE_DDR_B) + l2_chunk_off, L2_B_TABLE + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes,
                                1, active_tid_mask, 0xBEEF
                            );
                        } else {
                            __load(
                                GTX_MAIN_ADDR(BASE_DDR_B) + l2_chunk_off, L2_B_TABLE + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes
                            );
                        }
                    }
                }

                __credit_chk(active_tid_mask);

                for (uint32_t row = 0; row < HEIGHT; ++row) {
                    uint32_t row_off = row * ROW_BYTES;

                    for (uint32_t chunk_byte_off = 0; chunk_byte_off < ROW_BYTES; chunk_byte_off += MAX_DMA_BYTES) {
                        uint32_t chunk_bytes = min_u32(ROW_BYTES - chunk_byte_off, MAX_DMA_BYTES);
                        uint32_t l2_chunk_off = row_off + chunk_byte_off;
                        uint32_t is_last_store_chunk = (row + 1u == HEIGHT) && (chunk_byte_off + chunk_bytes == ROW_BYTES);

                        if (is_last_store_chunk) {
                            __store_cr(
                                L2_RESULT + l2_chunk_off, GTX_MAIN_ADDR(BASE_DDR_RESULT) + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes,
                                1, active_tid_mask
                            );
                        } else {
                            __store(
                                L2_RESULT + l2_chunk_off, GTX_MAIN_ADDR(BASE_DDR_RESULT) + l2_chunk_off,
                                chunk_bytes,
                                (uint16_t) chunk_bytes,
                                1, (uint16_t) chunk_bytes
                            );
                        }
                    }
                }
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; ++tid) {
                uint32_t row_start = row_start_for_tid(HEIGHT, tid);
                uint32_t rows_for_this_tid = rows_for_tid(HEIGHT, tid);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    if (rows_for_this_tid > 0u) {
                        __credit_chk(0xBEEF);

                        for (uint32_t r = 0; r < rows_for_this_tid; ++r) {
                            uint32_t row_idx = row_start + r;
                            uint32_t row_off = row_idx * ROW_BYTES;
                            uint32_t b_row_off = selected_b_row_off[row_idx];

                            for (uint32_t chunk_byte_off = 0; chunk_byte_off < ROW_BYTES; chunk_byte_off += MAX_DMA_BYTES) {
                                uint32_t chunk_bytes = min_u32(ROW_BYTES - chunk_byte_off, MAX_DMA_BYTES);
                                uint32_t chunk_elems = chunk_bytes / DTYPE;
                                uint32_t is_last_chunk = (chunk_byte_off + chunk_bytes) == ROW_BYTES;
                                uint32_t is_last_row = (r + 1u) == rows_for_this_tid;

                                __load(
                                    L2_A + row_off + chunk_byte_off, BANK_A,
                                    chunk_bytes,
                                    (uint16_t) chunk_bytes,
                                    1, (uint16_t) chunk_bytes
                                );

                                if (is_last_row && is_last_chunk) {
                                    __load_cr(
                                        L2_B_TABLE + b_row_off + chunk_byte_off, BANK_B,
                                        chunk_bytes,
                                        (uint16_t) chunk_bytes,
                                        1, (uint16_t) chunk_bytes,
                                        1, tid_mask(tid), nest_id
                                    );
                                } else {
                                    __load(
                                        L2_B_TABLE + b_row_off + chunk_byte_off, BANK_B,
                                        chunk_bytes,
                                        (uint16_t) chunk_bytes,
                                        1, (uint16_t) chunk_bytes
                                    );
                                }

                                __add_vv(chunk_elems);

                                if (is_last_row && is_last_chunk) {
                                    __store_cr(
                                        BANK_R, L2_RESULT + row_off + chunk_byte_off,
                                        chunk_bytes,
                                        (uint16_t) chunk_bytes,
                                        1, (uint16_t) chunk_bytes,
                                        1, tid_mask(tid)
                                    );
                                } else {
                                    __store(
                                        BANK_R, L2_RESULT + row_off + chunk_byte_off,
                                        chunk_bytes,
                                        (uint16_t) chunk_bytes,
                                        1, (uint16_t) chunk_bytes
                                    );
                                }
                            }
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
