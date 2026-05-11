//==================================================================
// n1s16_round — element-wise ggml roundf semantics, 1 NEST x 16 SPUs
// dst[row] = sign(src0[row]) * floor(abs(src0[row]) + 0.5), 256 rows x 64 FP16 elements
// Process each row as two 32-wide vector chunks to avoid width-dependent upper-lane behavior.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               64
#define HEIGHT              256

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x008000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)

#define CHUNK_ELEMS         32
#define CHUNK_BYTES         (CHUNK_ELEMS * DTYPE)
#define CHUNKS_PER_ROW      (WIDTH / CHUNK_ELEMS)

#define FP16_HALF           0x3800
#define FP16_NEG_HALF       0xB800
#define FP16_NEG_PREV_HALF  0xB7FF  // -0.499755859375, largest FP16 value below -0.5 by magnitude


int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF, 0xBEEF);

                __credit_chk(0xFFFF);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row_off = (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_BYTES;

                        for (uint8_t c = 0; c < CHUNKS_PER_ROW; c++) {
                            uint32_t chunk_off = (uint32_t)c * CHUNK_BYTES;

                            __load(L2_A + row_off + chunk_off, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);

                            __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                            __abs_v(CHUNK_ELEMS);                // R = abs(x)
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __add_vs(CHUNK_ELEMS, FP16_HALF, 0);  // R = abs(x) + 0.5
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __floor_v(CHUNK_ELEMS);              // R = floor(abs(x) + 0.5)
                            __copy(BANK_R, BANK_C,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);

                            __load(L2_A + row_off + chunk_off, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __add_vs(CHUNK_ELEMS, FP16_NEG_HALF, 0); // R = x - 0.5
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __step_v(CHUNK_ELEMS);               // R = positive x > 0.5 mask
                            __copy(BANK_R, BANK_B,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);

                            __load(L2_A + row_off + chunk_off, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __neg_v(CHUNK_ELEMS);                // R = -x
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __add_vs(CHUNK_ELEMS, FP16_NEG_PREV_HALF, 0); // R = -x - prev(0.5)
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __step_v(CHUNK_ELEMS);               // R = negative x <= -0.5 mask
                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __add_vv(CHUNK_ELEMS);               // R = combined nonzero-result mask

                            __copy(BANK_R, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __copy(BANK_C, BANK_B,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __mul_vv(CHUNK_ELEMS);               // R = rounded magnitude or +0
                            __copy(BANK_R, BANK_C,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);

                            __load(L2_A + row_off + chunk_off, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            if (r == ROWS_PER_SPU - 1 && c == CHUNKS_PER_ROW - 1) {
                                __credit_ld((uint32_t)(1u << tid), (uint32_t)(1u << nest_id));
                            }
                            __sign_v(CHUNK_ELEMS);               // R = sign(x)
                            __copy(BANK_R, BANK_B,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __copy(BANK_C, BANK_A,
                                CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
                            __mul_vv(CHUNK_ELEMS);               // R = sign(x) * rounded magnitude

                            if (r == ROWS_PER_SPU - 1 && c == CHUNKS_PER_ROW - 1) {
                                __store_cr(BANK_R, L2_RESULT + row_off + chunk_off,
                                    CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES, 1, 0x1 << tid);
                            } else {
                                __store(BANK_R, L2_RESULT + row_off + chunk_off,
                                    CHUNK_BYTES, (uint16_t)CHUNK_BYTES, 1, (uint16_t)CHUNK_BYTES);
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
