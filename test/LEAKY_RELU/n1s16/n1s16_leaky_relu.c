//==================================================================
// n1s16_leaky_relu — element-wise leaky ReLU with 1 NEST x 16 SPUs
// dst[row] = leaky_relu(src0[row]), HEIGHT rows x WIDTH FP16 elements
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               251
#define HEIGHT              127

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x010000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define FP16_ZERO           0x0000
#define FP16_NEG_SLOPE      0x3000  // 0.125, matching generate_data ggml_leaky_relu

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_BASE_PER_SPU   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_EXTRA          (ROWS_PER_NEST % SPU_NUM_PER_NEST)
#define ACTIVE_SPUS         ((ROWS_PER_NEST < SPU_NUM_PER_NEST) ? ROWS_PER_NEST : SPU_NUM_PER_NEST)
#define ACTIVE_SPU_MASK     ((ACTIVE_SPUS == SPU_NUM_PER_NEST) ? 0xFFFFu : ((1u << ACTIVE_SPUS) - 1u))


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
                    1, ACTIVE_SPU_MASK, 0xBEEF);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    uint8_t rows_for_tid = ROWS_BASE_PER_SPU + ((tid < ROWS_EXTRA) ? 1 : 0);
                    uint32_t row_start = (uint32_t)tid * ROWS_BASE_PER_SPU + ((tid < ROWS_EXTRA) ? tid : ROWS_EXTRA);
                    if (rows_for_tid > 0) {
                        __credit_chk(0xBEEF);
                    }
                    for (uint8_t r = 0; r < rows_for_tid; r++) {
                        uint32_t row_off = (row_start + r) * ROW_BYTES;

                        __load(L2_A + row_off, BANK_A,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        if (r == rows_for_tid - 1) __credit_ld((uint32_t)(0x1u << tid), (uint32_t)(0x1u << nest_id));

                        // ggml leaky_relu: max(x, 0) + negative_slope * min(x, 0)
                        // Save x, build positive part in B, negative scaled part in A.
                        __copy(BANK_A, BANK_C,
                            ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __clamp_min(WIDTH, FP16_ZERO, 0);
                        __copy(BANK_R, BANK_B,
                            ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);

                        __copy(BANK_C, BANK_A,
                            ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __clamp_max(WIDTH, FP16_ZERO, 0);
                        __copy(BANK_R, BANK_A,
                            ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __mul_vs(WIDTH, FP16_NEG_SLOPE, 0);
                        __copy(BANK_R, BANK_A,
                            ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);

                        __add_vv(WIDTH);

                        if (r == rows_for_tid - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, 0x1 << tid);
                        } else {
                            __store(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
