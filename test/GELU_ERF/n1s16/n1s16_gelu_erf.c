//==================================================================
// n1s16_gelu_erf — element-wise GELU (erf variant) with 1 NEST x 16 SPUs
// dst[row] = gelu_erf(src0[row]), HEIGHT rows x WIDTH FP16 elements
//
// The GTX __gelu intrinsic implements the erf GELU form, reads from Bank R,
// and writes the result to Bank A.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

#define WIDTH               1024
#define HEIGHT              2048

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_ALIGN            0x001000u
#define L2_RESULT           (((WIDTH * HEIGHT * DTYPE) + (L2_ALIGN - 1u)) & ~(L2_ALIGN - 1u))

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)


int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                // Load src0 (with credit to all 16 SPUs)
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)ROW_BYTES,
                    (uint16_t)ROWS_PER_NEST, (uint16_t)ROW_BYTES,
                    1, 0xFFFF, 0xBEEF);

                // Wait all SPUs done
                __credit_chk(0xFFFF);

                // Store result back to DDR
                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)ROW_BYTES,
                    (uint16_t)ROWS_PER_NEST, (uint16_t)ROW_BYTES,
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row_off = (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_BYTES;

                        // L2 -> L1: src0 row to Bank R for __gelu input
                        __load(L2_A + row_off, BANK_R,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
);
                        if (r == ROWS_PER_SPU - 1) __credit_ld((uint32_t)(0x1u << tid), (uint32_t)(0x1u << nest_id));

                        // R -> A: gelu_erf(x)
                        __gelu(WIDTH);

                        // L1 -> L2: result from Bank A (not Bank R!)
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_A, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, 0x1 << tid);
                        } else {
                            __store(BANK_A, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
