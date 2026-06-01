//==================================================================
// n1s16_pool_1d — ggml POOL_1D average pooling
//
// generate_data.cpp builds ggml_pool_1d(src0, AVG, k=2, s=2, p=0).
// Self-check shape [128,1]: input 128 FP16 values, output 64 FP16 values.
// Each SPU handles four contiguous output elements using the direct GTX
// average-pooling intrinsic over a 1 x 8 input tile.
//==================================================================

#ifndef N1S16_POOL_1D_C
#define N1S16_POOL_1D_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4

#define INPUT_LEN           4096
#define OUTPUT_LEN          2048
#define K_SIZE              2
#define STRIDE              2

#define OUT_PER_SPU         (OUTPUT_LEN / SPU_NUM_PER_NEST)
#define IN_PER_SPU          (OUT_PER_SPU * STRIDE)

#define INPUT_BYTES         (INPUT_LEN * DTYPE)
#define OUTPUT_BYTES        (OUTPUT_LEN * DTYPE)
#define SPU_INPUT_BYTES     (IN_PER_SPU * DTYPE)
#define SPU_OUTPUT_BYTES    (OUT_PER_SPU * DTYPE)

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// FP16 0.5, the reciprocal of K_SIZE=2, used by pool.a for averaging.
#define FP16_HALF           0x3F000000

int main(void) {
    uint8_t nest_id = 0;
    uint16_t active_tid_mask = 0xFFFF;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    INPUT_BYTES,
                    (uint16_t)INPUT_BYTES,
                    1, (uint16_t)INPUT_BYTES,
                    1, active_tid_mask, 0xBEEF);

                __credit_chk(active_tid_mask);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES,
                    1, (uint16_t)OUTPUT_BYTES,
                    1, active_tid_mask);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint32_t in_off = (uint32_t)tid * SPU_INPUT_BYTES;
                uint32_t out_off = (uint32_t)tid * SPU_OUTPUT_BYTES;
                uint16_t tid_mask = (uint16_t)(1u << tid);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    __load_cr(L2_A + in_off, BANK_A,
                        SPU_INPUT_BYTES,
                        (uint16_t)SPU_INPUT_BYTES,
                        1, (uint16_t)SPU_INPUT_BYTES,
                        1, tid_mask, nest_id);

                    // Treat the 1D tile as a 1x8 row and pool along width:
                    // input cols=8 -> output cols=4 with kernel 1x2, stride 1x2.
                    __pool_a(1, IN_PER_SPU, 1, OUT_PER_SPU, 1, K_SIZE, 1, STRIDE, FP16_HALF);

                    __store_cr(BANK_R, L2_RESULT + out_off,
                        SPU_OUTPUT_BYTES,
                        (uint16_t)SPU_OUTPUT_BYTES,
                        1, (uint16_t)SPU_OUTPUT_BYTES,
                        1, tid_mask);
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();

    return 0;
}

#endif
