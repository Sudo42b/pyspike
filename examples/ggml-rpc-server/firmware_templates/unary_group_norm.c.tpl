//==================================================================
// {{OP_NAME}} (generated) — GROUP_NORM (num_groups=1) over a 2D FP16 tensor
// Source: test/GROUP_NORM/n1s16/n1s16_group_norm.c
// Template name: unary_group_norm.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"

#define DTYPE               2

#define WIDTH               {{WIDTH}}
#define HEIGHT              {{HEIGHT}}
#define TOTAL_ELEMS         (WIDTH * HEIGHT)
#define TOTAL_BYTES         (TOTAL_ELEMS * DTYPE)

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_R              0x50000

#define ACTIVE_SPU_MASK     0x0001

// DEFAULT_EPS in generate_data.cpp is 1e-5; nearest FP16 encoding is 0x00A8.
#define FP16_EPS            {{EPS_FP16}}

int main(void) {
    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);
            __start_shared();
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    (uint32_t)TOTAL_BYTES, (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES,
                    1, ACTIVE_SPU_MASK, 0xBEEF);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)TOTAL_BYTES, (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES,
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            __start_thread(0);
                __set_spm_addr(BANK_R, 0x30000, BANK_B, BANK_A);
                __credit_chk(0xBEEF);

                __load(L2_A, BANK_A,
                    (uint32_t)TOTAL_BYTES, (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES);
                __credit_ld(ACTIVE_SPU_MASK, (1u << nest_id));

                // One group in ggml GROUP_NORM spans all elements for this 2D tensor.
                __layernorm(TOTAL_ELEMS, BANK_A, BANK_B, BANK_R, FP16_EPS);

                __store_cr(BANK_R, L2_RESULT,
                    (uint32_t)TOTAL_BYTES, (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES,
                    1, ACTIVE_SPU_MASK);
            __end_thread(0);
        __end_plan(nest_id);
    }

    __join();
    return 0;
}
