//==================================================================
// {{OP_NAME}} (generated) — element-wise scale (dst = src * scale); scale at DDR BASE_DDR_B
// Source: test/SCALE/n1s16/n1s16_scale.c
// Template name: unary_scale.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               {{WIDTH}}
#define HEIGHT              {{HEIGHT}}

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000   // scale factor (1 FP16)
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define TOTAL_ELEMS         (WIDTH * HEIGHT)
#define TOTAL_BYTES         (TOTAL_ELEMS * DTYPE)
#define SVR_ELEMS           16
#define SVR_BYTES           (SVR_ELEMS * DTYPE)
#define TOTAL_CHUNKS        ((TOTAL_ELEMS + SVR_ELEMS - 1) / SVR_ELEMS)
#define CHUNKS_PER_SPU      (TOTAL_CHUNKS / SPU_NUM_PER_NEST)
#define CHUNK_REMAINDER     (TOTAL_CHUNKS % SPU_NUM_PER_NEST)


int main(void) {


    // Read scale factor from DDR via CPU (before __split)
    volatile uint16_t *scale_ptr = (volatile uint16_t *)GTX_MAIN_ADDR(BASE_DDR_B);
    uint16_t scale_val = *scale_ptr;

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * TOTAL_BYTES;

                // Load src0 to L2_A with credit to all SPUs
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES,
                    1, 0xFFFF, 0xBEEF);

                // Wait all SPUs done
                __credit_chk(0xFFFF);

                // Store result to DDR
                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)TOTAL_BYTES,
                    1, (uint16_t)TOTAL_BYTES,
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    uint16_t chunks_before = (uint16_t)tid * CHUNKS_PER_SPU + ((tid < CHUNK_REMAINDER) ? tid : CHUNK_REMAINDER);
                    uint16_t chunks_for_tid = CHUNKS_PER_SPU + ((tid < CHUNK_REMAINDER) ? 1 : 0);

                    for (uint16_t c = 0; c < chunks_for_tid; c++) {
                        uint16_t chunk_idx = chunks_before + c;
                        uint32_t elem_idx = (uint32_t)chunk_idx * SVR_ELEMS;
                        uint32_t byte_off = elem_idx * DTYPE;
                        uint32_t bytes_left = (uint32_t)TOTAL_BYTES - byte_off;
                        uint32_t chunk_bytes = (bytes_left < SVR_BYTES) ? bytes_left : SVR_BYTES;

                        // Load one 16-element SVR tile from L2 to Bank A.
                        // The final tile may read padding bytes, but only its valid bytes are stored.
                        __load(L2_A + byte_off, BANK_A,
                            SVR_BYTES, (uint16_t)SVR_BYTES, 1, (uint16_t)SVR_BYTES);
                        if (c == chunks_for_tid - 1) __credit_ld(0xBEEF, 0xBEEF);

                        // SVR: load 16 FP16 elements from Bank A, multiply by scalar
                        __load_svr(BANK_A, 0);
                        __mul_is(0, scale_val, 1, 0);
                        __store_svr(BANK_R, 1);

                        // Store result L1 -> L2
                        if (c == chunks_for_tid - 1) {
                            __store_cr(BANK_R, L2_RESULT + byte_off, chunk_bytes, (uint16_t)chunk_bytes, 1, (uint16_t)chunk_bytes, 1, 0x1 << tid);
                        } else {
                            __store(BANK_R, L2_RESULT + byte_off, chunk_bytes, (uint16_t)chunk_bytes, 1, (uint16_t)chunk_bytes);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
