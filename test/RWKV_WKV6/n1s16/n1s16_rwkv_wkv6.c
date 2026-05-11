//==================================================================
// GGML_OP_RWKV_WKV6 — RWKV v6 WKV attention (n1s16)
//
// Fixed generator binding used by this kernel:
//   shape [S,T] = [2,2], H = 1, n_seqs = 1
//
// Implementation path: composed GTX path. The recurrence is ordered on
// SPU0, but the per-token state update and output dot products are
// expressed with GTX vector/dot intrinsics over the S-wide head vector.
//==================================================================

#ifndef N1S16_RWKV_WKV6_C
#define N1S16_RWKV_WKV6_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define ACTIVE_SPU_MASK     0x0001
#define NOT_USE             0xBEEF

#define S                   2
#define H                   1
#define T                   2
#define N_SEQS              1
#define C                   (S * H)
#define FP16_B              2
#define F32_B               4

#define VEC_ELEMS           S
#define VEC_BYTES           (VEC_ELEMS * FP16_B)
#define INPUT_ELEMS         (C * T)
#define INPUT_BYTES         (INPUT_ELEMS * FP16_B)
#define TF_ELEMS            (S * H)
#define TF_BYTES            (TF_ELEMS * FP16_B)
#define STATE_ELEMS         (S * S * H)
#define STATE_BYTES         (STATE_ELEMS * FP16_B)
#define HALF_RESULT_ELEMS   (INPUT_ELEMS + STATE_ELEMS)
#define HALF_RESULT_BYTES   (HALF_RESULT_ELEMS * FP16_B)
#define RESULT_BYTES        (HALF_RESULT_ELEMS * F32_B)

#define BASE_DDR_K          0x1000000
#define BASE_DDR_V          0x2000000
#define BASE_DDR_R          0x3000000
#define BASE_DDR_TF         0x4000000
#define BASE_DDR_TD         0x5000000
#define BASE_DDR_STATE      0x6000000
#define BASE_DDR_RESULT     0xf000000

#define L2_K                0x000000
#define L2_V                0x000100
#define L2_R                0x000200
#define L2_TF               0x000300
#define L2_TD               0x000400
#define L2_STATE            0x000500
#define L2_RESULT           0x000600

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define K_VEC               (BANK_A + 0x0000)
#define V_BCAST             (BANK_A + 0x0020)
#define R_VEC               (BANK_A + 0x0040)
#define TF_VEC              (BANK_A + 0x0060)
#define TD_VEC              (BANK_A + 0x0080)
#define STATE_MAT           (BANK_A + 0x00A0)
#define STATE_VEC           (BANK_A + 0x00C0)
#define KV_VEC              (BANK_A + 0x00E0)
#define TMP_VEC             (BANK_A + 0x0100)
#define ACC_VEC             (BANK_A + 0x0120)
#define HALF_RESULT         (BANK_A + 0x0140)
#define SVR_TMP             0

int main(void) {
    __split();
    {
        __start_plan(NEST_ID);

            __start_shared();
                __fill(L2_RESULT,
                    (uint32_t)RESULT_BYTES, (uint16_t)RESULT_BYTES, 1, 0, 0);

                __load(GTX_MAIN_ADDR(BASE_DDR_K), L2_K,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_V), L2_V,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_R), L2_R,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_TF), L2_TF,
                    TF_BYTES, (uint16_t)TF_BYTES, 1, (uint16_t)TF_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_TD), L2_TD,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_STATE), L2_STATE,
                    STATE_BYTES, (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES,
                    1, ACTIVE_SPU_MASK, NOT_USE);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    RESULT_BYTES, (uint16_t)RESULT_BYTES, 1, (uint16_t)RESULT_BYTES,
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; ++tid) {
                __start_thread(tid);
                    if (tid == 0) {
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(NOT_USE);

                        __load(L2_TF, TF_VEC,
                            TF_BYTES, (uint16_t)TF_BYTES, 1, (uint16_t)TF_BYTES);
                        __load(L2_STATE, STATE_MAT,
                            STATE_BYTES, (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES);
                        __credit_ld(ACTIVE_SPU_MASK, NEST_ID);

                        for (uint8_t t = 0; t < T; ++t) {
                            const uint32_t token_off = (uint32_t)t * C * FP16_B;

                            __load(L2_K + token_off, K_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);
                            __load(L2_R + token_off, R_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);
                            __load(L2_TD + token_off, TD_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);

                            for (uint8_t j = 0; j < S; ++j) {
                                for (uint8_t i = 0; i < S; ++i) {
                                    __copy(STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                        STATE_VEC + (uint32_t)i * FP16_B,
                                        0, FP16_B, 1, 0);
                                }

                                __load(L2_V + token_off + (uint32_t)j * FP16_B, V_BCAST,
                                    0, FP16_B, VEC_ELEMS, FP16_B);

                                // kv_i = k_i * v_j
                                __set_spm_addr(KV_VEC, 0, V_BCAST, K_VEC);
                                __mul_vv(VEC_ELEMS);

                                // tmp_i = kv_i * time_faaaa_i + state_i,j
                                __set_spm_addr(TMP_VEC, 0, TF_VEC, KV_VEC);
                                __mul_vv(VEC_ELEMS);
                                __set_spm_addr(ACC_VEC, 0, STATE_VEC, TMP_VEC);
                                __add_vv(VEC_ELEMS);

                                // y_t,j = dot(r_t, tmp)
                                __set_spm_addr(BANK_R, 0, ACC_VEC, R_VEC);
                                __dot_product(VEC_ELEMS, SVR_TMP);
                                __store_svr(TMP_VEC, SVR_TMP);
                                __copy(TMP_VEC, HALF_RESULT + token_off + (uint32_t)j * FP16_B,
                                    0, FP16_B, 1, 0);

                                // state_i,j = state_i,j * time_decay_i + kv_i
                                __set_spm_addr(TMP_VEC, 0, TD_VEC, STATE_VEC);
                                __mul_vv(VEC_ELEMS);
                                __set_spm_addr(STATE_VEC, 0, KV_VEC, TMP_VEC);
                                __add_vv(VEC_ELEMS);

                                for (uint8_t i = 0; i < S; ++i) {
                                    __copy(STATE_VEC + (uint32_t)i * FP16_B,
                                        STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                        0, FP16_B, 1, 0);
                                }
                            }
                        }

                        for (uint8_t i = 0; i < S; ++i) {
                            for (uint8_t j = 0; j < S; ++j) {
                                __copy(STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                    HALF_RESULT + INPUT_BYTES + ((uint32_t)i * S + j) * FP16_B,
                                    0, FP16_B, 1, 0);
                            }
                        }

                        __store_cr(HALF_RESULT, L2_RESULT,
                            HALF_RESULT_BYTES, (uint16_t)HALF_RESULT_BYTES, 1, (uint16_t)HALF_RESULT_BYTES,
                            1, ACTIVE_SPU_MASK);
                    }
                __end_thread(tid);
            }

        __end_plan(NEST_ID);
    }
    __join();

    return 0;
}

#endif
