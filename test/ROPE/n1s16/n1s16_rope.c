//==================================================================
// Test        : n1s16_rope (1 NEST x 16 SPUs)
// Description : ggml ROPE / NEOX, static n_dims=4, seq_len=256.
//==================================================================

#ifndef N1S16_ROPE_C
#define N1S16_ROPE_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16


#define WIDTH               4
#define HEIGHT              256
#define DTYPE               2

#define BASE_DDR_SRC        0x1000000
#define BASE_DDR_POS        0x2000000
#define BASE_DDR_RESULT     0xf000000

#define L2_SRC              0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define DATA_BYTES          (HEIGHT * ROW_BYTES)
#define ROWS_PER_SPU        (HEIGHT / SPU_NUM_PER_NEST)
#define SVR_ADDR            0x800

#define FP16_ONE            0x3C00
#define FP16_INV_2PI        0x3118
#define FP16_TWO_PI         0x4648
#define FP16_TWO_PI_LO      0x3488
#define FP16_INV_PI         0x3518
#define FP16_PI             0x4248
#define FP16_HALF           0x3800
#define FP16_TWO            0x4000
#define FP16_SIX            0x4600
#define FP16_NEG_HALF       0xB800
#define FP16_INV24          0x2955
#define FP16_NEG_INV720     0x95B0
#define FP16_INV40320       0x01A0
#define FP16_NEG_INV6       0xB155
#define FP16_INV120         0x2044
#define FP16_NEG_INV5040    0x8A80
#define FP16_INV362880      0x002E
#define FP16_NEG_INV3628800 0x8003
#define FP16_NEG_INV39916800 0x8002
#define FP16_SCALE_001      0x211F

// [1.0, 0.01, 1.0, 0.01], repeated lane group.
#define THETA_SCALE_WORD    0x211F3C00211F3C00ULL
// [-1, -1, +1, +1] for NEOX: [x2,x3,x0,x1] * sin * sign.
#define SIGN_WORD           0x3C003C00BC00BC00ULL

static inline void write_svr_words(uint8_t svr, uint64_t word) {
    __wrspr(SVR_ADDR + svr * 4 + 0, 0, word, 0);
    __wrspr(SVR_ADDR + svr * 4 + 1, 0, word, 0);
    __wrspr(SVR_ADDR + svr * 4 + 2, 0, word, 0);
    __wrspr(SVR_ADDR + svr * 4 + 3, 0, word, 0);
}

static uint16_t float_to_fp16(float val) {
    uint32_t f;
    __builtin_memcpy(&f, &val, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t) (sign | (mant >> 13));
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

static inline float rope_wrap_pi(float x) {
    int32_t k = (int32_t)(x * 0.15915494309189535f + (x >= 0.0f ? 0.5f : -0.5f));
    return x - (float)k * 6.283185307179586f;
}

static inline void rope_sincos_fp16(float theta, uint16_t * cos_h, uint16_t * sin_h) {
    float x = rope_wrap_pi(theta);
    float x2 = x * x;
    float cos_v = 1.0f + x2 * (-0.5f + x2 * (0.041666666666666664f +
                  x2 * (-0.001388888888888889f + x2 * (0.0000248015873015873f +
                  x2 * (-0.0000002755731922398589f + x2 * 0.00000000208767569878681f)))));
    float sin_v = x * (1.0f + x2 * (-0.16666666666666666f + x2 * (0.008333333333333333f +
                  x2 * (-0.0001984126984126984f + x2 * (0.0000027557319223985893f +
                  x2 * -0.00000002505210838544172f)))));
    *cos_h = float_to_fp16(cos_v);
    *sin_h = float_to_fp16(sin_v);
}

static inline void rope_coeff_words(uint32_t pos, uint64_t * cos_word, uint64_t * sin_word) {
    uint16_t c0, s0, c1, s1;
    rope_sincos_fp16((float)pos, &c0, &s0);
    rope_sincos_fp16((float)pos * 0.01f, &c1, &s1);
    *cos_word = ((uint64_t)c1 << 48) | ((uint64_t)c0 << 32) | ((uint64_t)c1 << 16) | (uint64_t)c0;
    *sin_word = ((uint64_t)s1 << 48) | ((uint64_t)s0 << 32) | ((uint64_t)s1 << 16) | (uint64_t)s0;
}

int main(void) {
    __split();
    {
        uint8_t nest_id = 0;
        __start_plan(nest_id);
            __start_shared();
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_SRC), L2_SRC,
                    (uint32_t)DATA_BYTES, (uint16_t)DATA_BYTES, 1, (uint16_t)DATA_BYTES,
                    1, 0xFFFF, 0xBEEF);

                __credit_chk(0xFFFF);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)DATA_BYTES, (uint16_t)DATA_BYTES, 1, (uint16_t)DATA_BYTES,
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row = (uint32_t)tid * ROWS_PER_SPU + r;
                        uint32_t row_off = row * ROW_BYTES;

                        __load(L2_SRC + row_off, BANK_A,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        if (r == ROWS_PER_SPU - 1) {
                            __credit_ld((uint32_t)1 << tid, (uint32_t)1 << nest_id);
                        }

                        // Bounded scalar metadata: read ggml position id for this sequence row.
                        uint32_t pos = *(((uint32_t *)GTX_MAIN_ADDR(BASE_DDR_POS)) + row);
                        uint64_t cos_word;
                        uint64_t sin_word;
                        rope_coeff_words(pos, &cos_word, &sin_word);

                        __load_svr(BANK_A, 0);             // x = [x0,x1,x2,x3]

                        write_svr_words(12, cos_word);     // cos([p,.01p,p,.01p])
                        write_svr_words(13, sin_word);     // sin([p,.01p,p,.01p])

                        // Build [x2,x3,x0,x1] in BANK_C for NEOX pair rotation.
                        __store_svr(BANK_B, 0);
                        __copy(BANK_B + 2 * DTYPE, BANK_C + 0 * DTYPE, 0, 2 * DTYPE, 1, 0);
                        __copy(BANK_B + 0 * DTYPE, BANK_C + 2 * DTYPE, 0, 2 * DTYPE, 1, 0);
                        __load_svr(BANK_C, 14);

                        write_svr_words(15, SIGN_WORD);
                        __mul_ii(0, 12, 16);               // src*cos
                        __mul_ii(14, 13, 17);              // swapped*sin
                        __mul_ii(17, 15, 17);              // sign-adjusted
                        __add_ii(16, 17, 18);              // rope result

                        __store_svr(BANK_R, 18);

                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, (uint32_t)1 << tid);
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
    return 0;
}

#endif
