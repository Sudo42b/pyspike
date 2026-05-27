//==================================================================
// {{OP_NAME}} (generated) — element-wise FP16 square (dst = src * src)
// Source: test/SQR/n1s16/n1s16_sqr.c
// Template name: unary_sqr.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_NUM             1u
#define SPU_NUM_PER_NEST     16u
#define DTYPE                2u

#define WIDTH                {{WIDTH}}
#define HEIGHT               {{HEIGHT}}
#define BASE_DDR_A           0x1000000u
#define BASE_DDR_RESULT      0xf000000u

#define L2_A                 0x000000u
#define L2_RESULT            0x020000u

#define BANK_A               0x00000u
#define BANK_B               0x20000u
#define BANK_C               0x30000u
#define BANK_R               0x50000u

#define TOTAL_ELEMS          (WIDTH * HEIGHT)
#define MAX_DMA_BYTES        65535u
#define TILE_MAX_ELEMS       (MAX_DMA_BYTES / DTYPE)
#define LOAD_READY_TOKEN_BASE 0xAD00u

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t elems_for_tid(uint32_t total_elems, uint8_t tid) {
    uint32_t q = total_elems / SPU_NUM_PER_NEST;
    uint32_t r = total_elems % SPU_NUM_PER_NEST;

    return tid < r ? q + 1u : q;
}

static inline uint32_t elem_start_for_tid(uint32_t total_elems, uint8_t tid) {
    uint32_t q = total_elems / SPU_NUM_PER_NEST;
    uint32_t r = total_elems % SPU_NUM_PER_NEST;

    if (tid < r) {
        return (uint32_t)tid * (q + 1u);
    }

    return r * (q + 1u) + ((uint32_t)tid - r) * q;
}

static inline uint16_t active_tid_mask_for_elems(uint32_t total_elems) {
    uint32_t active_tid_count = min_u32(total_elems, SPU_NUM_PER_NEST);

    if (active_tid_count >= SPU_NUM_PER_NEST) {
        return 0xFFFFu;
    }

    return (uint16_t)((1u << active_tid_count) - 1u);
}

int main(void) {
    uint8_t nest_id = 0;

    for (uint32_t tile_elem = 0; tile_elem < TOTAL_ELEMS; tile_elem += TILE_MAX_ELEMS) {
        uint32_t tile_elems = min_u32(TOTAL_ELEMS - tile_elem, TILE_MAX_ELEMS);
        uint32_t tile_bytes = tile_elems * DTYPE;
        uint32_t tile_ddr_off = tile_elem * DTYPE;
        uint32_t tile_idx = tile_elem / TILE_MAX_ELEMS;
        uint16_t active_tid_mask = active_tid_mask_for_elems(tile_elems);
        uint32_t load_token = LOAD_READY_TOKEN_BASE + tile_idx;
        uint16_t tile_bytes_u16 = (uint16_t)tile_bytes;

        __split();

        {
            __start_plan(nest_id);

                __start_shared();
                    __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + tile_ddr_off, L2_A,
                        tile_bytes, tile_bytes_u16, 1u, tile_bytes_u16, 1u,
                        active_tid_mask, load_token);

                    __credit_chk(active_tid_mask);

                    __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + tile_ddr_off,
                        tile_bytes, tile_bytes_u16, 1u, tile_bytes_u16, 1u,
                        active_tid_mask);
                __end_shared();

                for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                    uint32_t elems_this_tid = elems_for_tid(tile_elems, tid);

                    __start_thread(tid);
                        if (elems_this_tid > 0u) {
                            uint16_t tid_mask = (uint16_t)(1u << tid);
                            uint32_t elem_start = elem_start_for_tid(tile_elems, tid);
                            uint32_t chunk_off = elem_start * DTYPE;
                            uint32_t chunk_bytes = elems_this_tid * DTYPE;
                            uint16_t chunk_bytes_u16 = (uint16_t)chunk_bytes;

                            __credit_chk(load_token);

                            __load(L2_A + chunk_off, BANK_A,
                                chunk_bytes, chunk_bytes_u16, 1u, chunk_bytes_u16);

                            __set_spm_addr(BANK_R, BANK_C, BANK_A, BANK_A);
                            __mul_vv(elems_this_tid);

                            __store_cr(BANK_R, L2_RESULT + chunk_off,
                                chunk_bytes, chunk_bytes_u16, 1u, chunk_bytes_u16,
                                1u, tid_mask);
                        }
                    __end_thread(tid);
                }

            __end_plan(nest_id);
        }

        __join();
    }

    return 0;
}
