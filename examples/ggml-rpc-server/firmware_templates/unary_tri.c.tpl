//==================================================================
// {{OP_NAME}} (generated) — triangular mask; W/H/tri_type dynamic, copy+fill only
// Source: test/TRI/n1s16/n1s16_tri.c
// Template name: unary_tri.c.tpl
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_TARGET_ID          0u
#define DTYPE                   2u

#define BASE_DDR_OP_PARAMS      0x0ff0000u
#define BASE_DDR_A              0x1000000u
#define BASE_DDR_RESULT         0xf000000u

#define OP_PARAMS_MAGIC         0x50504f47u
#define OP_PARAM_MAGIC          (0x00u / 4u)
#define OP_PARAM_SHAPE_W        (0x10u / 4u)
#define OP_PARAM_SHAPE_H        (0x14u / 4u)
#define OP_PARAM_TRI_TYPE       (0xb8u / 4u)

#define TRI_UPPER_DIAG          0u
#define TRI_UPPER               1u
#define TRI_LOWER_DIAG          2u
#define TRI_LOWER               3u

#define L2_ZERO                 0x000000u

#define ZERO_F16_PATTERN        0x0000000000000000ull
#define MAX_COPY_BYTES          65520u
#define MAX_FILL_BYTES          65534u

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t positive_or_u32(int32_t value, uint32_t fallback) {
    return value > 0 ? (uint32_t)value : fallback;
}

static inline uint32_t valid_tri_type(int32_t value) {
    return value >= 0 && value <= (int32_t)TRI_LOWER ?
        (uint32_t)value : TRI_UPPER_DIAG;
}

static void copy_bytes(uint32_t src_base, uint32_t dst_base, uint32_t byte_count) {
    uint32_t copied = 0u;

    while (copied < byte_count) {
        uint32_t chunk_bytes = min_u32(byte_count - copied, MAX_COPY_BYTES);

        __copy_mem(GTX_MAIN_ADDR(src_base) + copied,
            GTX_MAIN_ADDR(dst_base) + copied,
            chunk_bytes,
            (uint16_t)chunk_bytes,
            1u,
            (uint16_t)chunk_bytes,
            (uint16_t)(chunk_bytes >> 16));

        copied += chunk_bytes;
    }
}

static void fill_zero_result_bytes(uint32_t dst_byte_off, uint32_t byte_count) {
    uint32_t filled = 0u;

    while (filled < byte_count) {
        uint32_t chunk_bytes = min_u32(byte_count - filled, MAX_FILL_BYTES);

        __fill(L2_ZERO, chunk_bytes, (uint16_t)chunk_bytes,
            1u, ZERO_F16_PATTERN, 0u);

        __store(L2_ZERO,
            GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_byte_off + filled,
            chunk_bytes, (uint16_t)chunk_bytes, 1u, chunk_bytes);

        filled += chunk_bytes;
    }
}

static void zero_row_run(uint32_t row, uint32_t start_col, uint32_t cols,
        uint32_t row_bytes) {
    if (cols == 0u) {
        return;
    }

    uint32_t byte_off = row * row_bytes + start_col * DTYPE;
    fill_zero_result_bytes(byte_off, cols * DTYPE);
}

static void copy_row_run(uint32_t row, uint32_t start_col, uint32_t cols,
        uint32_t row_bytes) {
    if (cols == 0u) {
        return;
    }

    uint32_t byte_off = row * row_bytes + start_col * DTYPE;
    copy_bytes(BASE_DDR_A + byte_off, BASE_DDR_RESULT + byte_off,
        cols * DTYPE);
}

static void zero_discarded_triangle(uint32_t n, uint32_t tri_type) {
    uint32_t row_bytes = n * DTYPE;

    for (uint32_t row = 0u; row < n; row++) {
        uint32_t start_col = 0u;
        uint32_t cols = 0u;

        if (tri_type == TRI_UPPER_DIAG) {
            cols = row;
        } else if (tri_type == TRI_UPPER) {
            cols = row + 1u;
        } else if (tri_type == TRI_LOWER_DIAG) {
            start_col = row + 1u;
            cols = n - start_col;
        } else {
            start_col = row;
            cols = n - row;
        }

        zero_row_run(row, start_col, cols, row_bytes);
    }
}

static void copy_kept_triangle(uint32_t n, uint32_t tri_type) {
    uint32_t row_bytes = n * DTYPE;

    for (uint32_t row = 0u; row < n; row++) {
        uint32_t start_col = 0u;
        uint32_t cols = 0u;

        if (tri_type == TRI_UPPER_DIAG) {
            start_col = row;
            cols = n - row;
        } else if (tri_type == TRI_UPPER) {
            if (row + 1u < n) {
                start_col = row + 1u;
                cols = n - start_col;
            }
        } else if (tri_type == TRI_LOWER_DIAG) {
            cols = row + 1u;
        } else {
            cols = row;
        }

        copy_row_run(row, start_col, cols, row_bytes);
    }
}

int main(void) {
    uint8_t nest_id = NEST_TARGET_ID;
    volatile int32_t * params =
        (volatile int32_t *)(uintptr_t)GTX_MAIN_ADDR(BASE_DDR_OP_PARAMS);

    uint32_t width = {{WIDTH}}u;
    uint32_t height = {{HEIGHT}}u;
    uint32_t tri_type = {{TRI_TYPE}}u;

    if ((uint32_t)params[OP_PARAM_MAGIC] == OP_PARAMS_MAGIC) {
        width = positive_or_u32(params[OP_PARAM_SHAPE_W], width);
        height = positive_or_u32(params[OP_PARAM_SHAPE_H], height);
        tri_type = valid_tri_type(params[OP_PARAM_TRI_TYPE]);
    }

    uint32_t n = min_u32(width, height);

    __split();
    {
        __start_plan(nest_id);

            __start_shared();
                zero_discarded_triangle(n, tri_type);
            __end_shared();

        __end_plan(nest_id);
    }
    __join();

    copy_kept_triangle(n, tri_type);

    return 0;
}
