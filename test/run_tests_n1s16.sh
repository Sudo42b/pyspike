#!/bin/bash
# =================================================================
# Build and run all n1s16 kernel tests on GTX ISS
#
# Usage:
#   ./run_tests_n1s16.sh [--all | --build-only | kernel_name ...]
#   ./run_tests_n1s16.sh n1s16_abs             # test one kernel
#   ./run_tests_n1s16.sh --all                 # test all kernels
#   ./run_tests_n1s16.sh --e2e [KERNEL]         # E2E: generate → build → run → compare
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configurable paths (override via environment variables or .env file)
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

GFW="${GTX_FIRMWARE:-${GFW:-$PROJECT_ROOT/gtx-firmware}}"
if [ ! -d "$GFW/include" ]; then
    echo "ERROR: GTX firmware not found at $GFW"
    echo "  Run: git submodule update --init --recursive"
    echo "  Or set GTX_FIRMWARE=/path/to/gtx-firmware in .env"
    exit 1
fi

GISS="${GTX_ISS:-${ISS:-$PROJECT_ROOT/simulator/GTX_ISS}}"
SISS="${SISS:-$PROJECT_ROOT/riscv-isa-sim/build/spike}"
# Aliases used inside run_kernel_iss/run_kernel_spike (legacy variable names).
ISS="$GISS"
SPIKE="$SISS"
USE_SPIKE=0
USE_VFIO=0

CC="${CROSS_CC:-riscv64-unknown-elf-gcc}"
CFLAGS="-march=rv64g_xgtxnpu -mabi=lp64d -mcmodel=large -O0 -g -ffreestanding -nostartfiles \
  -ffunction-sections -fdata-sections -std=c11 \
  -DGTX_MAIN_OFFSET=0x370000000ULL \
  -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
  -Wno-missing-field-initializers -Wno-strict-aliasing \
  -Wno-incompatible-pointer-types -Wno-compare-distinct-pointer-types"
# Firmware sources moved to backup/ in latest gtx-firmware commit (cfafa1b).
# Use backup paths if the main include/ structure is missing.
GFW_INC="$GFW/include"
GFW_INTRIN="$GFW/include/gtx/intrinsics"
if [ ! -f "$GFW_INTRIN/intrin_level1.c" ] && [ -f "$GFW/backup/include/gtx/intrinsics/intrin_level1.c" ]; then
    GFW_INC="$GFW/backup/include"
    GFW_INTRIN="$GFW/backup/include/gtx/intrinsics"
fi
INCS="-I$SCRIPT_DIR -I$PROJECT_ROOT/example -I$GFW_INC -I$GFW_INC/gtx -I$GFW_INTRIN"
LDFLAGS="-T $SCRIPT_DIR/spike_gtx_link.ld -nostdlib -Wl,--gc-sections -lgcc"
COMMON_SRCS="$GFW_INTRIN/intrin_level1.c \
  $GFW_INTRIN/intrin_level2.c \
  $GFW_INTRIN/intrin_level3.c \
  $SCRIPT_DIR/spike_crt.S"

# --- Output size auto-detection ---
# Replaces hardcoded OUTPUT_SIZES map.
# Priority: ref.txt data lines × 32 → Python metadata → fallback 0x400
get_output_size() {
    local kernel=$1
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local ref="$kernel_dir/data/${kernel}_ref.txt"

    # 1) ref.txt에서 자동 감지 (데이터 라인 수 × 32바이트)
    if [ -f "$ref" ]; then
        local data_lines
        data_lines=$(grep -cv '^@' "$ref" 2>/dev/null || echo 0)
        # 빈 줄 제외
        data_lines=$(grep -cvE '^@|^$' "$ref" 2>/dev/null || echo 0)
        if [ "$data_lines" -gt 0 ]; then
            printf "0x%x" $((data_lines * 32))
            return
        fi
    fi

    # 2) Python 메타데이터에서 계산
    if [ -f "$SCRIPT_DIR/generate_n1s16_tests.py" ]; then
        local py_size
        py_size=$(python3 "$SCRIPT_DIR/generate_n1s16_tests.py" --output-size "$kernel" 2>/dev/null)
        if [ -n "$py_size" ]; then
            echo "$py_size"
            return
        fi
    fi

    # 3) fallback
    echo "0x400"
}

# Directory mapping: kernel_name → new relative directory path
declare -A KERNEL_DIR
KERNEL_DIR=(
    [n1s16_abs]=ABS/n1s16
    [n1s16_neg]=NEG/n1s16
    [n1s16_floor]=FLOOR/n1s16
    [n1s16_ceil]=CEIL/n1s16
    [n1s16_round]=ROUND/n1s16
    [n1s16_step]=STEP/n1s16
    [n1s16_sgn]=SGN/n1s16
    [n1s16_exp]=EXP/n1s16
    [n1s16_gelu]=GELU/n1s16
    [n1s16_sigmoid]=SIGMOID/n1s16
    [n1s16_tanh]=TANH/n1s16
    [n1s16_leaky_relu]=LEAKY_RELU/n1s16
    [n1s16_relu]=RELU/n1s16
    [n1s16_hardswish]=HARDSWISH/n1s16
    [n1s16_hardsigmoid]=HARDSIGMOID/n1s16
    [n1s16_silu]=SILU/n1s16
    [n1s16_gelu_quick]=GELU_QUICK/n1s16
    [n1s16_elu]=ELU/n1s16
    [n1s16_expm1]=EXPM1/n1s16
    [n1s16_softplus]=SOFTPLUS/n1s16
    [n1s16_sqr]=SQR/n1s16
    [n1s16_sqrt]=SQRT/n1s16
    [n1s16_log]=LOG/n1s16
    [n1s16_sin]=SIN/n1s16
    [n1s16_cos]=COS/n1s16
    [n1s16_scale]=SCALE/n1s16
    [n1s16_add1]=ADD1/n1s16
    [n1s16_clamp]=CLAMP/n1s16
    [n1s16_fill]=FILL/n1s16
    [n1s16_sub_vv]=SUB/n1s16
    [n1s16_div_vv]=DIV/n1s16
    [n1s16_acc]=ACC/n1s16
    [n1s16_dup]=DUP/n1s16
    [n1s16_cpy]=CPY/n1s16
    [n1s16_set]=SET/n1s16
    [n1s16_arange]=ARANGE/n1s16
    [n1s16_repeat]=REPEAT/n1s16
    [n1s16_concat]=CONCAT/n1s16
    [n1s16_pad]=PAD/n1s16
    [n1s16_roll]=ROLL/n1s16
    [n1s16_get_rows]=GET_ROWS/n1s16
    [n1s16_set_rows]=SET_ROWS/n1s16
    [n1s16_upscale]=UPSCALE/n1s16
    [n1s16_sum]=SUM/n1s16
    [n1s16_sum_rows]=SUM_ROWS/n1s16
    [n1s16_mean]=MEAN/n1s16
    [n1s16_softmax]=SOFT_MAX/n1s16
    [n1s16_norm]=NORM/n1s16
    [n1s16_rms_norm]=RMS_NORM/n1s16
    [n1s16_l2_norm]=L2_NORM/n1s16
    [n1s16_group_norm]=GROUP_NORM/n1s16
    [n1s16_diag_mask_inf]=DIAG_MASK_INF/n1s16
    [n1s16_diag_mask_zero]=DIAG_MASK_ZERO/n1s16
    [n1s16_pool_1d]=POOL_1D/n1s16
    [n1s16_pool_2d]=POOL_2D/n1s16
    [n1s16_im2col]=IM2COL/n1s16
    [n1s16_conv_2d]=CONV_2D/n1s16
    [n1s16_out_prod]=OUT_PROD/n1s16
    [n1s16_flash_attn_ext]=FLASH_ATTN_EXT/n1s16
    [n1s16_glu]=GLU/n1s16
    [n1s16_mul_mat]=MUL_MAT/n1s16
    [n1s16_add_rel_pos]=ADD_REL_POS/n1s16
    [n1s16_rope]=ROPE/n1s16
    [n1s16_get_rel_pos]=GET_REL_POS/n1s16
    [n1s16_win_part]=WIN_PART/n1s16
    [n1s16_ssm_conv]=SSM_CONV/n1s16
    [n1s16_ssm_scan]=SSM_SCAN/n1s16
    [n1s16_timestep_embd]=TIMESTEP_EMBEDDING/n1s16
    [n1s16_conv_tr1d]=CONV_TRANSPOSE_1D/n1s16
    [n1s16_add_vv]=ADD/n1s16
    [n1s16_trunc]=TRUNC/n1s16
    [n1s16_gelu_erf]=GELU_ERF/n1s16
    [n1s16_xielu]=XIELU/n1s16
    [n1s16_tri]=TRI/n1s16
    [n1s16_diag]=DIAG/n1s16
    [n1s16_cumsum]=CUMSUM/n1s16
    [n1s16_argmax]=ARGMAX/n1s16
    [n1s16_count_equal]=COUNT_EQUAL/n1s16
    [n1s16_pad_reflect_1d]=PAD_REFLECT_1D/n1s16
    [n1s16_argsort]=ARGSORT/n1s16
    [n1s16_top_k]=TOP_K/n1s16
    [n1s16_reglu]=REGLU/n1s16
    [n1s16_geglu]=GEGLU/n1s16
    [n1s16_swiglu_oai]=SWIGLU_OAI/n1s16
    [n1s16_geglu_erf]=GEGLU_ERF/n1s16
    [n1s16_geglu_quick]=GEGLU_QUICK/n1s16
    [n1s16_conv_2d_dw]=CONV_2D_DW/n1s16
    [n1s16_conv_transpose_2d]=CONV_TRANSPOSE_2D/n1s16
    [n1s16_im2col_3d]=IM2COL_3D/n1s16
    [n1s16_add_id]=ADD_ID/n1s16
    [n1s16_conv_3d]=CONV_3D/n1s16
    [n1s16_solve_tri]=SOLVE_TRI/n1s16
    [n1s16_mul_mat_id]=MUL_MAT_ID/n1s16
    # Recurrent attention ops
    [n1s16_rwkv_wkv6]=RWKV_WKV6/n1s16
    [n1s16_gated_linear_attn]=GATED_LINEAR_ATTN/n1s16
    [n1s16_rwkv_wkv7]=RWKV_WKV7/n1s16
    # Phase 6 — Non-runner test activation
    [n1s16_mul]=MUL/n1s16
    [n1s16_win_unpart]=WIN_UNPART/n1s16
    # Data movement ops
    [n1s16_cont]=CONT/n1s16
    [n1s16_reshape]=RESHAPE/n1s16
    [n1s16_view]=VIEW/n1s16
    [n1s16_transpose]=TRANSPOSE/n1s16
    [n1s16_permute]=PERMUTE/n1s16
)

# Kernels included in this phase. Inclusion criteria:
#   1) source file under $kernel_dir/$kernel.c exists,
#   2) data/${kernel}_input.txt and data/${kernel}_ref.txt exist,
#   3) kernel runs to completion on Spike (no hang/timeout) AND result matches ref.
#
# Per the phase rule "exclude any kernel that does not pass" the rest are listed
# in EXCLUDED_KERNELS below with the reason. KERNEL_DIR retains the full mapping
# so excluded kernels can be re-enabled once their issue is resolved.
ALL_KERNELS=(
    n1s16_abs
    n1s16_acc
    n1s16_add1
    n1s16_add_id
    n1s16_add_rel_pos
    n1s16_add_vv
    n1s16_arange
    n1s16_clamp
    n1s16_concat
    n1s16_conv_2d
    n1s16_conv_tr1d
    n1s16_conv_transpose_2d
    n1s16_cos
    n1s16_cpy
    n1s16_cumsum
    n1s16_diag
    n1s16_diag_mask_inf
    n1s16_diag_mask_zero
    n1s16_div_vv
    n1s16_dup
    n1s16_elu
    n1s16_exp
    n1s16_expm1
    n1s16_fill
    n1s16_floor
    n1s16_gated_linear_attn
    n1s16_geglu
    n1s16_gelu
    n1s16_get_rows
    n1s16_gelu_erf
    n1s16_gelu_quick
    n1s16_hardsigmoid
    n1s16_hardswish
    n1s16_im2col
    n1s16_im2col_3d
    n1s16_l2_norm
    n1s16_leaky_relu
    n1s16_mean
    n1s16_mul
    n1s16_mul_mat
    n1s16_mul_mat_id
    n1s16_neg
    n1s16_out_prod
    n1s16_pad
    n1s16_pad_reflect_1d
    n1s16_pool_1d
    n1s16_pool_2d
    n1s16_reglu
    n1s16_relu
    n1s16_repeat
    n1s16_rms_norm
    n1s16_roll
    n1s16_rope
    n1s16_round
    n1s16_rwkv_wkv6
    n1s16_rwkv_wkv7
    n1s16_scale
    n1s16_set
    n1s16_set_rows
    n1s16_sgn
    n1s16_sigmoid
    n1s16_silu
    n1s16_sin
    n1s16_softmax
    n1s16_step
    n1s16_sub_vv
    n1s16_sum
    n1s16_swiglu_oai
    n1s16_tanh
    n1s16_timestep_embd
    n1s16_tri
    n1s16_trunc
    n1s16_xielu
)

# Excluded from this phase. Categories:
#   - data missing       : no input.txt and/or ref.txt under data/
#   - data prefix mismatch: data files exist but use a different name prefix
#   - hang on Spike      : kernel builds but exceeds SPIKE_TIMEOUT consistently;
#                          decoder funct7 set matches Spike's dispatch table,
#                          so root cause is runtime semantics (rs3/rs4 staging,
#                          credit.ld.chk, plan/thread side effects) not encoding
#   - ref mismatch       : runs to completion but result diff != ref
EXCLUDED_KERNELS=(
    # --- data missing or name mismatch ---
    n1s16_argmax           # data missing
    n1s16_argsort          # data missing
    n1s16_ceil             # data missing
    n1s16_cont             # data missing
    n1s16_conv_2d_dw       # data missing
    n1s16_conv_3d          # data missing
    n1s16_count_equal      # data missing
    n1s16_flash_attn_ext   # data missing
    n1s16_get_rel_pos      # data prefix mismatch (yaml_get_rel_pos_*)
    n1s16_glu              # data missing
    n1s16_permute          # data missing
    n1s16_reshape          # data missing
    n1s16_softplus         # data missing
    n1s16_solve_tri        # data missing
    n1s16_sqr              # data missing
    n1s16_sqrt             # data missing
    n1s16_ssm_conv         # data missing
    n1s16_ssm_scan         # data missing
    n1s16_sum_rows         # data missing
    n1s16_top_k            # data missing
    n1s16_transpose        # data missing
    n1s16_upscale          # data missing
    n1s16_view             # data missing
    n1s16_win_part         # data missing
    n1s16_win_unpart       # data missing
    # --- ref mismatch even with FP16 ULP=2 / atol 0.01 tolerance ---
    # (remaining semantics gaps:
    #  reduction precision (FP32 accum), MATH_V LOG, 2D pool, DMA repeat)
    n1s16_group_norm       # FAIL: result all-zero (reduction/L0-broadcast chain)
    n1s16_log              # FAIL: result all-zero (MATH_V LN + CPU L1 sync)
    n1s16_norm             # FAIL: result all-zero (reduction/sqrt L0 path)
    n1s16_fill             # FAIL: ULP2 fail
    n1s16_get_rows         # FAIL: ULP2 fail
    n1s16_group_norm       # FAIL: ULP2 fail
    n1s16_l2_norm          # FAIL: ULP2 fail
    n1s16_log              # FAIL: ULP2 fail
    n1s16_mean             # FAIL: ULP2 fail
    n1s16_norm             # FAIL: ULP2 fail
    n1s16_pool_1d          # FAIL: ULP2 fail
    n1s16_pool_2d          # FAIL: ULP2 fail
    n1s16_repeat           # FAIL: ULP2 fail
    n1s16_rms_norm         # FAIL: ULP2 fail (was ULP1 ≈ 98.7%; spike fix shifted result)
    n1s16_sum              # FAIL: ULP2 fail
    n1s16_tri              # FAIL: ULP2 fail
    # --- timeout/slow (>300s) ---
    n1s16_geglu_erf        # timeout/slow
    n1s16_geglu_quick      # timeout/slow
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
BUILD_FAIL_COUNT=0
TIMEOUT_COUNT=0
RESULTS=()

build_kernel() {
    local kernel=$1
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local src="$kernel_dir/$kernel.c"
    local elf="$kernel_dir/$kernel.elf"
    
    if [ ! -f "$src" ]; then
        echo "[SKIP] $kernel: source not found"
        return 1
    fi
    
    $CC $CFLAGS $INCS "$src" $COMMON_SRCS $LDFLAGS -o "$elf" 2>/dev/null
    return $?
}

run_kernel_iss() {
    local kernel=$1
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local elf="$kernel_dir/$kernel.elf"
    local data_dir="$kernel_dir/data"
    local input="$data_dir/${kernel}_input.txt"
    local result="$data_dir/${kernel}_result.hex"
    local ref="$data_dir/${kernel}_ref.txt"
    local output_size
    output_size=$(get_output_size "$kernel")

    if [ ! -f "$elf" ]; then
        echo "[SKIP] $kernel: ELF not found"
        return 1
    fi
    if [ ! -f "$input" ]; then
        echo "[SKIP] $kernel: input data not found"
        return 1
    fi

    # Run ISS (GTX_MAIN_ADDR offset mode: -S 0x370000000, -V disable virtual device)
    "$ISS" -I "$elf" -S 0x370000000 -L "$input" \
        -B 0x37f000000 -E "$output_size" \
        -T "$result" -V -l 0 2>/dev/null
    
    return $?
}

run_kernel_spike() {
    local kernel=$1
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local elf="$kernel_dir/$kernel.elf"
    local data_dir="$kernel_dir/data"
    local input="$data_dir/${kernel}_input.txt"
    local result="$data_dir/${kernel}_result.hex"
    local output_size
    output_size=$(get_output_size "$kernel")

    if [ ! -f "$elf" ]; then
        echo "[SKIP] $kernel: ELF not found"
        return 1
    fi
    if [ ! -x "$SPIKE" ]; then
        echo "[ERROR] Spike not found at $SPIKE"
        echo "  Set SPIKE env var or build spike first: cd build && ../configure && make"
        return 1
    fi
    if [ ! -f "$input" ]; then
        echo "[SKIP] $kernel: input data not found"
        return 1
    fi

    # Run on Spike with GTX NPU extension
    # DDR environment: init from input, dump output region at 0x37f000000
    # GTX_DDR_REVERSED=1: data files use SystemC memory order (rightmost byte = offset 0)
    local rc=0
    LD_LIBRARY_PATH="${SPIKE%/*}" \
    GTX_DDR_REVERSED=1 \
    GTX_DDR_INIT="$input" \
    GTX_DDR_DUMP="$result" \
    GTX_DDR_DUMP_ADDR=0x37f000000 \
    GTX_DDR_DUMP_SIZE="$output_size" \
    timeout "${SPIKE_TIMEOUT:-120}" "$SPIKE" --extension=gtx_npu "$elf" 2>/dev/null || rc=$?

    # timeout returns 124 on timeout, 137 on KILL
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        echo "[TIMEOUT] $kernel"
        return 2
    fi
    return $rc
}

# start_vfio_server() {
#     local server="$PROJECT_ROOT/riscv-isa-sim/build/gtx_vfio_server"
#     local spike_path="$SPIKE"
#     local lib_path="${SPIKE%/*}"  # dirname of spike binary

#     if [ ! -x "$server" ]; then
#         echo "[ERROR] gtx_vfio_server not found at $server"
#         echo "  Run: ./setup.sh --build-vfio"
#         return 1
#     fi

#     echo "Starting VFIO server: $VFIO_SOCK"
#     "$server" "$VFIO_SOCK" "$spike_path" "$lib_path" 2>/dev/null &
#     VFIO_SERVER_PID=$!

#     # Wait for socket (up to 10 seconds)
#     for i in $(seq 1 100); do
#         if [ -S "$VFIO_SOCK" ]; then
#             echo "VFIO server ready (PID $VFIO_SERVER_PID)"
#             return 0
#         fi
#         if ! kill -0 "$VFIO_SERVER_PID" 2>/dev/null; then
#             echo "[ERROR] VFIO server exited before creating socket"
#             return 1
#         fi
#         sleep 0.1
#     done
#     echo "[ERROR] VFIO socket not created within 10s"
#     return 1
# }

# run_kernel_vfio() {
#     local kernel=$1
#     local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
#     local elf="$kernel_dir/$kernel.elf"
#     local data_dir="$kernel_dir/data"
#     local input="$data_dir/${kernel}_input.txt"
#     local result="$data_dir/${kernel}_result.hex"
#     local output_size
#     output_size=$(get_output_size "$kernel")

#     if [ ! -f "$elf" ]; then
#         echo "[SKIP] $kernel: ELF not found"
#         return 1
#     fi
#     if [ ! -f "$input" ]; then
#         echo "[SKIP] $kernel: input data not found"
#         return 1
#     fi

#     local client="$PROJECT_ROOT/riscv-isa-sim/build/gtx_test_client"
#     if [ ! -x "$client" ]; then
#         echo "[ERROR] gtx_test_client not found"
#         return 1
#     fi

#     # Check if server is still running; restart if needed
#     if [ -n "$VFIO_SERVER_PID" ] && ! kill -0 "$VFIO_SERVER_PID" 2>/dev/null; then
#         echo "VFIO server exited, restarting..."
#         rm -f "$VFIO_SOCK"
#         start_vfio_server || return 1
#     fi

#     # Run via VFIO client (server must be running)
#     local rc=0
#     timeout "${SPIKE_TIMEOUT:-60}" "$client" "$VFIO_SOCK" \
#         --run-kernel "$elf" "$input" "$result" "$output_size" || rc=$?

#     if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
#         echo "[TIMEOUT] $kernel"
#         return 2
#     fi
#     return $rc
# }

run_kernel() {
    if [ "$USE_VFIO" = "1" ]; then
        run_kernel_vfio "$@"
    elif [ "$USE_SPIKE" = "1" ]; then
        run_kernel_spike "$@"
    else
        run_kernel_iss "$@"
    fi
}

compare_result() {
    local kernel=$1
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local data_dir="$kernel_dir/data"
    local result="$data_dir/${kernel}_result.hex"
    local ref="$data_dir/${kernel}_ref.txt"

    if [ ! -f "$result" ] || [ ! -f "$ref" ]; then
        return 1
    fi

    # Strip @ lines from ref
    grep -v '^@' "$ref" > "/tmp/_ref_${kernel}.txt"

    # 1) strict byte-level diff
    if diff -i "$result" "/tmp/_ref_${kernel}.txt" > /dev/null 2>&1; then
        rm -f "/tmp/_ref_${kernel}.txt"
        return 0
    fi

    # 2) FP16 ULP fallback (default ULP=2, atol=0.01) — disable with COMPARE_STRICT=1
    if [ "${COMPARE_STRICT:-0}" != "1" ] && [ -f "$PROJECT_ROOT/gtx/verify.py" ]; then
        local ulp="${COMPARE_ULP:-2}"
        local atol="${COMPARE_ATOL:-0.01}"
        if python3 "$PROJECT_ROOT/gtx/verify.py" "$result" "/tmp/_ref_${kernel}.txt" \
             --fp16 --ulp "$ulp" --atol "$atol" > /dev/null 2>&1; then
            rm -f "/tmp/_ref_${kernel}.txt"
            return 0
        fi
    fi

    rm -f "/tmp/_ref_${kernel}.txt"
    return 1
}

test_kernel() {
    local kernel=$1
    local update_ref=${2:-0}
    
    # Build (skip if ELF already exists)
    local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
    local elf="$kernel_dir/$kernel.elf"
    if [ ! -f "$elf" ]; then
        if ! build_kernel "$kernel" 2>/dev/null; then
            echo "[BUILD_FAIL] $kernel"
            BUILD_FAIL_COUNT=$((BUILD_FAIL_COUNT + 1))
            RESULTS+=("[BUILD_FAIL] $kernel")
            return
        fi
    fi
    
    # Run ISS/Spike
    local run_rc=0
    run_kernel "$kernel" || run_rc=$?
    if [ $run_rc -eq 2 ]; then
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        RESULTS+=("[TIMEOUT] $kernel")
        return
    elif [ $run_rc -ne 0 ]; then
        echo "[RUN_FAIL] $kernel"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("[RUN_FAIL] $kernel")
        return
    fi
    
    if [ "$update_ref" = "1" ]; then
        # Use ISS output as reference
        local kernel_dir="$SCRIPT_DIR/${KERNEL_DIR[$kernel]}"
        local data_dir="$kernel_dir/data"
        local result="$data_dir/${kernel}_result.hex"
        local ref="$data_dir/${kernel}_ref.txt"
        echo "@f000000" > "$ref"
        cat "$result" >> "$ref"
        echo "[REF_UPDATED] $kernel"
        RESULTS+=("[REF_UPDATED] $kernel")
        return
    fi
    
    # Compare
    if compare_result "$kernel"; then
        echo "[PASS] $kernel"
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("[PASS] $kernel")
    else
        echo "[FAIL] $kernel"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("[FAIL] $kernel")
    fi
}

# ============================================================================
# Main
# ============================================================================

# Parse mode flags (--spike, --vfio)
while [ $# -gt 0 ]; do
    case "$1" in
        --spike) USE_SPIKE=1; shift ;;
        --vfio)  USE_VFIO=1; shift ;;
        *) break ;;
    esac
done

if [ "$USE_VFIO" = "1" ]; then
    start_vfio_server || exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--spike|--vfio] [--all | --build-only | --generate | --e2e | --update-ref KERNEL | kernel_name ...]"
    echo "  --spike      Run on Spike (spike --extension=gtx_npu) instead of GTX ISS"
    echo "  --vfio       Run via VFIO client/server (gtx_vfio_server + gtx_test_client)"
    echo "  --generate   Generate test data (input + ref) via Python"
    echo "  --e2e        E2E: generate → build → run → compare"
    echo "  Total kernels: ${#ALL_KERNELS[@]}"
    exit 1
fi

case "$1" in
    --generate)
        shift
        if [ "${1:-}" = "--all" ] || [ $# -eq 0 ]; then
            echo "=== Generating test data for all n1s16 kernels ==="
            python3 "$SCRIPT_DIR/generate_n1s16_tests.py" --all
        else
            for kernel in "$@"; do
                python3 "$SCRIPT_DIR/generate_n1s16_tests.py" "$kernel"
            done
        fi
        ;;
    --e2e)
        shift
        if [ "${1:-}" = "--all" ] || [ $# -eq 0 ]; then
            echo "=== E2E: generate → build → run → compare (all kernels) ==="
            echo ""
            for kernel in "${ALL_KERNELS[@]}"; do
                python3 "$SCRIPT_DIR/generate_n1s16_tests.py" "$kernel" 2>/dev/null || true
                test_kernel "$kernel"
            done
        else
            echo "=== E2E: generate → build → run → compare ==="
            echo ""
            for kernel in "$@"; do
                python3 "$SCRIPT_DIR/generate_n1s16_tests.py" "$kernel" 2>/dev/null || true
                test_kernel "$kernel"
            done
        fi
        echo ""
        echo "========================================"
        echo "Results: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  TIMEOUT=$TIMEOUT_COUNT  BUILD_FAIL=$BUILD_FAIL_COUNT"
        echo "========================================"
        ;;
    --build-only)
        echo "=== Building all n1s16 kernels ==="
        for kernel in "${ALL_KERNELS[@]}"; do
            if build_kernel "$kernel" 2>/dev/null; then
                echo "[BUILD_OK] $kernel"
            else
                echo "[BUILD_FAIL] $kernel"
                BUILD_FAIL_COUNT=$((BUILD_FAIL_COUNT + 1))
            fi
        done
        echo ""
        echo "Build failures: $BUILD_FAIL_COUNT / ${#ALL_KERNELS[@]}"
        ;;
    --all)
        echo "=== Testing all n1s16 kernels ==="
        echo ""
        for kernel in "${ALL_KERNELS[@]}"; do
            test_kernel "$kernel"
        done
        echo ""
        echo "========================================"
        echo "Results: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  TIMEOUT=$TIMEOUT_COUNT  BUILD_FAIL=$BUILD_FAIL_COUNT  SKIP=$SKIP_COUNT"
        echo "Total: ${#ALL_KERNELS[@]}"
        echo "========================================"
        ;;
    --update-ref)
        shift
        if [ "$1" = "--all" ]; then
            for kernel in "${ALL_KERNELS[@]}"; do
                test_kernel "$kernel" 1
            done
        else
            for kernel in "$@"; do
                test_kernel "$kernel" 1
            done
        fi
        ;;
    *)
        for kernel in "$@"; do
            test_kernel "$kernel"
        done
        echo ""
        echo "Results: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  TIMEOUT=$TIMEOUT_COUNT  BUILD_FAIL=$BUILD_FAIL_COUNT"
        ;;
esac
