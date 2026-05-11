#!/bin/bash
#============================================================================
# bench_backends.sh — Compare Manual / OpenBLAS / CUDA build performance
#
# Usage:
#   ./bench_backends.sh                  # all 103 tests, 1 run each
#   ./bench_backends.sh --runs 3         # 3 runs per build (median)
#   ./bench_backends.sh --only-cuda      # skip manual/BLAS, CUDA only
#   ./bench_backends.sh --per-test       # show per-test wall clock breakdown
#============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPIKE_BUILD="$PROJECT_ROOT/riscv-isa-sim/build"
SPIKE="$SPIKE_BUILD/spike"
LIB="$SPIKE_BUILD/libgtx_npu.so"

RUNS=1
ONLY_CUDA=false
PER_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)   RUNS="$2"; shift 2 ;;
        --only-cuda) ONLY_CUDA=true; shift ;;
        --per-test) PER_TEST=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Collect all test kernels from run_tests_n1s16.sh
cd "$SCRIPT_DIR"
source <(grep -E '^ALL_KERNELS=|^OUTPUT_SIZES=|^KERNEL_DIR=' run_tests_n1s16.sh 2>/dev/null || true)

if [ -z "${ALL_KERNELS+x}" ]; then
    # Fallback: extract from script
    ALL_KERNELS=($(grep -oP 'n1s16_\w+' run_tests_n1s16.sh | sort -u))
fi

TOTAL=${#ALL_KERNELS[@]}
echo "=============================================="
echo " Backend Benchmark: $TOTAL tests × $RUNS run(s)"
echo "=============================================="
echo ""

# Build configurations
declare -a BUILD_NAMES BUILD_ENVS
if [ "$ONLY_CUDA" = false ]; then
    BUILD_NAMES+=("Manual")
    BUILD_ENVS+=("")
    BUILD_NAMES+=("OpenBLAS")
    BUILD_ENVS+=("GTX_USE_BLAS=1")
fi
BUILD_NAMES+=("CUDA")
BUILD_ENVS+=("CUDA_HOME=/usr/local/cuda-13.1 GTX_USE_CUDA=1")

RESULTS_DIR="/tmp/bench_backends_$$"
mkdir -p "$RESULTS_DIR"

run_test_suite() {
    local build_name="$1"
    local run_num="$2"
    local outfile="$RESULTS_DIR/${build_name}_run${run_num}.txt"

    local pass=0 fail=0 timeout=0

    for kernel in "${ALL_KERNELS[@]}"; do
        # Find kernel directory
        local kdir=""
        for dir in "$SCRIPT_DIR"/*/n1s16/; do
            if [ -f "$dir/${kernel}.elf" ]; then
                kdir="$dir"
                break
            fi
        done
        if [ -z "$kdir" ]; then
            continue
        fi

        local elf="$kdir/${kernel}.elf"
        local input="$kdir/data/${kernel}_input.txt"
        local ref="$kdir/data/${kernel}_ref.txt"

        if [ ! -f "$elf" ] || [ ! -f "$input" ]; then
            continue
        fi

        # Run with timing
        local t_start=$(date +%s%N)
        local result_hex="/tmp/bench_${build_name}_${kernel}.hex"

        timeout 30 env \
            GTX_DDR_INIT="$input" \
            GTX_DDR_DUMP="$result_hex" \
            GTX_DDR_DUMP_ADDR=0x37f000000 \
            GTX_DDR_DUMP_SIZE=0x2000 \
            GTX_DDR_REVERSED=1 \
            LD_LIBRARY_PATH="$SPIKE_BUILD" \
            "$SPIKE" --extension=gtx_npu "$elf" 2>/dev/null
        local exit_code=$?

        local t_end=$(date +%s%N)
        local elapsed_ms=$(( (t_end - t_start) / 1000000 ))

        local status="PASS"
        if [ $exit_code -ne 0 ]; then
            if [ $exit_code -eq 124 ]; then
                status="TIMEOUT"
                ((timeout++))
            else
                status="FAIL"
                ((fail++))
            fi
        else
            ((pass++))
        fi

        if [ "$PER_TEST" = true ]; then
            printf "  %-30s %6dms  %s\n" "$kernel" "$elapsed_ms" "$status"
        fi

        echo "$kernel $elapsed_ms $status" >> "$outfile"
    done

    # Total time
    local total_ms=0
    if [ -f "$outfile" ]; then
        total_ms=$(awk '{sum+=$2} END{print sum}' "$outfile")
    fi
    local total_s=$(echo "scale=1; $total_ms / 1000" | bc)

    echo "$build_name run$run_num: ${total_s}s (PASS=$pass FAIL=$fail TIMEOUT=$timeout)" >> "$RESULTS_DIR/summary.txt"
    echo "$total_ms" >> "$RESULTS_DIR/${build_name}_totals.txt"
}

# Run each build
for i in "${!BUILD_NAMES[@]}"; do
    build_name="${BUILD_NAMES[$i]}"
    build_env="${BUILD_ENVS[$i]}"

    echo "--- Building: $build_name ---"
    cd "$PROJECT_ROOT"
    if [ -n "$build_env" ]; then
        eval "$build_env ./setup.sh --rebuild" 2>/dev/null
    else
        ./setup.sh --rebuild 2>/dev/null
    fi
    cd "$SCRIPT_DIR"

    for run in $(seq 1 $RUNS); do
        echo "  Run $run/$RUNS..."
        if [ "$PER_TEST" = true ]; then
            echo ""
        fi
        run_test_suite "$build_name" "$run"
    done
    echo ""
done

# Summary
echo "=============================================="
echo " RESULTS"
echo "=============================================="
echo ""

cat "$RESULTS_DIR/summary.txt"
echo ""

# Median calculation if multiple runs
if [ "$RUNS" -gt 1 ]; then
    echo "--- Median Wall Clock ---"
    for build_name in "${BUILD_NAMES[@]}"; do
        if [ -f "$RESULTS_DIR/${build_name}_totals.txt" ]; then
            median=$(sort -n "$RESULTS_DIR/${build_name}_totals.txt" | awk "NR==$(( (RUNS+1)/2 ))")
            median_s=$(echo "scale=1; $median / 1000" | bc)
            echo "  $build_name: ${median_s}s"
        fi
    done
    echo ""
fi

# Per-test comparison (top 10 slowest)
echo "--- Top 10 Slowest Tests (first run) ---"
for build_name in "${BUILD_NAMES[@]}"; do
    echo ""
    echo "  [$build_name]"
    if [ -f "$RESULTS_DIR/${build_name}_run1.txt" ]; then
        sort -k2 -rn "$RESULTS_DIR/${build_name}_run1.txt" | head -10 | \
            awk '{printf "    %-30s %6dms  %s\n", $1, $2, $3}'
    fi
done

# Cleanup
rm -rf "$RESULTS_DIR"
echo ""
echo "Done."
