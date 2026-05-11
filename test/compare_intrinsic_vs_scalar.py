#!/usr/bin/env python3
"""
GTX Intrinsic vs Scalar 연산 비교
- GTX intrinsic: Spike ISS에서 실행한 결과
- Scalar: Python numpy FP16로 계산한 결과
- Reference: 테스트 데이터의 정답 (ref.txt)
"""

import numpy as np
import struct
import os
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def hex_to_fp16_array(hex_lines):
    """hex 문자열 배열 → FP16 float 배열 (big-endian)"""
    values = []
    for line in hex_lines:
        line = line.strip()
        if not line or line.startswith('@'):
            continue
        # 4 hex chars = 1 FP16 value (big-endian)
        for i in range(0, len(line), 4):
            h = line[i:i+4]
            if len(h) == 4:
                raw = int(h, 16)
                # big-endian FP16 → convert to float
                buf = struct.pack('>H', raw)
                val = np.frombuffer(buf, dtype='>e')[0]
                values.append(float(val))
    return np.array(values, dtype=np.float32)


def fp16_array_to_hex_lines(arr, elems_per_line=16):
    """float 배열 → hex 문자열 배열 (big-endian FP16)"""
    lines = []
    fp16_arr = arr.astype(np.float16)
    for i in range(0, len(fp16_arr), elems_per_line):
        chunk = fp16_arr[i:i+elems_per_line]
        hex_str = ''
        for v in chunk:
            buf = np.array([v], dtype='>e').tobytes()
            hex_str += buf.hex()
        lines.append(hex_str)
    return lines


def load_data(filepath):
    """입력/참조 파일 로드 → dict{address: hex_lines}"""
    sections = {}
    current_addr = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@'):
                current_addr = line
                sections[current_addr] = []
            elif line and current_addr is not None:
                sections[current_addr].append(line)
    return sections


def load_result_hex(filepath):
    """result.hex 로드 (@ 없는 순수 hex)"""
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('@'):
                lines.append(line)
    return lines


# ============================================================
# Scalar 연산 정의 (numpy FP16 정밀도)
# ============================================================

def scalar_add(a, b):
    return np.float16(a) + np.float16(b)

def scalar_gelu(x):
    """GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    x = float(x)
    c = math.sqrt(2.0 / math.pi)
    return np.float16(x * 0.5 * (1.0 + math.tanh(c * (x + 0.044715 * x * x * x))))

def scalar_sigmoid(x):
    x = float(x)
    if x >= 0:
        return np.float16(1.0 / (1.0 + math.exp(-x)))
    else:
        ex = math.exp(x)
        return np.float16(ex / (1.0 + ex))

def scalar_tanh(x):
    return np.float16(math.tanh(float(x)))

def scalar_relu(x):
    return np.float16(max(0.0, float(x)))

def scalar_silu(x):
    """SiLU: x * sigmoid(x)"""
    x = float(x)
    if x >= 0:
        s = 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        s = ex / (1.0 + ex)
    return np.float16(x * s)


def compute_scalar_op(op_name, inputs):
    """입력 데이터에 scalar 연산 적용"""
    if op_name == 'add_vv':
        a, b = inputs
        return np.array([scalar_add(a[i], b[i]) for i in range(len(a))], dtype=np.float32)

    a = inputs[0]
    ops = {
        'gelu': scalar_gelu,
        'sigmoid': scalar_sigmoid,
        'tanh': scalar_tanh,
        'relu': scalar_relu,
        'silu': scalar_silu,
    }
    func = ops[op_name]
    return np.array([float(func(v)) for v in a], dtype=np.float32)


def compare_arrays(name_a, arr_a, name_b, arr_b, max_show=10):
    """두 배열 비교, 차이점 출력"""
    if len(arr_a) != len(arr_b):
        print(f"  ⚠ 길이 불일치: {name_a}={len(arr_a)}, {name_b}={len(arr_b)}")
        min_len = min(len(arr_a), len(arr_b))
        arr_a = arr_a[:min_len]
        arr_b = arr_b[:min_len]

    diff_mask = np.abs(arr_a - arr_b) > 0
    # FP16 exact match
    ha = np.array(arr_a, dtype=np.float16)
    hb = np.array(arr_b, dtype=np.float16)
    exact_match = np.array_equal(ha.view(np.uint16), hb.view(np.uint16))

    # ULP difference
    ua = ha.view(np.uint16).astype(np.int32)
    ub = hb.view(np.uint16).astype(np.int32)
    ulp_diff = np.abs(ua - ub)

    n_exact = np.sum(ulp_diff == 0)
    n_1ulp = np.sum(ulp_diff == 1)
    n_2ulp = np.sum(ulp_diff == 2)
    n_gt2 = np.sum(ulp_diff > 2)
    max_ulp = int(np.max(ulp_diff))

    total = len(arr_a)
    print(f"  {name_a} vs {name_b}: exact={n_exact}/{total}, 1-ULP={n_1ulp}, 2-ULP={n_2ulp}, >2-ULP={n_gt2}, max_ULP={max_ulp}")

    if n_gt2 > 0:
        # 큰 차이 상세 출력
        gt2_idx = np.where(ulp_diff > 2)[0]
        show = min(max_show, len(gt2_idx))
        print(f"    >2-ULP 차이 상위 {show}개:")
        sorted_idx = gt2_idx[np.argsort(ulp_diff[gt2_idx])[::-1]]
        for idx in sorted_idx[:show]:
            print(f"      [{idx:4d}] {name_a}={arr_a[idx]:10.6f} (0x{ha[idx].view(np.uint16):04x})  "
                  f"{name_b}={arr_b[idx]:10.6f} (0x{hb[idx].view(np.uint16):04x})  "
                  f"ULP={ulp_diff[idx]}")

    return exact_match


def run_comparison(op_name, op_dir, input_file, ref_file, spike_result_file):
    """하나의 연산에 대해 3자 비교 수행"""
    print(f"\n{'='*70}")
    print(f"  {op_name.upper()} 비교")
    print(f"{'='*70}")

    # 1. 입력 로드
    input_data = load_data(input_file)
    ref_data = load_data(ref_file)

    addrs = sorted(input_data.keys())
    print(f"  Input 섹션: {addrs}")

    inputs = []
    for addr in addrs:
        arr = hex_to_fp16_array(input_data[addr])
        inputs.append(arr)
        print(f"    {addr}: {len(arr)} elements, range=[{arr.min():.4f}, {arr.max():.4f}]")

    # 2. Reference (정답) 로드
    ref_lines = []
    for addr in ref_data:
        ref_lines.extend(ref_data[addr])
    ref_arr = hex_to_fp16_array(ref_lines)
    print(f"  Reference: {len(ref_arr)} elements")

    # 3. GTX intrinsic (Spike) 결과 로드
    if os.path.exists(spike_result_file):
        spike_lines = load_result_hex(spike_result_file)
        spike_arr = hex_to_fp16_array(spike_lines)
        print(f"  GTX Intrinsic (Spike): {len(spike_arr)} elements")
    else:
        spike_arr = None
        print(f"  GTX Intrinsic (Spike): 결과 없음")

    # 4. Scalar 연산
    scalar_arr = compute_scalar_op(op_name, inputs)
    print(f"  Scalar (Python FP16): {len(scalar_arr)} elements")

    # 5. 샘플 값 출력
    print(f"\n  --- 처음 16개 값 비교 ---")
    print(f"  {'idx':>4s}  {'Input':>10s}  {'Scalar':>10s}  {'GTX':>10s}  {'Ref':>10s}")
    print(f"  {'':->4s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}")
    for i in range(min(16, len(ref_arr))):
        inp_str = f"{inputs[0][i]:10.4f}"
        scl_str = f"{scalar_arr[i]:10.4f}"
        gtx_str = f"{spike_arr[i]:10.4f}" if spike_arr is not None else "    N/A   "
        ref_str = f"{ref_arr[i]:10.4f}"
        print(f"  {i:4d}  {inp_str}  {scl_str}  {gtx_str}  {ref_str}")

    # 6. 정량 비교
    print(f"\n  --- ULP 비교 ---")
    if spike_arr is not None:
        compare_arrays("GTX", spike_arr, "Ref", ref_arr)
        compare_arrays("Scalar", scalar_arr, "Ref", ref_arr)
        compare_arrays("GTX", spike_arr, "Scalar", scalar_arr)
    else:
        compare_arrays("Scalar", scalar_arr, "Ref", ref_arr)


def main():
    base = SCRIPT_DIR

    ops = [
        {
            'name': 'add_vv',
            'dir': 'ADD/n1s16',
            'input': 'n1s16_add_vv_input.txt',
            'ref': 'n1s16_add_vv_ref.txt',
            'spike': '/tmp/add_vv_result.hex',
        },
        {
            'name': 'gelu',
            'dir': 'GELU/n1s16',
            'input': 'n1s16_gelu_input.txt',
            'ref': 'n1s16_gelu_ref.txt',
            'spike': '/tmp/gelu_result.hex',
        },
        {
            'name': 'sigmoid',
            'dir': 'SIGMOID/n1s16',
            'input': 'n1s16_sigmoid_input.txt',
            'ref': 'n1s16_sigmoid_ref.txt',
            'spike': '/tmp/sigmoid_result.hex',
        },
        {
            'name': 'tanh',
            'dir': 'TANH/n1s16',
            'input': 'n1s16_tanh_input.txt',
            'ref': 'n1s16_tanh_ref.txt',
            'spike': '/tmp/tanh_result.hex',
        },
        {
            'name': 'relu',
            'dir': 'RELU/n1s16',
            'input': 'n1s16_relu_input.txt',
            'ref': 'n1s16_relu_ref.txt',
            'spike': '/tmp/relu_result.hex',
        },
        {
            'name': 'silu',
            'dir': 'SILU/n1s16',
            'input': 'n1s16_silu_input.txt',
            'ref': 'n1s16_silu_ref.txt',
            'spike': '/tmp/silu_result.hex',
        },
    ]

    print("=" * 70)
    print("  GTX Intrinsic vs Scalar 연산 비교")
    print("  - GTX: Spike ISS에서 GTX NPU intrinsic으로 실행한 결과")
    print("  - Scalar: Python에서 FP16 정밀도로 계산한 결과")
    print("  - Ref: 테스트 데이터의 정답 (reference)")
    print("=" * 70)

    for op in ops:
        op_dir = os.path.join(base, op['dir'])
        data_dir = os.path.join(op_dir, 'data')
        run_comparison(
            op['name'],
            op_dir,
            os.path.join(data_dir, op['input']),
            os.path.join(data_dir, op['ref']),
            op['spike'],
        )

    print(f"\n{'='*70}")
    print("  비교 완료")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
