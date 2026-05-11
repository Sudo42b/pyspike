#!/usr/bin/env python3
"""
E2E 테스트 데이터 생성기 — GTX NPU n1s16 커널

Usage:
  python3 generate_n1s16_tests.py n1s16_abs             # 단일 커널 input + ref 생성
  python3 generate_n1s16_tests.py --all                  # 전체 커널 생성
  python3 generate_n1s16_tests.py --input-only n1s16_abs # input만 생성 (ref는 Spike)
  python3 generate_n1s16_tests.py --output-size n1s16_abs # 출력 크기만 출력 (hex)
  python3 generate_n1s16_tests.py --list                 # 지원 커널 목록
"""

import numpy as np
import struct
import os
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import scalar ops from compare_all_ops.py
sys.path.insert(0, SCRIPT_DIR)
from compare_all_ops import (
    UNARY_OPS, OP_CONFIG,
    compute_scalar_unary, compute_scalar_binary,
    compute_scalar_parameterized, compute_scalar_reduction,
    compute_scalar_norm, compute_scalar_glu,
    compute_scalar_arange, compute_scalar_diag_mask,
    compute_scalar_cumsum, safe_sigmoid,
)


# ============================================================
# Hex 인코딩 유틸리티
# ============================================================

BYTES_PER_LINE = 32  # 256-bit bus word

# DDR address map (relative, before GTX_MAIN_OFFSET)
DDR_ADDR_A = 0x1000000
DDR_ADDR_B = 0x2000000
DDR_ADDR_R = 0xf000000


def fp16_to_bytes(values):
    """FP16 값 배열 → big-endian raw bytes."""
    arr = np.asarray(values, dtype=np.float16)
    raw = b''
    for v in arr:
        raw += struct.pack('>H', v.view(np.uint16))
    return raw


def bytes_to_hex_lines(raw_bytes, base_addr):
    """Raw bytes → @address + hex 라인 리스트.

    Spike의 ddr_init_from_file()이 hex를 읽을 때 내부적으로 bus-word reversal을
    수행하므로, hex 파일 자체는 FP16 big-endian 값을 left-to-right 순서로 기록한다.
    이는 compare_all_ops.py의 hex_to_fp16_array()와 호환된다.
    """
    lines = [f"@{base_addr:x}"]
    for i in range(0, len(raw_bytes), BYTES_PER_LINE):
        chunk = raw_bytes[i:i + BYTES_PER_LINE]
        if len(chunk) < BYTES_PER_LINE:
            chunk += b'\x00' * (BYTES_PER_LINE - len(chunk))
        lines.append(chunk.hex())
    return lines


def fp16_array_to_hex_lines(values, base_addr):
    """FP16 값 배열 → @address + bus-word-reversed hex 라인."""
    return bytes_to_hex_lines(fp16_to_bytes(values), base_addr)


def write_hex_file(filepath, lines):
    """Hex 라인 리스트를 파일로 저장."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def align32(n):
    """32바이트 경계로 올림 정렬."""
    return ((n + 31) // 32) * 32


# ============================================================
# 커널 메타데이터
# ============================================================

def _unary(w=8, h=64):
    return {'type': 'unary', 'width': w, 'height': h}

def _binary(w=8, h=64):
    return {'type': 'binary', 'width': w, 'height': h}

def _param(params, w=8, h=64):
    return {'type': 'param', 'width': w, 'height': h, 'params': params}

def _reduce_all(w=8, h=64):
    return {'type': 'reduce_all', 'width': w, 'height': h}

def _reduce_rows(w=8, h=64):
    return {'type': 'reduce_rows', 'width': w, 'height': h}

def _norm(w=8, h=64):
    return {'type': 'norm', 'width': w, 'height': h}

def _glu(w_in=16, w_out=8, h=64):
    return {'type': 'glu', 'width_in': w_in, 'width_out': w_out, 'height': h}

def _data_move(out_bytes):
    return {'type': 'data_move', 'out_bytes': out_bytes}

def _complex(out_bytes):
    return {'type': 'complex', 'out_bytes': out_bytes}


KERNEL_META = {
    # --- Unary element-wise ---
    'abs': _unary(), 'neg': _unary(), 'floor': _unary(), 'ceil': _unary(),
    'round': _unary(), 'trunc': _unary(), 'step': _unary(), 'sgn': _unary(),
    'exp': _unary(), 'gelu': _unary(), 'sigmoid': _unary(), 'tanh': _unary(),
    'leaky_relu': _unary(), 'relu': _unary(), 'hardswish': _unary(),
    'hardsigmoid': _unary(), 'silu': _unary(), 'gelu_quick': _unary(),
    'elu': _unary(), 'expm1': _unary(), 'softplus': _unary(),
    'sqr': _unary(), 'sqrt': _unary(), 'log': _unary(),
    'sin': _unary(), 'cos': _unary(), 'gelu_erf': _unary(), 'xielu': _unary(),
    # --- Binary ---
    'add_vv': _binary(), 'sub_vv': _binary(), 'div_vv': _binary(), 'acc': _binary(),
    # --- Parameterized ---
    'scale': _param([2.0]), 'add1': _param([0.5]),
    'clamp': _param([-0.5, 0.5]), 'fill': _param([1.0]),
    # --- Reductions ---
    'sum': _reduce_all(), 'sum_rows': _reduce_rows(), 'mean': _reduce_rows(),
    # --- Normalization ---
    'softmax': _norm(), 'norm': _norm(), 'rms_norm': _norm(),
    'l2_norm': _norm(), 'group_norm': _norm(),
    # --- GLU variants ---
    'glu': _glu(), 'reglu': _glu(), 'geglu': _glu(),
    'geglu_erf': _glu(), 'geglu_quick': _glu(), 'swiglu_oai': _glu(),
    # --- Data movement (input-only, ref from Spike) ---
    'dup': _data_move(0x400), 'cpy': _data_move(0x400),
    'set': _data_move(0x400), 'arange': _data_move(0x400),
    'repeat': _data_move(0x400), 'concat': _data_move(0x400),
    'pad': _data_move(0x400), 'roll': _data_move(0x400),
    'get_rows': _data_move(0x400), 'set_rows': _data_move(0x400),
    'upscale': _data_move(0x400), 'add_id': _data_move(0x400),
    'cont': _data_move(0x400), 'reshape': _data_move(0x400),
    'view': _data_move(0x200), 'transpose': _data_move(0x400),
    'permute': _data_move(0x400),
    # --- Special (partial scalar support) ---
    'diag_mask_inf': _unary(), 'diag_mask_zero': _unary(),
    'cumsum': _complex(0x2000),
    # --- Complex (input-only, ref from Spike) ---
    'pool_1d': _complex(0x80), 'pool_2d': _complex(0x80),
    'im2col': _complex(0x480), 'im2col_3d': _complex(0xD80),
    'conv_2d': _complex(0x80), 'conv_2d_dw': _complex(0x400),
    'conv_3d': _complex(0x80), 'conv_transpose_2d': _complex(0x60),
    'out_prod': _complex(0x400),
    'flash_attn_ext': _complex(0x80),
    'mul_mat': _complex(0x200), 'mul_mat_id': _complex(0x200),
    'solve_tri': _complex(0x200),
    'argmax': _complex(0x100), 'argsort': _complex(0x4000),
    'top_k': _complex(0x600), 'count_equal': _complex(0x20),
    'tri': _complex(0x2000), 'diag': _complex(0x2000),
    'pad_reflect_1d': _complex(0x800),
    'rope': _complex(0x400), 'add_rel_pos': _complex(0x400),
    'get_rel_pos': _complex(0x40), 'win_part': _complex(0x200),
    'win_unpart': _complex(0x200),
    'ssm_conv': _complex(0x400), 'ssm_scan': _complex(0x400),
    'timestep_embd': _complex(0x400), 'conv_tr1d': _complex(0x400),
    'rwkv_wkv6': _complex(0x20), 'rwkv_wkv7': _complex(0x20),
    'gated_linear_attn': _complex(0x20),
    'mul': _complex(0x200000),
}

# Kernel name → op name mapping (strip n1s16_ prefix)
def kernel_to_op(kernel_name):
    """n1s16_abs → abs, n1s16_add_vv → add_vv"""
    return kernel_name.replace('n1s16_', '', 1)


# ============================================================
# 출력 크기 계산
# ============================================================

def compute_output_size(op_name, meta=None):
    """커널 메타데이터에서 출력 바이트 수 계산 (32B 정렬)."""
    if meta is None:
        meta = KERNEL_META.get(op_name, {})

    ktype = meta.get('type', 'complex')

    if ktype in ('unary', 'binary', 'norm', 'param'):
        w = meta.get('width', 8)
        h = meta.get('height', 64)
        return align32(w * h * 2)

    elif ktype == 'reduce_all':
        return 0x20  # 32 bytes padded

    elif ktype == 'reduce_rows':
        h = meta.get('height', 64)
        return align32(h * 2)

    elif ktype == 'glu':
        w_out = meta.get('width_out', 8)
        h = meta.get('height', 64)
        return align32(w_out * h * 2)

    elif ktype in ('data_move', 'complex'):
        return meta.get('out_bytes', 0x400)

    return 0x400  # fallback


def get_output_size_from_ref(kernel_name):
    """ref.txt에서 데이터 라인 수로 출력 크기 계산. 없으면 None."""
    op_name = kernel_to_op(kernel_name)
    cfg = OP_CONFIG.get(op_name)
    if cfg is None:
        return None
    data_dir = os.path.join(SCRIPT_DIR, cfg['dir'], 'data')
    ref_file = os.path.join(data_dir, f"{kernel_name}_ref.txt")
    if not os.path.exists(ref_file):
        return None
    data_lines = 0
    with open(ref_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('@'):
                data_lines += 1
    if data_lines > 0:
        return data_lines * BYTES_PER_LINE
    return None


def get_output_size(kernel_name):
    """출력 크기 결정: ref.txt → 메타데이터 → fallback 순."""
    # 1) ref.txt에서 자동 감지
    ref_size = get_output_size_from_ref(kernel_name)
    if ref_size is not None:
        return ref_size

    # 2) 메타데이터에서 계산
    op_name = kernel_to_op(kernel_name)
    meta = KERNEL_META.get(op_name)
    if meta is not None:
        return compute_output_size(op_name, meta)

    # 3) fallback
    return 0x400


# ============================================================
# 입력 데이터 생성
# ============================================================

def generate_fp16_random(n_elements, seed=42, low=-2.0, high=2.0):
    """랜덤 FP16 값 배열 생성 (NaN/Inf 방지)."""
    rng = np.random.RandomState(seed)
    vals = rng.uniform(low, high, n_elements).astype(np.float16)
    # NaN/Inf를 0으로 치환
    vals = np.where(np.isfinite(vals), vals, np.float16(0.0))
    return vals


def generate_fp16_positive(n_elements, seed=42, low=0.01, high=2.0):
    """양수 FP16 값 (sqrt, log 등용)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(low, high, n_elements).astype(np.float16)


def generate_input(op_name, meta, seed=42):
    """커널 타입에 맞는 입력 데이터 생성.

    Returns: {ddr_addr: np.float16_array} dict
    """
    ktype = meta.get('type', 'complex')
    w = meta.get('width', 8)
    h = meta.get('height', 64)
    n = w * h

    if ktype == 'unary':
        # 양수만 필요한 연산
        if op_name in ('sqrt', 'log'):
            return {DDR_ADDR_A: generate_fp16_positive(n, seed)}
        return {DDR_ADDR_A: generate_fp16_random(n, seed)}

    elif ktype == 'binary':
        a = generate_fp16_random(n, seed)
        b = generate_fp16_random(n, seed + 1)
        # div: 0 방지
        if op_name == 'div_vv':
            b = np.where(b == 0, np.float16(1.0), b)
        return {DDR_ADDR_A: a, DDR_ADDR_B: b}

    elif ktype == 'param':
        a = generate_fp16_random(n, seed)
        params = meta.get('params', [1.0])
        param_arr = np.array(params, dtype=np.float16)
        # @2000000에 param 값 저장 (나머지 0 패딩)
        param_padded = np.zeros(max(16, len(params)), dtype=np.float16)
        param_padded[:len(params)] = param_arr
        return {DDR_ADDR_A: a, DDR_ADDR_B: param_padded}

    elif ktype in ('reduce_all', 'reduce_rows'):
        return {DDR_ADDR_A: generate_fp16_random(n, seed, low=-1.0, high=1.0)}

    elif ktype == 'norm':
        return {DDR_ADDR_A: generate_fp16_random(n, seed, low=-1.0, high=1.0)}

    elif ktype == 'glu':
        w_in = meta.get('width_in', 16)
        n_glu = w_in * h
        return {DDR_ADDR_A: generate_fp16_random(n_glu, seed, low=-1.0, high=1.0)}

    elif ktype == 'data_move':
        return {DDR_ADDR_A: generate_fp16_random(n, seed)}

    elif ktype == 'complex':
        # complex 커널은 입력 구조가 다양 — 기본 A만 생성
        return {DDR_ADDR_A: generate_fp16_random(n, seed)}

    return {DDR_ADDR_A: generate_fp16_random(n, seed)}


# ============================================================
# 레퍼런스 계산
# ============================================================

def compute_reference(op_name, meta, inputs):
    """NumPy 스칼라 연산으로 기대 출력 계산.

    Returns: np.float16_array or None (Spike fallback 필요)
    """
    ktype = meta.get('type', 'complex')
    a = inputs.get(DDR_ADDR_A)
    b = inputs.get(DDR_ADDR_B)
    w = meta.get('width', 8)

    if a is None:
        return None

    a32 = a.astype(np.float32)

    if ktype == 'unary':
        if op_name in UNARY_OPS:
            result = compute_scalar_unary(op_name, a32)
            return np.array(result, dtype=np.float16)
        elif op_name == 'diag_mask_inf':
            result = compute_scalar_diag_mask('diag_mask_inf', a32, w)
            return np.array(result, dtype=np.float16)
        elif op_name == 'diag_mask_zero':
            result = compute_scalar_diag_mask('diag_mask_zero', a32, w)
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'binary':
        if b is None:
            return None
        b32 = b.astype(np.float32)
        result = compute_scalar_binary(op_name, a32, b32)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'param':
        params = [float(v) for v in meta.get('params', [1.0])]
        result = compute_scalar_parameterized(op_name, a32, params)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'reduce_all':
        result = compute_scalar_reduction(op_name, a32, w)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'reduce_rows':
        result = compute_scalar_reduction(op_name, a32, w)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'norm':
        result = compute_scalar_norm(op_name, a32, w)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    elif ktype == 'glu':
        w_in = meta.get('width_in', 16)
        w_out = meta.get('width_out', 8)
        result = compute_scalar_glu(op_name, a32, w_in, w_out)
        if result is not None:
            return np.array(result, dtype=np.float16)
        return None

    return None


# ============================================================
# 파일 쓰기
# ============================================================

def write_test_data(kernel_name, input_only=False):
    """커널의 input.txt (+ ref.txt) 생성.

    Returns: (input_written, ref_written) bool tuple
    """
    op_name = kernel_to_op(kernel_name)
    meta = KERNEL_META.get(op_name)
    cfg = OP_CONFIG.get(op_name)

    if meta is None:
        print(f"[SKIP] {kernel_name}: 메타데이터 없음")
        return False, False

    if cfg is None:
        print(f"[SKIP] {kernel_name}: OP_CONFIG 없음")
        return False, False

    data_dir = os.path.join(SCRIPT_DIR, cfg['dir'], 'data')

    # Seed를 커널 이름에서 결정 (재현 가능)
    seed = hash(kernel_name) % (2**31)

    # 입력 생성
    inputs = generate_input(op_name, meta, seed)

    # input.txt 쓰기
    input_lines = []
    for addr in sorted(inputs.keys()):
        input_lines.extend(fp16_array_to_hex_lines(inputs[addr], addr))

    input_file = os.path.join(data_dir, f"{kernel_name}_input.txt")
    write_hex_file(input_file, input_lines)

    if input_only:
        print(f"[INPUT] {kernel_name}: {input_file}")
        return True, False

    # 레퍼런스 계산
    ref_arr = compute_reference(op_name, meta, inputs)

    if ref_arr is None:
        print(f"[INPUT] {kernel_name}: ref 자동 생성 불가 (--update-ref 사용)")
        return True, False

    # ref.txt 쓰기
    ref_lines = fp16_array_to_hex_lines(ref_arr, DDR_ADDR_R)
    ref_file = os.path.join(data_dir, f"{kernel_name}_ref.txt")
    write_hex_file(ref_file, ref_lines)

    out_size = get_output_size(kernel_name)
    print(f"[GEN] {kernel_name}: input + ref 생성 (out_size=0x{out_size:x})")
    return True, True


# ============================================================
# CLI
# ============================================================

def get_all_kernel_names():
    """OP_CONFIG에서 전체 커널 이름 목록."""
    return [cfg['kernel'] for cfg in OP_CONFIG.values()]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]

    # --output-size: 출력 크기만 출력
    if args[0] == '--output-size':
        kernel = args[1] if len(args) > 1 else None
        if kernel:
            size = get_output_size(kernel)
            print(f"0x{size:x}")
        sys.exit(0)

    # --list: 커널 목록
    if args[0] == '--list':
        for op_name, meta in KERNEL_META.items():
            ktype = meta.get('type', '?')
            out = compute_output_size(op_name, meta)
            kernel = f"n1s16_{op_name}"
            print(f"  {kernel:30s} {ktype:12s} out=0x{out:x}")
        sys.exit(0)

    # --input-only 플래그
    input_only = '--input-only' in args
    if input_only:
        args.remove('--input-only')

    # --all
    if args[0] == '--all':
        kernels = get_all_kernel_names()
        total, gen_input, gen_ref = 0, 0, 0
        for k in kernels:
            i, r = write_test_data(k, input_only)
            total += 1
            gen_input += int(i)
            gen_ref += int(r)
        print(f"\n총 {total}개: input={gen_input}, ref={gen_ref}")
        sys.exit(0)

    # 개별 커널
    for kernel in args:
        write_test_data(kernel, input_only)


if __name__ == '__main__':
    main()
