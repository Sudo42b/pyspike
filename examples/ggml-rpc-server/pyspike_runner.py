"""pyspike .elf execution adapter — generates per-shape firmware on demand,
builds it with build_kernel.sh, and runs it under pyspike.

Cache layout: $HOME/.cache/pyspike-rpc/kernels/<op>_<shape_key>.elf so each
(op, shape) tuple is built at most once across server restarts.

Currently supports: UNARY ABS for 1D fp16 tensors whose element count is a
multiple of WIDTH=8. Adding a new op = drop a template under
`firmware_templates/<op>.c.tpl` and add a small dispatcher entry below.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TEMPLATES_DIR = HERE / "firmware_templates"
CACHE_DIR = Path(os.environ.get("PYSPIKE_RPC_KERNEL_CACHE",
                                Path.home() / ".cache/pyspike-rpc/kernels"))

# n1s16 firmware geometry (matches the templates).
WIDTH = 8                              # fp16 elements per row
DTYPE_BYTES = 2                        # fp16
ROW_BYTES = WIDTH * DTYPE_BYTES        # 16

# Pyspike runtime env — mirrors test/verify_pyspike_ggml.run_op().
_PYSPIKE_ENV = dict(
    GTX_MX_IO_DTYPE="float16",
    GTX_DDR_REVERSE_MODE="elem",
    GTX_DDR_SIZE="2G",
    GTX_DDR_REVERSED="1",
    UV_LINK_MODE="copy",
)

DEFAULT_DUMP_ADDR = 0x37f000000
DEFAULT_DEVICE_BASE = 0x370000000
DEFAULT_INPUT_OFFSET = 0x1000000       # = BASE_DDR_A in the template


def _build_script() -> Path:
    """Path to the canonical build_kernel.sh (same one verify_pyspike_ggml.py uses)."""
    p = REPO_ROOT / "src/test/gtx/build_kernel.sh"
    if not p.is_file():
        raise FileNotFoundError(f"missing build_kernel.sh at {p}")
    return p


def _resolve_firmware_root() -> str:
    """Best-effort GTX firmware location. Caller can override via GTX_FIRMWARE."""
    if env := os.environ.get("GTX_FIRMWARE"):
        return env
    for c in ("/home/owner/gtx-firmware",
              "/home/sw.lee/supergate_sw/device/gtx-firmware"):
        if Path(c).is_dir():
            return c
    raise FileNotFoundError("set GTX_FIRMWARE — couldn't auto-detect")


def _resolve_kernel_inc() -> str:
    """Best-effort GTX_KERNEL_INC (holds `gtx/address.h`)."""
    if env := os.environ.get("GTX_KERNEL_INC"):
        return env
    for c in ("/home/sw.lee/supergate_sw/device/gtx_kernel/dsppp/src/include",
              "/home/owner/gtx_kernel/dsppp/src/include"):
        if Path(c).is_dir():
            return c
    raise FileNotFoundError("set GTX_KERNEL_INC — couldn't auto-detect")


def _render_template(op_name: str, **vars) -> str:
    tpl_path = TEMPLATES_DIR / f"{op_name}.c.tpl"
    if not tpl_path.is_file():
        raise FileNotFoundError(f"no template for op '{op_name}' at {tpl_path}")
    src = tpl_path.read_text()
    for k, v in vars.items():
        src = src.replace("{{" + k + "}}", str(v))
    return src


def _cache_key(op_name: str, dtype: str, ne: tuple) -> str:
    shape = "x".join(str(n) for n in ne)
    return f"{op_name}_{dtype}_{shape}"


def _build_kernel(c_src_text: str, cache_key: str) -> Path:
    """Build a kernel via the canonical build_kernel.sh — same flags
    (-march=rv64g_xgtxnpu, -lgcc, etc.) that verify_pyspike_ggml.py uses, so a
    new template behaves identically to the test/ corpus reference kernels.

    `test/gtx_csr.h` is on the template's include list, so the source is staged
    in a temp dir alongside a symlink/copy of the relevant headers via
    EXTRA_INC (passed through to the build script).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_elf = CACHE_DIR / f"{cache_key}.elf"
    if out_elf.exists():
        return out_elf

    src_dir = CACHE_DIR / f"{cache_key}_build"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_path = src_dir / f"{cache_key}.c"
    src_path.write_text(c_src_text)

    # Stage shared template headers (e.g. gtx_kernel.h for the CUDA-style
    # launch macro) next to the source so the compiler resolves them without
    # extending the build script's include search path.
    import shutil
    for header in TEMPLATES_DIR.glob("*.h"):
        shutil.copy(header, src_dir / header.name)

    try:
        script = _build_script()
        gfw = _resolve_firmware_root()
        kinc = _resolve_kernel_inc()
        # gtx_csr.h lives under test/; expose it via EXTRA_INC so the template's
        # `#include "gtx_csr.h"` resolves under build_kernel.sh.
        extra_inc = str(REPO_ROOT / "test")
        env = dict(os.environ,
                   GTX_FIRMWARE=gfw, GTX_KERNEL_INC=kinc,
                   EXTRA_INC=extra_inc,
                   PATH=f"/opt/riscv/bin:{os.environ.get('PATH', '')}")
        r = subprocess.run(
            ["bash", str(script), str(src_path), str(out_elf)],
            env=env, capture_output=True, text=True, timeout=180,
        )
        if r.returncode != 0 or not out_elf.exists():
            raise RuntimeError(
                f"build_kernel.sh failed (rc={r.returncode}):\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )
    except Exception:
        # leave the build dir for inspection on failure
        raise
    else:
        # success path: keep the .c around for cache hits to debug; remove the
        # transient object files build_kernel.sh might leave behind.
        pass

    return out_elf


def _find_uv() -> Optional[str]:
    """Locate the `uv` binary. ssh non-interactive shells often strip the
    user-level $HOME/.local/bin from PATH, so we explicitly check there too."""
    import shutil
    if found := shutil.which("uv"):
        return found
    for candidate in (Path.home() / ".local/bin/uv",
                      Path("/usr/local/bin/uv"),
                      Path("/opt/uv/bin/uv")):
        if candidate.is_file():
            return str(candidate)
    return None


def _pyspike_argv0() -> list[str]:
    """Return the prefix command that resolves pyspike's plugin search path.

    `uv run --no-sync pyspike` is the reference invocation (matches
    test/verify_pyspike_ggml.py) — calling the bare `pyspike` binary works
    for vanilla simulation but `--extlib=riscv.gtx` can only be resolved
    when uv puts the repo on sys.path. Override with PYSPIKE_BIN if you
    have a turn-key wrapper script."""
    if env := os.environ.get("PYSPIKE_BIN"):
        return [env]
    if uv := _find_uv():
        return [uv, "run", "--no-sync", "--project", str(REPO_ROOT), "pyspike"]
    import shutil
    venv_bin = REPO_ROOT / ".venv/bin/pyspike"
    if venv_bin.is_file():
        return [str(venv_bin)]
    if found := shutil.which("pyspike"):
        return [found]
    raise FileNotFoundError(
        "pyspike not found — set PYSPIKE_BIN or install `uv` + the venv")


def _run_pyspike(elf_path: Path, sections: list[tuple[int, bytes]],
                 dump_size: int, timeout: int = 300) -> bytes:
    """Invoke `pyspike` on `elf_path`. `sections` is a list of (ddr_offset, data)
    pairs written to the same hex init file (one `@addr` block each). Returns
    the first `dump_size` bytes of the dump."""
    if dump_size <= 0:
        return b""

    pyspike_argv0 = _pyspike_argv0()

    with tempfile.TemporaryDirectory(prefix="pyspike-rpc-") as tmp:
        inp = Path(tmp) / "input.hex"
        dmp = Path(tmp) / "dump.hex"
        _write_hex_sections(inp, sections)

        # riscv.pth normally injects src/main/python into sys.path, but it's
        # only honoured when the venv installs it as a `.pth` file under
        # site-packages. Editable installs miss that, so prepend the repo
        # root explicitly here.
        repo_py = str(REPO_ROOT)
        prior_pp = os.environ.get("PYTHONPATH", "")
        new_pp = repo_py if not prior_pp else f"{repo_py}:{prior_pp}"

        env = dict(os.environ, **_PYSPIKE_ENV,
                   PYTHONPATH=new_pp,
                   GTX_DDR_INIT=str(inp), GTX_DDR_DUMP=str(dmp),
                   GTX_DDR_DUMP_ADDR=f"{DEFAULT_DUMP_ADDR:#x}",
                   GTX_DDR_DUMP_SIZE=str(dump_size))

        r = subprocess.run(
            pyspike_argv0 + ["--extlib=riscv.gtx", "--extension=gtx",
             f"--device=gtx_ddr,{DEFAULT_DEVICE_BASE:#x}", str(elf_path)],
            env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"pyspike failed (rc={r.returncode}):\n"
                f"--- stdout (last) ---\n{r.stdout[-1000:]}\n"
                f"--- stderr (last) ---\n{r.stderr[-1000:]}"
            )
        if not dmp.exists():
            raise RuntimeError("pyspike produced no dump")
        return _decode_hex_file(dmp)[:dump_size]


# pyspike's DDR hex I/O reverses each 32-byte bus word. Under
# GTX_DDR_REVERSE_MODE=elem (used here, matching verify_pyspike_ggml.py),
# that's a 16-element fp16 reversal per line. The host therefore has to
# pre-reverse fp16 elements 16-at-a-time on the way in, and undo the same
# reversal on the way out — otherwise pyspike applies its reversal on a
# mismatched line width and the firmware sees scrambled data.
_BUS_WORD_BYTES = 32
_FP16_PER_BUS_WORD = _BUS_WORD_BYTES // 2  # = 16


def _reverse_fp16_per_bus_word(data: bytes) -> bytes:
    """Reverse fp16 element order within each 32-byte bus word."""
    pad = (-len(data)) % _BUS_WORD_BYTES
    buf = data + b"\x00" * pad
    out = bytearray(len(buf))
    for i in range(0, len(buf), _BUS_WORD_BYTES):
        chunk = buf[i:i + _BUS_WORD_BYTES]
        out[i:i + _BUS_WORD_BYTES] = b"".join(
            chunk[j:j + 2] for j in range(_BUS_WORD_BYTES - 2, -2, -2))
    # caller knows how many real bytes it asked for; trim the padding.
    return bytes(out[:len(data)])


def _write_hex_sections(path: Path,
                        sections: list[tuple[int, bytes]]) -> None:
    """Write multiple (ddr_offset, data) sections to a single hex file. Each
    section gets its own `@addr` header and is pre-reversed + padded to
    bus-word boundary so pyspike's `ddr_load_from_hex` reads each 32-byte line
    and reverses back to the original element order."""
    with open(path, "w") as f:
        for ddr_offset, data in sections:
            pad = (-len(data)) % _BUS_WORD_BYTES
            padded = data + b"\x00" * pad
            pre = _reverse_fp16_per_bus_word(padded)
            f.write(f"@{ddr_offset:x}\n")
            for i in range(0, len(pre), _BUS_WORD_BYTES):
                f.write(pre[i:i + _BUS_WORD_BYTES].hex() + "\n")


def _decode_hex_file(path: Path) -> bytes:
    raw = bytearray()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            raw.extend(bytes.fromhex(line))
    return _reverse_fp16_per_bus_word(bytes(raw))


# -----------------------------------------------------------------------------
# Public dispatcher entry point
# -----------------------------------------------------------------------------

# op_name → (intrinsic call, load_bank, store_bank). The unary_intrin1.c.tpl
# skeleton runs load(L2 → load_bank) → intrinsic → store(store_bank → L2);
# each intrinsic's bank in/out convention comes from test/<OP>/n1s16/n1s16_<op>.c
# (the corpus the ISS is verified against). The two patterns are:
#   A→R : abs, neg, exp, sgn, step, relu, sqrt  (intrinsic reads BANK_A, writes BANK_R)
#   R→A : sigm, tanh, gelu                      (reads BANK_R, writes BANK_A)
_UNARY_INTRIN1_CALLS: dict[str, tuple[str, str, str]] = {
    "abs":      ("__abs_v(WIDTH);",                 "BANK_A", "BANK_R"),
    "neg":      ("__neg_v(WIDTH);",                 "BANK_A", "BANK_R"),
    "exp":      ("__exp_v(WIDTH, 0);",              "BANK_A", "BANK_R"),
    "sgn":      ("__sign_v(WIDTH);",                "BANK_A", "BANK_R"),
    "step":     ("__step_v(WIDTH);",                "BANK_A", "BANK_R"),
    "relu":     ("__clamp_min(WIDTH, 0x0000, 0);",  "BANK_A", "BANK_R"),
    "sqrt":     ("__sqrt_v(WIDTH);",                "BANK_A", "BANK_R"),
    "floor":    ("__floor_v(WIDTH);",               "BANK_A", "BANK_R"),
    "trunc":    ("__trunc_v(WIDTH);",               "BANK_A", "BANK_R"),
    "sigmoid":  ("__sigm(WIDTH);",                  "BANK_R", "BANK_A"),
    "tanh":     ("__tanh(WIDTH);",                  "BANK_R", "BANK_A"),
    "gelu":     ("__gelu(WIDTH);",                  "BANK_R", "BANK_A"),
    "gelu_erf": ("__gelu(WIDTH);",                  "BANK_R", "BANK_A"),
    # Level-3 __silu(vec_size, A, B, R): per intrin_level3.c the intrinsic
    # reads from BANK_R and writes to BANK_A (it runs sigm into A then mul.vv
    # with B holding the original x).
    "silu":     ("__silu(WIDTH, BANK_A, BANK_B, BANK_R);", "BANK_R", "BANK_A"),
}


def _run_unary_fp16(op_name: str, input_bytes: bytes) -> bytes:
    """Render unary_intrin1.c.tpl for `op_name`, build (cached), run on pyspike.

    `input_bytes` length must be a multiple of ROW_BYTES (16 = 8 fp16 elems).
    Returns same-length bytes of op(input) in fp16.
    """
    if op_name not in _UNARY_INTRIN1_CALLS:
        raise KeyError(f"no unary intrinsic registered for op '{op_name}'")
    if len(input_bytes) == 0 or len(input_bytes) % ROW_BYTES != 0:
        raise ValueError(
            f"input must be a non-empty multiple of {ROW_BYTES} bytes, "
            f"got {len(input_bytes)}")
    height = len(input_bytes) // ROW_BYTES
    call, load_bank, store_bank = _UNARY_INTRIN1_CALLS[op_name]
    src = _render_template("unary_intrin1",
                           HEIGHT=height,
                           OP_NAME=f"unary_{op_name}",
                           INTRIN_CALL=call,
                           LOAD_BANK=load_bank,
                           STORE_BANK=store_bank)
    elf = _build_kernel(src, _cache_key(f"unary_{op_name}", "f16", (height,)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_unary_abs_fp16(input_bytes: bytes) -> bytes:
    return _run_unary_fp16("abs", input_bytes)


def run_unary_silu_fp16(input_bytes: bytes) -> bytes:
    return _run_unary_fp16("silu", input_bytes)


# ---- binary (vector + vector) elementwise ops ----
# Same bank pattern across all four (A,B → R) — matches test/ADD/SUB/MUL/DIV.
_BINARY_INTRIN1_CALLS: dict[str, str] = {
    "add": "__add_vv(WIDTH);",
    "sub": "__sub_vv(WIDTH);",
    "mul": "__mul_vv(WIDTH);",
    "div": "__div_vv(WIDTH);",
    # ACC is functionally add (test/ACC kernel uses __add_vv) — ggml ACC is
    # accumulate-in-place, but at the kernel level it's still an elementwise
    # add of two contiguous tensors.
    "acc": "__add_vv(WIDTH);",
}

# binary_intrin1.c.tpl reads src0 at BASE_DDR_A=0x1000000, src1 at 0x2000000.
DEFAULT_INPUT_B_OFFSET = 0x2000000


def _run_binary_fp16(op_name: str, src0: bytes, src1: bytes) -> bytes:
    """Render binary_intrin1.c.tpl for `op_name`, build (cached), run on pyspike
    with two DDR sections (src0 → 0x1000000, src1 → 0x2000000).
    """
    if op_name not in _BINARY_INTRIN1_CALLS:
        raise KeyError(f"no binary intrinsic registered for op '{op_name}'")
    if len(src0) != len(src1):
        raise ValueError(f"binary inputs must be equal length ({len(src0)} vs {len(src1)})")
    if len(src0) == 0 or len(src0) % ROW_BYTES != 0:
        raise ValueError(
            f"input must be a non-empty multiple of {ROW_BYTES} bytes, "
            f"got {len(src0)}")
    height = len(src0) // ROW_BYTES
    src = _render_template("binary_intrin1",
                           HEIGHT=height,
                           OP_NAME=f"binary_{op_name}",
                           INTRIN_CALL=_BINARY_INTRIN1_CALLS[op_name])
    elf = _build_kernel(src, _cache_key(f"binary_{op_name}", "f16", (height,)))
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_OFFSET, src0),
                         (DEFAULT_INPUT_B_OFFSET, src1)],
                        dump_size=len(src0))


# ---- MUL_MAT (matrix multiply) ----
# Inherits the exact kernel from test/MUL_MAT/n1s16/n1s16_mul_mat.c with M,K,N
# parameterised. ggml mul_mat semantics: src0 has ggml shape (K,M) — reversed
# to numpy (M,K); src1 has ggml shape (K,N) → numpy (N,K); dst is (M,N) on the
# numpy side, written contiguously row-major.
def _run_mul_mat_fp16(src0_a: bytes, src1_b: bytes,
                      M: int, K: int, N: int) -> bytes:
    """A:(M,K) src0 → 0x1000000, B:(N,K) src1 → 0x2000000.
    Result R:(M,N) is dumped from 0xf000000 (mapped via DEFAULT_DUMP_ADDR).
    Single-shot (no tiling). Use `run_mul_mat_tiled_fp16` for shapes that
    don't fit in one firmware call.
    """
    expected_a = M * K * DTYPE_BYTES
    expected_b = N * K * DTYPE_BYTES
    if len(src0_a) != expected_a:
        raise ValueError(f"src0 size mismatch: got {len(src0_a)}, expected {expected_a}")
    if len(src1_b) != expected_b:
        raise ValueError(f"src1 size mismatch: got {len(src1_b)}, expected {expected_b}")
    result_bytes = M * N * DTYPE_BYTES
    src = _render_template("mul_mat", M=M, K=K, N=N)
    elf = _build_kernel(src, _cache_key("mul_mat", "f16", (M, K, N)))
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_OFFSET, src0_a),
                         (DEFAULT_INPUT_B_OFFSET, src1_b)],
                        dump_size=result_bytes,
                        timeout=900)


# Per-call tile bounds for the host-side MUL_MAT splitter. Picked to fit the
# reference n1s16_mul_mat.c proven shape (M≤1024, N≤512) — anything larger
# is broken into multiple firmware calls. K is left to the firmware's own
# inner tiling (the kernel handles K independent of host).
MUL_MAT_TILE_M = 1024
MUL_MAT_TILE_N = 512


def run_mul_mat_tiled_fp16(src0_a: bytes, src1_b: bytes,
                           M: int, K: int, N: int,
                           tile_m: int = MUL_MAT_TILE_M,
                           tile_n: int = MUL_MAT_TILE_N) -> bytes:
    """Host-side M·N tile loop around `_run_mul_mat_fp16`. The firmware kernel
    itself handles K via its inner tiling, so no partial-sum accumulation is
    needed across calls — each (m_tile, n_tile) call returns its slice of the
    full (M,N) result and we splice it into the right rows/cols.

    A is laid out (M,K) row-major, B is (N,K) row-major (the kernel does
    R = A @ B^T per ggml mul_mat semantics).
    """
    if len(src0_a) != M * K * DTYPE_BYTES:
        raise ValueError(f"src0 size mismatch: got {len(src0_a)}, expected {M*K*DTYPE_BYTES}")
    if len(src1_b) != N * K * DTYPE_BYTES:
        raise ValueError(f"src1 size mismatch: got {len(src1_b)}, expected {N*K*DTYPE_BYTES}")
    if K % WIDTH != 0:
        raise ValueError(f"K={K} must be a multiple of WIDTH={WIDTH}")
    # tile widths must keep WIDTH alignment for the firmware's row layout.
    if tile_m % 1 != 0 or tile_n % 1 != 0:  # M/N are row counts, not byte counts
        raise ValueError("tile_m / tile_n must be positive integers")

    # Fast path: whole problem fits in a single call.
    if M <= tile_m and N <= tile_n:
        return _run_mul_mat_fp16(src0_a, src1_b, M, K, N)

    result = bytearray(M * N * DTYPE_BYTES)
    K_bytes = K * DTYPE_BYTES

    for m_off in range(0, M, tile_m):
        m_size = min(tile_m, M - m_off)
        # A_tile = A[m_off : m_off+m_size, :] — m_size rows of K fp16 each.
        a_byte_start = m_off * K_bytes
        a_byte_end = (m_off + m_size) * K_bytes
        a_tile = src0_a[a_byte_start:a_byte_end]

        for n_off in range(0, N, tile_n):
            n_size = min(tile_n, N - n_off)
            # B_tile = B[n_off : n_off+n_size, :] — n_size rows of K fp16 each.
            b_byte_start = n_off * K_bytes
            b_byte_end = (n_off + n_size) * K_bytes
            b_tile = src1_b[b_byte_start:b_byte_end]

            r_tile = _run_mul_mat_fp16(a_tile, b_tile, m_size, K, n_size)
            # r_tile is (m_size, n_size) row-major fp16; splice into result.
            for r in range(m_size):
                src_off = r * n_size * DTYPE_BYTES
                dst_off = ((m_off + r) * N + n_off) * DTYPE_BYTES
                result[dst_off:dst_off + n_size * DTYPE_BYTES] = \
                    r_tile[src_off:src_off + n_size * DTYPE_BYTES]
    return bytes(result)


# -----------------------------------------------------------------------------
# Shape-parameterised "simple unary" ops (SQR, SUM_ROWS, GROUP_NORM, NORM,
# SCALE). Each carries its own template; the runner picks WIDTH/HEIGHT from
# the input tensor's ne (innermost dim → WIDTH, remaining elements → HEIGHT)
# and supplies any op-specific scalar (eps, scale) via placeholder
# substitution or a second DDR section.
# -----------------------------------------------------------------------------


def _f32_to_fp16_bits(x: float) -> int:
    """Convert a float32 to its IEEE-754 binary16 bit pattern (uint16)."""
    import numpy as np
    return int(np.float16(x).view(np.uint16))


def _shape_for_simple_unary(input_bytes: bytes, width: int) -> tuple[int, int]:
    """Pick (WIDTH, HEIGHT) for a kernel that wants a row-major (W,H) split.
    The caller fixes WIDTH (per-row element count); we derive HEIGHT from the
    total fp16 element count. Raises ValueError if the input isn't aligned.
    """
    if len(input_bytes) == 0 or len(input_bytes) % DTYPE_BYTES != 0:
        raise ValueError(f"input must be a non-empty multiple of {DTYPE_BYTES}")
    elems = len(input_bytes) // DTYPE_BYTES
    if width <= 0 or elems % width != 0:
        raise ValueError(f"element count {elems} not divisible by WIDTH={width}")
    return width, elems // width


def run_sqr_fp16(input_bytes: bytes, width: int) -> bytes:
    """SQR: dst[i] = x[i] * x[i]. WIDTH is the per-row element count; HEIGHT
    derives from input size. Element split across SPUs."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_sqr",
                           OP_NAME="unary_sqr",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_sqr", "f16", (w, h)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_sum_rows_fp16(input_bytes: bytes, width: int) -> bytes:
    """SUM_ROWS: dst[row] = sum(src[row, :WIDTH]). Returns HEIGHT fp16 outputs."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_sum_rows",
                           OP_NAME="unary_sum_rows",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_sum_rows", "f16", (w, h)))
    out_bytes = h * DTYPE_BYTES
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=out_bytes)


def run_group_norm_fp16(input_bytes: bytes, width: int, eps: float) -> bytes:
    """GROUP_NORM with num_groups=1 (whole tensor normalised together).
    eps is the ggml op_params[1] float32; we quantise to fp16 for the kernel's
    `__layernorm` epsilon argument.
    """
    w, h = _shape_for_simple_unary(input_bytes, width)
    eps_bits = _f32_to_fp16_bits(eps)
    src = _render_template("unary_group_norm",
                           OP_NAME="unary_group_norm",
                           WIDTH=w, HEIGHT=h,
                           EPS_FP16=f"0x{eps_bits:04X}")
    elf = _build_kernel(src, _cache_key("unary_group_norm", "f16",
                                        (w, h, eps_bits)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_norm_fp16(input_bytes: bytes, width: int, eps: float) -> bytes:
    """NORM (per-row layer normalisation). HEIGHT rows × WIDTH cols."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    eps_bits = _f32_to_fp16_bits(eps)
    src = _render_template("unary_norm",
                           OP_NAME="unary_norm",
                           WIDTH=w, HEIGHT=h,
                           EPS_FP16=f"0x{eps_bits:04X}")
    elf = _build_kernel(src, _cache_key("unary_norm", "f16",
                                        (w, h, eps_bits)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_scale_fp16(input_bytes: bytes, width: int, scale: float) -> bytes:
    """SCALE: dst = src * scale. Scale arrives in DDR at BASE_DDR_B as a single
    fp16 element; firmware loads it via CPU read before __split.
    Caller routes only when ggml op_params[1] (bias) == 0; the host applies
    bias separately when it's non-zero.
    """
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_scale",
                           OP_NAME="unary_scale",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_scale", "f16", (w, h)))
    # Scale factor: one fp16 placed at BASE_DDR_B. The hex writer pads each
    # section to a 32-byte bus word and reverses element order — the 16-elem
    # reverse undoes itself when pyspike reloads it (verified by ABS/SILU).
    import numpy as np
    scale_bytes = np.float16(scale).tobytes()
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_OFFSET, input_bytes),
                         (DEFAULT_INPUT_B_OFFSET, scale_bytes)],
                        dump_size=len(input_bytes))


def run_ceil_fp16(input_bytes: bytes, width: int) -> bytes:
    """CEIL: dst = -floor(-x). Per-row tile, arbitrary (W, H)."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_ceil",
                           OP_NAME="unary_ceil",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_ceil", "f16", (w, h)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_expm1_fp16(input_bytes: bytes, width: int) -> bytes:
    """EXPM1: dst = exp(x) - 1. Per-row, requires HEIGHT % 16 == 0
    (firmware uses ROWS_PER_SPU = HEIGHT / 16 with no remainder handling)."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    if h % 16 != 0:
        raise ValueError(f"expm1 requires HEIGHT % 16 == 0, got HEIGHT={h}")
    src = _render_template("unary_expm1",
                           OP_NAME="unary_expm1",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_expm1", "f16", (w, h)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_clamp_fp16(input_bytes: bytes, width: int,
                   min_val: float, max_val: float) -> bytes:
    """CLAMP: dst = clamp(x, min, max). min/max arrive as 2 fp16 at DDR_B."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_clamp",
                           OP_NAME="unary_clamp",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_clamp", "f16", (w, h)))
    import numpy as np
    mm_bytes = np.array([min_val, max_val], dtype=np.float16).tobytes()
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_OFFSET, input_bytes),
                         (DEFAULT_INPUT_B_OFFSET, mm_bytes)],
                        dump_size=len(input_bytes))


def run_mean_fp16(input_bytes: bytes, width: int) -> bytes:
    """MEAN (per-row): dst[row] = sum(row) / WIDTH. The 1/WIDTH fp16 constant
    is computed by the host and baked into the kernel. Requires HEIGHT % 16 == 0
    (same row-balancing constraint as EXPM1)."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    if h % 16 != 0:
        raise ValueError(f"mean requires HEIGHT % 16 == 0, got HEIGHT={h}")
    inv_w_bits = _f32_to_fp16_bits(1.0 / w)
    src = _render_template("unary_mean",
                           OP_NAME="unary_mean",
                           WIDTH=w, HEIGHT=h,
                           INV_W_FP16=f"0x{inv_w_bits:04X}")
    elf = _build_kernel(src, _cache_key("unary_mean", "f16",
                                        (w, h, inv_w_bits)))
    out_bytes = h * DTYPE_BYTES
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=out_bytes)


def run_arange_fp16(n: int) -> bytes:
    """ARANGE: dst[i] = i for i in [0, N). Firmware hard-codes start=0 and
    step=1 — caller must pre-check ggml op_params. COLS=8 row tiling, so
    N must be a positive multiple of 8.
    """
    if n <= 0 or n % 8 != 0:
        raise ValueError(f"arange requires N > 0 and N % 8 == 0, got N={n}")
    cols = 8
    rows = n // cols
    src = _render_template("unary_arange",
                           OP_NAME="unary_arange",
                           ROWS=rows, COLS=cols)
    elf = _build_kernel(src, _cache_key("unary_arange", "f16", (n,)))
    # No DDR input section — pyspike starts with a zero-initialised DDR.
    return _run_pyspike(elf, [], dump_size=n * DTYPE_BYTES)


def run_repeat_fp16(input_bytes: bytes,
                    src_ne: tuple,
                    dst_ne: tuple) -> bytes:
    """REPEAT: dst[idx] = src[idx % src.ne] across 4 dims. Each DST_NEx must
    be an integer multiple of SRC_NEx. Both src and dst must individually fit
    in their 512KB L2 pool; larger shapes need host-side tiling.
    """
    if len(src_ne) != 4 or len(dst_ne) != 4:
        raise ValueError("src_ne and dst_ne must be 4-tuples")
    s0, s1, s2, s3 = src_ne
    d0, d1, d2, d3 = dst_ne
    if any(s <= 0 for s in (s0, s1, s2, s3, d0, d1, d2, d3)):
        raise ValueError(f"all ne dims must be positive: src={src_ne} dst={dst_ne}")
    if any(d % s != 0 for s, d in zip(src_ne, dst_ne)):
        raise ValueError(f"each DST_NE must be a multiple of SRC_NE: "
                         f"src={src_ne} dst={dst_ne}")
    src_total = s0 * s1 * s2 * s3 * DTYPE_BYTES
    dst_total = d0 * d1 * d2 * d3 * DTYPE_BYTES
    if src_total != len(input_bytes):
        raise ValueError(f"src ne {src_ne} disagrees with input length {len(input_bytes)}")
    if src_total > 0x80000 or dst_total > 0x80000:
        raise ValueError(f"repeat src/dst exceed 512KB L2 budget "
                         f"(src={src_total}, dst={dst_total})")
    src_c = _render_template("unary_repeat",
                             OP_NAME="unary_repeat",
                             SRC_NE0=s0, SRC_NE1=s1, SRC_NE2=s2, SRC_NE3=s3,
                             DST_NE0=d0, DST_NE1=d1, DST_NE2=d2, DST_NE3=d3)
    elf = _build_kernel(src_c, _cache_key("unary_repeat", "f16",
                                          (s0, s1, s2, s3, d0, d1, d2, d3)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=dst_total)


def run_im2col_fp16(input_bytes: bytes,
                    in_h: int, in_w: int,
                    k_h: int, k_w: int,
                    stride: int) -> bytes:
    """IM2COL 2D: rearrange input into per-patch rows for matmul-based conv.
    Vendor kernel assumes single channel (IC=1), zero padding, dilation=1,
    and equal s0=s1=stride. Caller must pre-check the ggml op_params.

    Output layout: (OH*OW) patches × (KH*KW) fp16 elements each.
    The input is placed at BASE_DDR_INPUT=0x2000000 (not the usual 0x1000000)
    because the vendor IM2COL kernel reads input from DDR_B.
    """
    expected = in_h * in_w * DTYPE_BYTES
    if len(input_bytes) != expected:
        raise ValueError(
            f"im2col input bytes {len(input_bytes)} != {in_h}*{in_w}*{DTYPE_BYTES}")
    if k_h <= 0 or k_w <= 0 or stride <= 0:
        raise ValueError("im2col kernel/stride must be positive")
    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"im2col output non-positive: out_h={out_h} out_w={out_w}")
    src_c = _render_template("unary_im2col",
                             OP_NAME="unary_im2col",
                             IN_H=in_h, IN_W=in_w,
                             K_H=k_h, K_W=k_w, STRIDE=stride)
    elf = _build_kernel(src_c, _cache_key("unary_im2col", "f16",
                                          (in_h, in_w, k_h, k_w, stride)))
    dst_bytes = out_h * out_w * k_h * k_w * DTYPE_BYTES
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_B_OFFSET, input_bytes)],
                        dump_size=dst_bytes)


def run_pad_fp16(input_bytes: bytes,
                 src_rows: int, src_cols: int,
                 pad_right: int, pad_bottom: int) -> bytes:
    """PAD: append `pad_right` zero columns and `pad_bottom` zero rows.
    Only the right/bottom layout the vendor `__pad` kernel implements; ggml
    op_params carry per-dim pad widths and the server pre-checks left/top=0.
    """
    expected = src_rows * src_cols * DTYPE_BYTES
    if len(input_bytes) != expected:
        raise ValueError(
            f"pad src bytes {len(input_bytes)} != {src_rows}*{src_cols}*{DTYPE_BYTES}")
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError("pad amounts must be non-negative")
    src = _render_template("unary_pad",
                           OP_NAME="unary_pad",
                           SRC_ROWS=src_rows, SRC_COLS=src_cols,
                           PAD_RIGHT=pad_right, PAD_BOTTOM=pad_bottom)
    elf = _build_kernel(src, _cache_key("unary_pad", "f16",
                                        (src_rows, src_cols, pad_right, pad_bottom)))
    dst_rows = src_rows + pad_bottom
    dst_cols = src_cols + pad_right
    dst_bytes = dst_rows * dst_cols * DTYPE_BYTES
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=dst_bytes)


def run_concat_fp16(src0: bytes, src1: bytes,
                    src0_cols: int, src1_cols: int,
                    rows: int) -> bytes:
    """CONCAT axis=0: dst row = [src0 row | src1 row]. Both srcs must have
    the same number of rows; the vendor kernel currently assumes equal col
    counts (it copies `SRC_COLS` from each side), so the runner enforces it.
    """
    if src0_cols != src1_cols:
        raise ValueError(
            f"concat kernel only supports equal col counts: "
            f"src0={src0_cols}, src1={src1_cols}")
    expected = src0_cols * rows * DTYPE_BYTES
    if len(src0) != expected or len(src1) != expected:
        raise ValueError(
            f"concat input size mismatch: src0={len(src0)} src1={len(src1)} "
            f"expected={expected}")
    src_c = _render_template("unary_concat",
                             OP_NAME="unary_concat",
                             SRC_COLS=src0_cols, ROWS=rows)
    elf = _build_kernel(src_c, _cache_key("unary_concat", "f16",
                                          (src0_cols, rows)))
    dst_bytes = 2 * src0_cols * rows * DTYPE_BYTES
    return _run_pyspike(elf,
                        [(DEFAULT_INPUT_OFFSET, src0),
                         (DEFAULT_INPUT_B_OFFSET, src1)],
                        dump_size=dst_bytes)


def run_pool_2d_avg_fp16(input_bytes: bytes,
                         in_h: int, in_w: int,
                         k_h: int, k_w: int,
                         s_h: int, s_w: int) -> bytes:
    """POOL_2D average. Output size derived from input + kernel + stride;
    padding 0 (the vendor `__pool_a` kernel assumes no padding). Reciprocal
    1/(K_H*K_W) is injected into the kernel as an fp16 constant.
    """
    expected = in_h * in_w * DTYPE_BYTES
    if len(input_bytes) != expected:
        raise ValueError(
            f"pool input bytes {len(input_bytes)} != {in_h}*{in_w}*{DTYPE_BYTES}")
    if k_h <= 0 or k_w <= 0 or s_h <= 0 or s_w <= 0:
        raise ValueError("pool kernel/stride must be positive")
    out_h = (in_h - k_h) // s_h + 1
    out_w = (in_w - k_w) // s_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"pool output size non-positive: out_h={out_h} out_w={out_w}")
    inv_k_bits = _f32_to_fp16_bits(1.0 / (k_h * k_w))
    src_c = _render_template("unary_pool_2d_avg",
                             OP_NAME="unary_pool_2d_avg",
                             IN_H=in_h, IN_W=in_w,
                             OUT_H=out_h, OUT_W=out_w,
                             K_H=k_h, K_W=k_w, S_H=s_h, S_W=s_w,
                             INV_K_FP16=f"0x{inv_k_bits:04X}")
    elf = _build_kernel(src_c, _cache_key("unary_pool_2d_avg", "f16",
                                          (in_h, in_w, k_h, k_w, s_h, s_w)))
    dst_bytes = out_h * out_w * DTYPE_BYTES
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=dst_bytes)


def run_tri_fp16(input_bytes: bytes, width: int, tri_type: int) -> bytes:
    """TRI: copy src into dst then zero one triangle. tri_type encodes which
    triangle is kept vs zeroed (0=upper_diag, 1=upper, 2=lower_diag, 3=lower
    in the vendor kernel). The kernel processes min(W, H) rows/cols, so
    rectangular inputs work by truncation.
    """
    w, h = _shape_for_simple_unary(input_bytes, width)
    if tri_type < 0 or tri_type > 3:
        raise ValueError(f"tri_type must be in [0, 3], got {tri_type}")
    src = _render_template("unary_tri",
                           OP_NAME="unary_tri",
                           WIDTH=w, HEIGHT=h,
                           TRI_TYPE=tri_type)
    elf = _build_kernel(src, _cache_key("unary_tri", "f16",
                                        (w, h, tri_type)))
    return _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                        dump_size=len(input_bytes))


def run_sum_fp16(input_bytes: bytes, width: int) -> bytes:
    """SUM (total reduction): dst[0] = sum of all input elements.
    Firmware dumps a 32-byte block (1 fp16 sum + 15 padding); we trim
    to the single fp16 scalar the caller expects."""
    w, h = _shape_for_simple_unary(input_bytes, width)
    src = _render_template("unary_sum",
                           OP_NAME="unary_sum",
                           WIDTH=w, HEIGHT=h)
    elf = _build_kernel(src, _cache_key("unary_sum", "f16", (w, h)))
    raw = _run_pyspike(elf, [(DEFAULT_INPUT_OFFSET, input_bytes)],
                       dump_size=32)
    return raw[:DTYPE_BYTES]


SUPPORTED_PYSPIKE_OPS = {
    **{
        f"unary_{name}_fp16":
            (lambda op: lambda data: _run_unary_fp16(op, data))(name)
        for name in _UNARY_INTRIN1_CALLS
    },
    **{
        f"binary_{name}_fp16":
            (lambda op: lambda a, b: _run_binary_fp16(op, a, b))(name)
        for name in _BINARY_INTRIN1_CALLS
    },
    "mul_mat_fp16":       _run_mul_mat_fp16,
    "mul_mat_tiled_fp16": run_mul_mat_tiled_fp16,
    "sqr_fp16":           run_sqr_fp16,
    "sum_rows_fp16":      run_sum_rows_fp16,
    "group_norm_fp16":    run_group_norm_fp16,
    "norm_fp16":          run_norm_fp16,
    "scale_fp16":         run_scale_fp16,
    "ceil_fp16":          run_ceil_fp16,
    "expm1_fp16":         run_expm1_fp16,
    "clamp_fp16":         run_clamp_fp16,
    "mean_fp16":          run_mean_fp16,
    "sum_fp16":           run_sum_fp16,
    "arange_fp16":        run_arange_fp16,
    "tri_fp16":           run_tri_fp16,
    "repeat_fp16":        run_repeat_fp16,
    "pad_fp16":           run_pad_fp16,
    "concat_fp16":        run_concat_fp16,
    "pool_2d_avg_fp16":   run_pool_2d_avg_fp16,
    "im2col_fp16":        run_im2col_fp16,
}
