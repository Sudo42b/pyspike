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


def _build_assets_dir() -> Path:
    """Where spike_crt.S and spike_gtx_link.ld live (same dir as build_kernel.sh)."""
    d = REPO_ROOT / "src/test/gtx"
    if not (d / "spike_crt.S").is_file() or not (d / "spike_gtx_link.ld").is_file():
        raise FileNotFoundError(f"missing spike_crt.S/spike_gtx_link.ld under {d}")
    return d


# GCC 15 (gtx-firmware's own Makefile) uses plain rv64g — the legacy build
# script's `xgtxnpu` extension is rejected as a non-standard `x*` ext.
_CFLAGS = (
    "-march=rv64g -mabi=lp64d -mcmodel=large -O3 -g -ffreestanding -nostartfiles "
    "-ffunction-sections -fdata-sections -std=c11 "
    "-DGTX_MAIN_OFFSET=0x370000000ULL "
    "-Wno-unused-parameter -Wno-unused-variable -Wno-unused-function "
    "-Wno-missing-field-initializers -Wno-strict-aliasing "
    "-Wno-incompatible-pointer-types -Wno-compare-distinct-pointer-types"
).split()


def _resolve_firmware_root() -> str:
    """Best-effort GTX firmware location. Caller can override via GTX_FIRMWARE."""
    if env := os.environ.get("GTX_FIRMWARE"):
        return env
    for c in ("/home/owner/gtx-firmware",
              "/home/sw.lee/supergate_sw/device/gtx-firmware"):
        if Path(c).is_dir():
            return c
    raise FileNotFoundError("set GTX_FIRMWARE — couldn't auto-detect")


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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_elf = CACHE_DIR / f"{cache_key}.elf"
    if out_elf.exists():
        return out_elf

    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False,
                                     dir=str(CACHE_DIR)) as f:
        f.write(c_src_text)
        src_path = Path(f.name)

    try:
        gfw = Path(_resolve_firmware_root())
        intrin = gfw / "src/gtx/intrinsics"
        assets = _build_assets_dir()
        incs = [
            "-I" + str(src_path.parent),
            "-I" + str(gfw / "include/gtx/intrinsics"),
            "-I" + str(gfw / "include"),
            "-I" + str(gfw / "include/gtx"),
        ]
        srcs = [
            str(src_path),
            str(intrin / "intrin_level1.c"),
            str(intrin / "intrin_level2.c"),
            str(intrin / "intrin_level3.c"),
            str(assets / "spike_crt.S"),
        ]
        cc = os.environ.get("CROSS_CC", "riscv64-unknown-elf-gcc")
        # `-lgcc` is left out: newer riscv64-unknown-elf-gcc (gcc-15) ships
        # libgcc as a builtin search path the linker can't always resolve via
        # `-lgcc`. The firmware doesn't currently call any libgcc builtins
        # (int divides, etc.) so the bare-link works.
        cmd = (
            [cc] + _CFLAGS + incs
            + ["-T", str(assets / "spike_gtx_link.ld"),
               "-nostdlib", "-Wl,--gc-sections"]
            + srcs + ["-o", str(out_elf)]
        )
        env = dict(os.environ,
                   PATH=f"/opt/riscv/bin:{os.environ.get('PATH', '')}")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(
                f"riscv-gcc failed (rc={r.returncode}):\n"
                f"--- stderr ---\n{r.stderr}"
            )
        if not out_elf.exists():
            raise RuntimeError(f"compile produced no elf at {out_elf}")
    finally:
        try:
            src_path.unlink()
        except OSError:
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


def _run_pyspike(elf_path: Path, input_bytes: bytes,
                 input_offset: int, dump_size: int,
                 timeout: int = 300) -> bytes:
    """Invoke `pyspike` on `elf_path`. Returns first `dump_size` bytes of dump."""
    if dump_size <= 0:
        return b""

    pyspike_argv0 = _pyspike_argv0()

    with tempfile.TemporaryDirectory(prefix="pyspike-rpc-") as tmp:
        inp = Path(tmp) / "input.hex"
        dmp = Path(tmp) / "dump.hex"
        _write_hex_section(inp, input_offset, input_bytes)

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


def _write_hex_section(path: Path, ddr_offset: int, data: bytes) -> None:
    with open(path, "w") as f:
        f.write(f"@{ddr_offset:x}\n")
        for i in range(0, len(data), 16):
            f.write(data[i:i + 16].hex() + "\n")


def _decode_hex_file(path: Path) -> bytes:
    out = bytearray()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            out.extend(bytes.fromhex(line))
    return bytes(out)


# -----------------------------------------------------------------------------
# Public dispatcher entry point
# -----------------------------------------------------------------------------

def _run_unary_fp16(op_name: str, input_bytes: bytes) -> bytes:
    """Shared elementwise-unary runner: render `<op_name>.c.tpl` with the
    HEIGHT inferred from the input length, build it (cached), and run it.

    `input_bytes` length must be a multiple of ROW_BYTES (16 = 8 fp16 elems).
    Returns same-length bytes of op(input) in fp16.
    """
    if len(input_bytes) == 0 or len(input_bytes) % ROW_BYTES != 0:
        raise ValueError(
            f"input must be a non-empty multiple of {ROW_BYTES} bytes, "
            f"got {len(input_bytes)}")
    height = len(input_bytes) // ROW_BYTES
    src = _render_template(op_name, HEIGHT=height)
    elf = _build_kernel(src, _cache_key(op_name, "f16", (height,)))
    return _run_pyspike(elf, input_bytes,
                        input_offset=DEFAULT_INPUT_OFFSET,
                        dump_size=len(input_bytes))


def run_unary_abs_fp16(input_bytes: bytes) -> bytes:
    """SiLU's lazy sibling — element-wise |x| in fp16."""
    return _run_unary_fp16("unary_abs", input_bytes)


def run_unary_silu_fp16(input_bytes: bytes) -> bytes:
    """y = x * sigmoid(x), element-wise, fp16. See firmware_templates/
    unary_silu.c.tpl — Level-3 GTX intrinsic `__silu` does the work."""
    return _run_unary_fp16("unary_silu", input_bytes)


SUPPORTED_PYSPIKE_OPS = {
    "unary_abs_fp16":  run_unary_abs_fp16,
    "unary_silu_fp16": run_unary_silu_fp16,
}
