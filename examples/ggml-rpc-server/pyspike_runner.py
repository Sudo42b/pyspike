"""pyspike .elf execution adapter.

M2 (current): infrastructure only — `run_kernel()` is wired up but the n1s16
firmware corpus has hardcoded shapes (e.g. HEIGHT=393217 in n1s16_abs.c) that
don't match arbitrary client tensors. Callers should only invoke this when
they've verified the shape constraint matches the kernel's compile-time layout.

M2.5 (future): a per-(op, shape) firmware .c template + build_kernel.sh +
.elf cache that lets us actually route arbitrary tensors to pyspike.

The wire format pyspike expects (mirrors test/verify_pyspike_ggml.py):
    Verilog $readmemh hex with @<offset> section headers.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

# Env defaults match verify_pyspike_ggml.run_op() exactly.
_PYSPIKE_ENV = dict(
    GTX_MX_IO_DTYPE="float16",
    GTX_DDR_REVERSE_MODE="elem",
    GTX_DDR_SIZE="2G",
    GTX_DDR_REVERSED="1",
    UV_LINK_MODE="copy",
)

# Firmware convention: all n1s16 kernels write their result at this DDR offset.
DEFAULT_DUMP_ADDR = 0x37f000000
DEFAULT_DEVICE_BASE = 0x370000000


@dataclass(frozen=True)
class KernelLayout:
    """Compile-time geometry of a pyspike .elf.

    For the n1s16 corpus today these are all the same: NEST=1, SPU=16, WIDTH=8
    FP16 elements, HEIGHT baked into the firmware (~393k for ABS).
    """
    nest: int = 1
    spu_per_nest: int = 16
    width_elems: int = 8
    height_rows: int = 393217
    input_base_ddr: int = 0x371000000   # = device_base + 0x1000000
    output_base_ddr: int = DEFAULT_DUMP_ADDR


def write_hex_section(path: str, ddr_offset: int, data: bytes) -> None:
    """Emit a Verilog $readmemh file: '@<offset_hex>' + 32-char hex lines."""
    with open(path, "w") as f:
        f.write(f"@{ddr_offset:x}\n")
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            f.write(chunk.hex() + "\n")


def run_kernel(elf_path: str, input_bytes: bytes,
               input_offset: int = 0x1000000,
               output_bytes: int = 0,
               timeout: int = 240) -> bytes:
    """Run pyspike on `elf_path` with `input_bytes` placed at `input_offset`
    within the gtx_ddr device. Returns `output_bytes` of dump from
    DEFAULT_DUMP_ADDR. Raises CalledProcessError on non-zero exit.
    """
    if not os.path.exists(elf_path):
        raise FileNotFoundError(elf_path)

    with tempfile.TemporaryDirectory(prefix="pyspike-rpc-") as tmp:
        inp = os.path.join(tmp, "input.hex")
        dmp = os.path.join(tmp, "dump.hex")
        write_hex_section(inp, input_offset, input_bytes)

        env = dict(os.environ, **_PYSPIKE_ENV,
                   GTX_DDR_INIT=inp, GTX_DDR_DUMP=dmp,
                   GTX_DDR_DUMP_ADDR=f"{DEFAULT_DUMP_ADDR:#x}",
                   GTX_DDR_DUMP_SIZE=str(output_bytes))

        cmd = ["pyspike", "--extlib=riscv.gtx", "--extension=gtx",
               f"--device=gtx_ddr,{DEFAULT_DEVICE_BASE:#x}", elf_path]
        subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=timeout)

        if not os.path.exists(dmp):
            raise RuntimeError(f"pyspike produced no dump at {dmp}")
        return _decode_hex_file(dmp)[:output_bytes]


def _decode_hex_file(path: str) -> bytes:
    """Decode a Verilog $readmemh file back into raw bytes (strip @ headers)."""
    out = bytearray()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            out.extend(bytes.fromhex(line))
    return bytes(out)
