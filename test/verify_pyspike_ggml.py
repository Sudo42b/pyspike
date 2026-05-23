#!/usr/bin/env python3
"""verify_pyspike_ggml.py — run ggml_ops_c kernels under pyspike and compare the
DDR dump against the numpy golden (ground truth) and the ISS ref.

For each op: build a pyspike-runnable elf (build_kernel.sh), run it on the op's
ggml_ops_c input with the ISS-compatible boundary config (FP16 I/O + element bus
reversal), dump the result region, then score it verify.py-style (sign-mag ULP<=1
or |abs|<=0.001; NaN/Inf on either side is a miss) against numpy and the ISS ref.

Env:
  GTX_GGML_BASE  ggml_ops_c corpus root (default below).

Usage:
  uv run --no-sync python3 test/verify_pyspike_ggml.py            # all numpy-golden ops
  uv run --no-sync python3 test/verify_pyspike_ggml.py abs neg    # selected ops
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GG = os.environ.get("GTX_GGML_BASE",
                    "/home/sw.lee/supergate_sw/device/gtx_kernel/ggml_ops_c")
BUILD = os.path.join(REPO, "src", "test", "gtx", "build_kernel.sh")
WORK = "/tmp/gtxrun"
sys.path.insert(0, os.path.join(REPO, "vendor", "gtx_cpp_reference", "test"))
from compare_all_ops import OP_CONFIG  # noqa: E402

os.makedirs(WORK, exist_ok=True)
_WORD = 16


def decode(path):
    """File → logical fp16 (LE per-element + 16-element bus-word reversal)."""
    raw = b"".join(bytes.fromhex(l.strip()) for l in open(path)
                   if l.strip() and not l.strip().startswith("@"))
    v = np.frombuffer(raw, "<f2").astype(np.float16).copy()
    m = (len(v) // _WORD) * _WORD
    if m:
        v[:m] = v[:m].reshape(-1, _WORD)[:, ::-1].reshape(-1)
    return v


def score(dump, ref):
    """verify.py-style verdict of dump vs ref (both fp16)."""
    n = min(len(dump), len(ref))
    if n == 0:
        return "LEN", 0, 0, 0
    du = dump[:n].view(np.uint16).astype(np.int32)
    ru = ref[:n].view(np.uint16).astype(np.int32)
    sm = lambda u: np.where(u & 0x8000, -(u & 0x7FFF), u & 0x7FFF)
    bad = ((du & 0x7C00) == 0x7C00) | ((ru & 0x7C00) == 0x7C00)
    ulp = np.where(bad, 0xFFFF, np.abs(sm(du) - sm(ru)))
    absd = np.where(bad, np.inf,
                    np.abs(dump[:n].astype(np.float32) - ref[:n].astype(np.float32)))
    within = (ulp <= 1) | (absd <= 0.001)
    npass = int(within.sum())
    finite_fail = int(np.sum(~within & ~bad))
    if len(dump) != len(ref):
        verdict = "LEN"
    elif npass == n:
        verdict = "PASS"
    elif finite_fail == 0:
        verdict = "NAN-EDGE"
    else:
        verdict = "FAIL"
    return verdict, npass, n, finite_fail


def ref_bytes(ref_file):
    return sum(len(l.strip()) // 2 for l in open(ref_file)
              if l.strip() and not l.strip().startswith("@"))


def run_op(op):
    cfg = OP_CONFIG[op]
    kernel = cfg["kernel"]
    ddir = os.path.join(GG, cfg["dir"], "data")
    ksrc = os.path.join(GG, cfg["dir"], f"{kernel}.c")
    inp = os.path.join(ddir, f"{kernel}_input.txt")
    ref = os.path.join(ddir, f"{kernel}_ref.txt")
    npg = os.path.join(ddir, f"{kernel}_numpy_golden.txt")
    elf = os.path.join(WORK, f"{kernel}.elf")
    dump = os.path.join(WORK, f"{kernel}_pyspike.hex")
    for p in (ksrc, inp, ref):
        if not os.path.exists(p):
            return op, "NO-SRC", ""
    osize = ref_bytes(ref)
    b = subprocess.run(["bash", BUILD, ksrc, elf], capture_output=True, text=True,
                       env=dict(os.environ, EXTRA_INC=GG))
    if b.returncode != 0 or not os.path.exists(elf):
        return op, "BUILD-ERR", b.stderr.strip().splitlines()[-1] if b.stderr else ""
    env = dict(os.environ, GTX_MX_IO_DTYPE="float16", GTX_DDR_REVERSE_MODE="elem",
               GTX_DDR_SIZE="2G", GTX_DDR_INIT=inp, GTX_DDR_DUMP=dump,
               GTX_DDR_DUMP_ADDR="0x37f000000", GTX_DDR_DUMP_SIZE=str(osize),
               GTX_DDR_REVERSED="1", UV_LINK_MODE="copy")
    r = subprocess.run(
        ["uv", "run", "--no-sync", "pyspike", "--extlib=riscv.gtx",
         "--extension=gtx", "--device=gtx_ddr,0x370000000", elf],
        capture_output=True, text=True, env=env, timeout=240)
    if not os.path.exists(dump):
        return op, "RUN-ERR", (r.stderr.strip().splitlines() or [""])[-1]
    d = decode(dump)
    nv, np_, nn, nff = score(d, decode(npg)) if os.path.exists(npg) else ("NOGOLD", 0, 0, 0)
    iv, ip_, inn, iff = score(d, decode(ref))
    return op, nv, f"vs_numpy={np_}/{nn} ff={nff}  vs_ISS[{iv}]={ip_}/{inn} ff={iff}"


def main():
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    ops = sel or [op for op, c in OP_CONFIG.items()
                  if os.path.exists(os.path.join(GG, c["dir"], "data",
                                                 f"{c['kernel']}_numpy_golden.txt"))]
    print(f"[verify_pyspike] base={GG}  ops={len(ops)}")
    tally = {}
    for op in ops:
        try:
            name, verdict, detail = run_op(op)
        except subprocess.TimeoutExpired:
            name, verdict, detail = op, "TIMEOUT", ""
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"  {name:<16} {verdict:<9} {detail}")
    print("\n[verify_pyspike] " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
