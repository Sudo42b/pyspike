"""Logical-fp32 e2e verification: decode BOTH dump and golden through the same
8-lane bus-word reverse to logical order, then compare values (ULP/abs). This is
layout-agnostic, so non-32B-aligned row widths verify cleanly too.

For each op: fresh fp32 input (flat 8-lane encoded) -> build -> run -> compare
logical(dump) vs logical(-a / |a| / ...) reference.
"""
import os, re, subprocess
import numpy as np

REPO = "/mnt/e/14_NIGHTLY/pyspike"
W = 8

DIRS = {
    "abs": ("ABS", "n1s16_abs", np.abs),
    "neg": ("NEG", "n1s16_neg", np.negative),
    "relu": ("RELU", "n1s16_relu", lambda x: np.maximum(x, 0.0)),
    "sqr": ("SQR", "n1s16_sqr", lambda x: x * x),
    "ceil": ("CEIL", "n1s16_ceil", np.ceil),
    "floor": ("FLOOR", "n1s16_floor", np.floor),
}


def enc(v):
    v = np.asarray(v, np.float32)
    pad = (-len(v)) % W
    if pad:
        v = np.concatenate([v, np.zeros(pad, np.float32)])
    return v.reshape(-1, W)[:, ::-1].reshape(-1).astype("<f4").tobytes()


def dec(byts):
    v = np.frombuffer(byts, "<f4").astype(np.float32).copy()
    m = (len(v) // W) * W
    if m:
        v[:m] = v[:m].reshape(-1, W)[:, ::-1].reshape(-1)
    return v


def shape(kc):
    t = open(kc).read()
    def d(n):
        m = re.search(rf"#define\s+{n}\s+(\d+)", t)
        return int(m.group(1)) if m else None
    return (d("WIDTH"), d("HEIGHT"))


out = []
for op, (sub, kern, fn) in DIRS.items():
    kc = f"{REPO}/test/{sub}/n1s16/{kern}.c"
    if not os.path.exists(kc):
        out.append(f"{op:<8} NO-KERNEL"); continue
    width, height = shape(kc)
    # cap the element count so runs stay fast; keep it a multiple of WIDTH so full
    # rows are processed, and feed via a small DDR window.
    rows = min(height, max(1, 2048 // max(width, 1)))
    n = rows * width
    a = np.random.default_rng(3).uniform(-5, 5, n).astype(np.float32)
    ib = enc(a)
    lines = ["@1000000"] + [ib.hex()[i:i+64] for i in range(0, len(ib.hex()), 64)]
    open(f"/tmp/vl_{op}_in.txt", "w").write("\n".join(lines) + "\n")
    golden_bytes = enc(fn(a).astype(np.float32))
    env = dict(os.environ, RISCV="/opt/riscv", GTX_DDR_SIZE="2G", GTX_DDR_REVERSED="1",
               GTX_DDR_REVERSE_MODE="elem", GTX_DDR_DUMP_ADDR="0x37f000000",
               GTX_DDR_INIT=f"/tmp/vl_{op}_in.txt", GTX_DDR_DUMP=f"/tmp/vl_{op}_o.txt",
               GTX_DDR_DUMP_SIZE=str(len(golden_bytes)), UV_LINK_MODE="copy")
    b = subprocess.run(["bash", "src/test/gtx/build_kernel.sh", kc, f"/tmp/vl_{op}.elf"],
                       cwd=REPO, env=env, capture_output=True, timeout=180)
    if b.returncode != 0:
        out.append(f"{op:<8} BUILD-FAIL"); continue
    r = subprocess.run(["uv", "run", "--no-sync", "pyspike", "--extlib=riscv.gtx",
                        "--extension=gtx", "--device=gtx_ddr,0x370000000", f"/tmp/vl_{op}.elf"],
                       cwd=REPO, env=env, capture_output=True, timeout=200)
    o = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", open(f"/tmp/vl_{op}_o.txt").read()))
    ld = dec(o)[:n]
    lg = dec(golden_bytes)[:n]
    m = min(len(ld), len(lg))
    ld, lg = ld[:m], lg[:m]
    fin = np.isfinite(ld) & np.isfinite(lg)
    exact = int((ld.view(np.int32) == lg.view(np.int32)).sum())
    ulp = np.abs(ld[fin].view(np.int32).astype(np.int64) - lg[fin].view(np.int32).astype(np.int64))
    okulp = int((ulp <= 2).sum()) + int((~fin).sum())
    v = "PASS" if (exact == m or okulp == m) else "FAIL"
    out.append(f"{op:<8} W={width:<3} {v}  logical_exact={exact}/{m} ulp<=2={okulp}/{m}")

open("/tmp/verify_logical_out.txt", "w").write("\n".join(out) + "\n")
