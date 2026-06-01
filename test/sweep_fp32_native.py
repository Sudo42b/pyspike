#!/usr/bin/env python3
"""/tmp/sweep_fp32.py — build+run+compare each op in unified fp32 (results to file).

Uses the same _DIRS map as /tmp/gen_fp32.py so it finds exactly the goldens that
generator produced. Compares the DDR dump to <kernel>_fp32_golden.txt by
byte-exact and fp32 ULP/abs tolerance.
"""
import os
import re
import subprocess
import sys

import numpy as np

REPO = "/mnt/e/14_NIGHTLY/pyspike"
RESULTS = "/tmp/sweep_fp32_results.txt"
ENV = dict(os.environ, RISCV="/opt/riscv", GTX_DDR_SIZE="2G",
           GTX_DDR_REVERSED="1", GTX_DDR_REVERSE_MODE="elem",
           GTX_DDR_DUMP_ADDR="0x37f000000", UV_LINK_MODE="copy")

_DIRS = {
    "abs": ("ABS", "n1s16_abs"), "neg": ("NEG", "n1s16_neg"),
    "sqr": ("SQR", "n1s16_sqr"), "sqrt": ("SQRT", "n1s16_sqrt"),
    "step": ("STEP", "n1s16_step"), "sign": ("SIGN", "n1s16_sign"),
    "floor": ("FLOOR", "n1s16_floor"), "ceil": ("CEIL", "n1s16_ceil"),
    "trunc": ("TRUNC", "n1s16_trunc"), "round": ("ROUND", "n1s16_round"),
    "exp": ("EXP", "n1s16_exp"), "sin": ("SIN", "n1s16_sin"),
    "cos": ("COS", "n1s16_cos"), "log": ("LOG", "n1s16_log"),
    "tanh": ("TANH", "n1s16_tanh"), "sigmoid": ("SIGMOID", "n1s16_sigmoid"),
    "silu": ("SILU", "n1s16_silu"), "gelu": ("GELU", "n1s16_gelu"),
    "gelu_erf": ("GELU_ERF", "n1s16_gelu_erf"),
    "hardswish": ("HARDSWISH", "n1s16_hardswish"),
    "hardsigmoid": ("HARDSIGMOID", "n1s16_hardsigmoid"),
    "softplus": ("SOFTPLUS", "n1s16_softplus"),
    "add_vv": ("ADD", "n1s16_add_vv"), "sub_vv": ("SUB", "n1s16_sub_vv"),
    "mul_vv": ("MUL", "n1s16_mul_vv"), "div_vv": ("DIV", "n1s16_div_vv"),
    "acc": ("ACC", "n1s16_acc"),
    "leaky_relu": ("LEAKY_RELU", "n1s16_leaky_relu"),
}


def _cmp(dump, golden):
    o = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", open(dump).read()))
    n = min(len(o), len(golden))
    if n == 0:
        return "EMPTY", 0, 0, 0
    exact = int((np.frombuffer(o[:n], np.uint8) == np.frombuffer(golden[:n], np.uint8)).sum())
    m = (n // 4) * 4
    oi = np.frombuffer(o[:m], "<i4").astype(np.int64)
    gi = np.frombuffer(golden[:m], "<i4").astype(np.int64)
    of = np.frombuffer(o[:m], "<f4"); gf = np.frombuffer(golden[:m], "<f4")
    bad = ~(np.isfinite(of) & np.isfinite(gf))
    ulp = np.where(bad, 0, np.abs(oi - gi))
    absd = np.where(bad, 0.0, np.abs(of.astype(np.float64) - gf.astype(np.float64)))
    within = (ulp <= 2) | (absd <= 1e-3) | bad
    verdict = "PASS" if (exact == n or bool(within.all())) else "FAIL"
    return verdict, exact, int(within.sum()), m // 4


def main():
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    ops = sel if sel else list(_DIRS)
    open(RESULTS, "w").write("")
    npass = nfail = nskip = 0
    for op in ops:
        if op not in _DIRS:
            continue
        sub, k = _DIRS[op]
        ddir = os.path.join(REPO, "test", sub, "n1s16", "data")
        gold = os.path.join(ddir, f"{k}_fp32_golden.txt")
        inp = os.path.join(ddir, f"{k}_input.txt")
        kc = os.path.join(REPO, "test", sub, "n1s16", f"{k}.c")
        if not (os.path.exists(gold) and os.path.exists(kc)):
            line = f"{op:<14} SKIP no-golden/kernel"; nskip += 1
        else:
            elf = f"/tmp/sw_{k}.elf"
            b = subprocess.run(["bash", "src/test/gtx/build_kernel.sh", kc, elf],
                               cwd=REPO, env=ENV, capture_output=True, timeout=180)
            if b.returncode != 0:
                line = f"{op:<14} BUILD-FAIL {b.stderr.decode()[-100:].strip()}"; nfail += 1
            else:
                g = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", open(gold).read()))
                dump = f"/tmp/sw_{k}_o.txt"
                e2 = dict(ENV, GTX_DDR_INIT=inp, GTX_DDR_DUMP=dump, GTX_DDR_DUMP_SIZE=str(len(g)))
                try:
                    subprocess.run(["uv", "run", "--no-sync", "pyspike", "--extlib=riscv.gtx",
                                    "--extension=gtx", "--device=gtx_ddr,0x370000000", elf],
                                   cwd=REPO, env=e2, capture_output=True, timeout=300)
                    v, exact, within, nel = _cmp(dump, g)
                    npass += (v == "PASS"); nfail += (v != "PASS")
                    line = f"{op:<14} {v}  exact={exact}/{len(g)} ok={within}/{nel}"
                except subprocess.TimeoutExpired:
                    line = f"{op:<14} RUN-TIMEOUT"; nfail += 1
        with open(RESULTS, "a") as f:
            f.write(line + "\n")
    with open(RESULTS, "a") as f:
        f.write(f"==== PASS={npass} FAIL={nfail} SKIP={nskip} ====\n")


if __name__ == "__main__":
    main()
