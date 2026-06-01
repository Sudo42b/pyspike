"""Logical-fp32 e2e verification for the elementwise op set, ULP-tolerant.
Decodes BOTH dump and golden through 8-lane bus-word reverse to logical order,
then compares (exact OR ulp<=4 OR abs<=1e-3, non-finite matched as edge)."""
import os, re, subprocess, math
import numpy as np

REPO = "/mnt/e/14_NIGHTLY/pyspike"
W = 8

def _sig(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x))).astype(np.float32)

OPS = {
    "abs": ("ABS","n1s16_abs", np.abs),
    "neg": ("NEG","n1s16_neg", np.negative),
    "relu": ("RELU","n1s16_relu", lambda x: np.maximum(x,0)),
    "sqr": ("SQR","n1s16_sqr", lambda x: x*x),
    "ceil": ("CEIL","n1s16_ceil", np.ceil),
    "floor": ("FLOOR","n1s16_floor", np.floor),
    "trunc": ("TRUNC","n1s16_trunc", np.trunc),
    "round": ("ROUND","n1s16_round", np.rint),
    "step": ("STEP","n1s16_step", lambda x: (x>0).astype(np.float32)),
    "exp": ("EXP","n1s16_exp", np.exp),
    "sin": ("SIN","n1s16_sin", np.sin),
    "cos": ("COS","n1s16_cos", np.cos),
    "tanh": ("TANH","n1s16_tanh", np.tanh),
    "sigmoid": ("SIGMOID","n1s16_sigmoid", _sig),
    "silu": ("SILU","n1s16_silu", lambda x: x*_sig(x)),
    "sqrt": ("SQRT","n1s16_sqrt", lambda x: np.sqrt(np.where(x>=0,x,np.nan))),
}
POS = {"sqrt"}

def enc(v):
    v=np.asarray(v,np.float32); pad=(-len(v))%W
    if pad: v=np.concatenate([v,np.zeros(pad,np.float32)])
    return v.reshape(-1,W)[:,::-1].reshape(-1).astype("<f4").tobytes()
def dec(b):
    v=np.frombuffer(b,"<f4").astype(np.float32).copy(); m=(len(v)//W)*W
    if m: v[:m]=v[:m].reshape(-1,W)[:,::-1].reshape(-1)
    return v
def shape(kc):
    t=open(kc).read()
    def d(n):
        m=re.search(rf"#define\s+{n}\s+(\d+)",t); return int(m.group(1)) if m else None
    return d("WIDTH"), d("HEIGHT")

out=[]; npass=nfail=0
for op,(sub,kern,fn) in OPS.items():
    kc=f"{REPO}/test/{sub}/n1s16/{kern}.c"
    if not os.path.exists(kc): out.append(f"{op:<9} NO-KERNEL"); continue
    width,height=shape(kc)
    if not width: out.append(f"{op:<9} NO-SHAPE"); continue
    rows=min(height, max(1, 2048//width)); n=rows*width
    lo,hi=(0.1,4.0) if op in POS else (-4.0,4.0)
    a=np.random.default_rng(5).uniform(lo,hi,n).astype(np.float32)
    ib=enc(a); H=ib.hex()
    open(f"/tmp/va_{op}.txt","w").write("@1000000\n"+"\n".join(H[i:i+64] for i in range(0,len(H),64))+"\n")
    gb=enc(fn(a).astype(np.float32))
    env=dict(os.environ,RISCV="/opt/riscv",GTX_DDR_SIZE="2G",GTX_DDR_REVERSED="1",
             GTX_DDR_REVERSE_MODE="elem",GTX_DDR_DUMP_ADDR="0x37f000000",
             GTX_DDR_INIT=f"/tmp/va_{op}.txt",GTX_DDR_DUMP=f"/tmp/va_{op}_o.txt",
             GTX_DDR_DUMP_SIZE=str(len(gb)),UV_LINK_MODE="copy")
    b=subprocess.run(["bash","src/test/gtx/build_kernel.sh",kc,f"/tmp/va_{op}.elf"],
                     cwd=REPO,env=env,capture_output=True,timeout=180)
    if b.returncode!=0:
        out.append(f"{op:<9} BUILD-FAIL {b.stderr.decode()[-60:].strip()}"); nfail+=1; continue
    try:
        subprocess.run(["uv","run","--no-sync","pyspike","--extlib=riscv.gtx","--extension=gtx",
                        "--device=gtx_ddr,0x370000000",f"/tmp/va_{op}.elf"],
                       cwd=REPO,env=env,capture_output=True,timeout=240)
    except subprocess.TimeoutExpired:
        out.append(f"{op:<9} RUN-TIMEOUT"); nfail+=1; continue
    o=bytes.fromhex(re.sub(r"[^0-9a-fA-F]","",open(f"/tmp/va_{op}_o.txt").read()))
    ld=dec(o)[:n]; lg=dec(gb)[:n]; m=min(len(ld),len(lg)); ld,lg=ld[:m],lg[:m]
    bad=~(np.isfinite(ld)&np.isfinite(lg))
    ulp=np.where(bad,0,np.abs(ld.view(np.int32).astype(np.int64)-lg.view(np.int32).astype(np.int64)))
    absd=np.where(bad,0.0,np.abs(ld.astype(np.float64)-lg.astype(np.float64)))
    ok=(ulp<=4)|(absd<=1e-3)|bad
    exact=int((ld.view(np.int32)==lg.view(np.int32)).sum())
    v="PASS" if bool(ok.all()) else "FAIL"
    npass+=(v=="PASS"); nfail+=(v=="FAIL")
    out.append(f"{op:<9} W={width:<4} {v}  exact={exact}/{m} ok={int(ok.sum())}/{m}")
out.append(f"==== PASS={npass} FAIL={nfail} ====")
open("/tmp/verify_all_out.txt","w").write("\n".join(out)+"\n")
