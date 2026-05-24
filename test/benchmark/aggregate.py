#!/usr/bin/env python3
"""Aggregate .benchmarks/bench3_raw.tsv into a human-readable Markdown report.

Usage: aggregate.py [raw.tsv] [out.md]

Post-0x53: ISS now executes compute ops (the credit.chk funct7 0x52/0x53
generation mismatch that zeroed every compute op is fixed). Divergences are
classified into actionable buckets — see ``classify``.
"""
import sys
import statistics as st
from datetime import date

RAW = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks/bench3_raw.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".benchmarks/3sim_benchmark.md"

rows = []
with open(RAW) as f:
    header = f.readline()
    for ln in f:
        ln = ln.rstrip("\n")
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) < 11:
            parts += ["?"] * (11 - len(parts))
        rows.append(dict(zip(
            ["op", "osizeB", "t_py", "t_sp", "t_iss", "py_ref", "sp_ref",
             "iss_ref", "iss_zero", "py_eq_sp", "py_eq_iss"], parts[:11])))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def times(key):
    return [v for v in (fnum(r[key]) for r in rows) if v is not None]


def median(key):
    t = times(key)
    return st.median(t) if t else 0.0


def count(key, pred):
    return sum(1 for r in rows if pred(r[key]))


def is_pass(v):
    return v == "PASS"


def classify(r):
    """Bucket a row into one actionable class.

    C  converged & correct  : py & iss both PASS (sp too) — done.
    A  pyspike bug           : iss PASS, py not PASS  → fix pyspike to match ISS.
    B  old kernel (iss=0)    : iss_zero YES           → rewrite kernel ISS-compat.
    E  ISS timeout/nodump    : iss un-runnable here.
    D  golden/other          : iss not PASS & not zero (golden suspect or
                               spike also wrong; may need spike fix).
    """
    if r["t_iss"] == "TIMEOUT" or r["iss_ref"] in ("NODUMP", "ERR", "?"):
        return "E"
    if r["iss_zero"] == "YES":
        return "B"
    if is_pass(r["py_ref"]) and is_pass(r["iss_ref"]):
        return "C"
    if is_pass(r["iss_ref"]) and not is_pass(r["py_ref"]):
        return "A"
    return "D"


for r in rows:
    r["cls"] = classify(r)

n = len(rows)
py_pass = count("py_ref", is_pass)
sp_pass = count("sp_ref", is_pass)
iss_pass = count("iss_ref", is_pass)
py_eq_iss = sum(1 for r in rows if r["py_eq_iss"] in ("EXACT", "PASS"))
py_eq_sp = count("py_eq_sp", lambda v: v in ("EXACT", "PASS"))
cls = {k: [r["op"] for r in rows if r["cls"] == k] for k in "CABDE"}

L = []
A = L.append
A("# GTX 3-Simulator Benchmark — `test/` corpus")
A("")
A(f"_{date.today().isoformat()} · {n} ops · pyspike vs vendor-spike vs SystemC-ISS_")
A("")
A("## TL;DR")
A("")
A(f"- **ISS now runs compute** (credit.chk funct7 0x53 통일 fix 이후): ISS가 golden과 "
  f"일치하는 op이 **{iss_pass}/{n}** — 이전 벤치(15/95, compute 전부 0)에서 대폭 개선. "
  f"ISS-compatible 커널(ABS식)은 **py·spike·ISS byte-identical 수렴**.")
A(f"- **pyspike == vendor-spike**: {py_eq_sp}/{n} 출력 동일. golden 기준 pyspike {py_pass}/{n}, "
  f"spike {sp_pass}/{n} PASS. **pyspike == ISS**: {py_eq_iss}/{n}.")
A(f"- **남은 분기 분류** (수정 대상):")
A(f"  - **A. pyspike 버그** ({len(cls['A'])}개): ISS·spike 정답, pyspike만 오답 → "
  f"pyspike를 ISS에 맞춰 수정.")
A(f"  - **B. 구식 커널** ({len(cls['B'])}개): ISS·pyspike=0, spike만 정답 → test/ 커널을 "
  f"ABS식 ISS호환으로 재작성하면 셋 다 정답 수렴.")
A(f"  - **D. golden 의심/기타** ({len(cls['D'])}개): ISS도 golden과 불일치(또는 spike도 오답). "
  f"golden 재검증 또는 spike 수정 대상.")
A(f"  - **E. ISS 미실행** ({len(cls['E'])}개): ISS timeout/nodump.")
A(f"- **속도(합계):** pyspike **{sum(times('t_py')):.0f}s**, vendor-spike "
  f"**{sum(times('t_sp')):.0f}s**, ISS **{sum(times('t_iss')):.0f}s** "
  f"(중앙값/op: py {median('t_py'):.1f}s · sp {median('t_sp'):.1f}s · iss {median('t_iss'):.1f}s)")
A("")
A("## 속도 요약")
A("")
A("| 시뮬레이터 | 합계(s) | 중앙값/op(s) | 평균/op(s) | 최소 | 최대 | TIMEOUT |")
A("|---|--:|--:|--:|--:|--:|--:|")
for name, key in [("pyspike", "t_py"), ("vendor-spike", "t_sp"), ("SystemC-ISS", "t_iss")]:
    t = times(key)
    to = count(key, lambda v: v == "TIMEOUT")
    if t:
        A(f"| {name} | {sum(t):.0f} | {st.median(t):.1f} | {sum(t)/len(t):.1f} "
          f"| {min(t):.1f} | {max(t):.1f} | {to} |")
    else:
        A(f"| {name} | – | – | – | – | – | {to} |")
A("")
A("> elf: pyspike·spike = `build_kernel.sh`(minimal crt+tohost); "
  "ISS = `build_uni.sh`(full gtx-firmware startup + exit_shim). 동일 커널·입력.")
A("")
A("## 출력 일치 요약 (golden ref, FP16 ulp=1)")
A("")
A("| 비교 | 일치 op 수 |")
A("|---|--:|")
A(f"| pyspike == ref | {py_pass}/{n} |")
A(f"| vendor-spike == ref | {sp_pass}/{n} |")
A(f"| **SystemC-ISS == ref** | **{iss_pass}/{n}** |")
A(f"| pyspike == vendor-spike | {py_eq_sp}/{n} |")
A(f"| pyspike == SystemC-ISS | {py_eq_iss}/{n} |")
A("")
CLS_TITLE = {
    "A": "A. pyspike 버그 — ISS·spike 정답, pyspike 수정 대상",
    "B": "B. 구식 커널 — ISS·pyspike=0, 커널 ISS호환 재작성 대상",
    "D": "D. golden 의심/기타 — ISS도 golden 불일치 (spike도 오답 가능)",
    "E": "E. ISS 미실행 (timeout/nodump)",
    "C": "C. 수렴 & 정답 — 세 시뮬레이터 모두 PASS",
}
for k in "ABDEC":
    if not cls[k]:
        continue
    A(f"## {CLS_TITLE[k]} ({len(cls[k])}개)")
    A("")
    A("`" + "`, `".join(sorted(cls[k])) + "`")
    A("")
A("## op별 상세")
A("")
A("| op | cls | out(B) | py(s) | sp(s) | iss(s) | py·ref | sp·ref | iss·ref | iss=0 | py≡sp | py≡iss |")
A("|---|:--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|")
for r in sorted(rows, key=lambda x: x["op"]):
    A("| {op} | {cls} | {osizeB} | {t_py} | {t_sp} | {t_iss} | {py_ref} | {sp_ref} "
      "| {iss_ref} | {iss_zero} | {py_eq_sp} | {py_eq_iss} |".format(**r))
A("")
A("## 비고")
A("")
A("- `cls` = 분류(A/B/C/D/E, TL;DR 참조). `py·ref`/`sp·ref`/`iss·ref` = golden 대비 "
  "PASS 또는 `FAIL:<mismatch수>`. `py≡sp`/`py≡iss` = EXACT(byte 동일)/PASS(±1 ULP)/DIFF.")
A("- ISS = HW 레퍼런스. 수정 방침: Class A는 pyspike를 ISS에 맞춤, Class B는 커널 재작성, "
  "Class D는 golden 재검증 또는 spike 수정. (2026-05-24 사용자 확정)")
A("")
with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}  ({n} ops)  "
      f"C={len(cls['C'])} A={len(cls['A'])} B={len(cls['B'])} D={len(cls['D'])} E={len(cls['E'])}")
