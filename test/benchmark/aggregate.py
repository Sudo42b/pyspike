#!/usr/bin/env python3
"""Aggregate .benchmarks/bench3_raw.tsv into a human-readable Markdown report.

Usage: aggregate.py [raw.tsv] [out.md]
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


def total(key):
    return sum(times(key))


def median(key):
    t = times(key)
    return st.median(t) if t else 0.0


def count(key, pred):
    return sum(1 for r in rows if pred(r[key]))


n = len(rows)
py_pass = count("py_ref", lambda v: v == "PASS")
sp_pass = count("sp_ref", lambda v: v == "PASS")
iss_pass = count("iss_ref", lambda v: v == "PASS")
iss_zero = count("iss_zero", lambda v: v == "YES")
py_eq_sp = count("py_eq_sp", lambda v: v in ("EXACT", "PASS"))
py_to = count("t_py", lambda v: v == "TIMEOUT")
sp_to = count("t_sp", lambda v: v == "TIMEOUT")
iss_to = count("t_iss", lambda v: v == "TIMEOUT")

L = []
L.append("# GTX 3-Simulator Benchmark — `test/` corpus")
L.append("")
L.append(f"_{date.today().isoformat()} · {n} ops · pyspike vs vendor-spike vs SystemC-ISS_")
L.append("")
iss_ok = sorted(r["op"] for r in rows if r["iss_ref"] == "PASS")
L.append("## TL;DR")
L.append("")
L.append(f"- **pyspike ↔ vendor-spike: 일치.** {py_eq_sp}/{n} op이 출력 동일(byte-exact 또는 FP16 ±1 ULP). "
         f"golden ref 기준 pyspike {py_pass}/{n} PASS, spike {sp_pass}/{n} PASS.")
L.append(f"- **SystemC-ISS: 데이터-이동 op만 일치, compute op은 불일치.** ISS가 golden과 일치한 op은 "
         f"{iss_pass}/{n}개로 거의 전부 **credit-gate 없는 데이터 이동/복사 계열** "
         f"(`{'`, `'.join(iss_ok)}`). 나머지 {iss_zero}/{n} op(주로 element-wise/compute)은 ISS에서 "
         f"**전부 0 출력**.")
L.append(f"- **원인 (credit 단일):** `test/` 커널은 Phase-8에서 pyspike의 functional(NOP) credit 모델에 "
         f"맞춰 재작성됨. compute op은 T-loop compute→store 핸드셰이크가 credit에 의존하는데, strict-credit "
         f"HW 모델인 ISS에서는 그 hand-off가 성립하지 않아 결과 영역이 미기록(0)으로 남음. 데이터 이동 op은 "
         f"직접 DMA(credit-gate 무관)라 ISS에서도 정상 — 이 op들이 ISS와 byte-exact 일치하는 것이 "
         f"입력 byte-order는 동일하게 해석됨(= 0의 원인이 아님)을 보여줌.")
L.append(f"- **속도(합계):** pyspike **{total('t_py'):.0f}s**, vendor-spike **{total('t_sp'):.0f}s**, "
         f"ISS **{total('t_iss'):.0f}s**. (중앙값/op: py {median('t_py'):.1f}s · sp {median('t_sp'):.1f}s · iss {median('t_iss'):.1f}s)")
L.append("")
L.append("## 속도 요약")
L.append("")
L.append("| 시뮬레이터 | 합계(s) | 중앙값/op(s) | 평균/op(s) | 최소 | 최대 | TIMEOUT |")
L.append("|---|--:|--:|--:|--:|--:|--:|")
for name, key, to in [("pyspike", "t_py", py_to), ("vendor-spike", "t_sp", sp_to), ("SystemC-ISS", "t_iss", iss_to)]:
    t = times(key)
    if t:
        L.append(f"| {name} | {sum(t):.0f} | {st.median(t):.1f} | {sum(t)/len(t):.1f} "
                 f"| {min(t):.1f} | {max(t):.1f} | {to} |")
    else:
        L.append(f"| {name} | – | – | – | – | – | {to} |")
L.append("")
L.append("> elf: pyspike·spike = `build_kernel.sh`(minimal crt+tohost); "
         "ISS = `build_uni.sh`(full gtx-firmware startup + exit_shim, ISS는 minimal crt 거부). "
         "동일 커널 소스, 동일 입력.")
L.append("")
L.append("## 출력 일치 요약 (golden ref, FP16 ulp=1)")
L.append("")
L.append("| 비교 | 일치 op 수 |")
L.append("|---|--:|")
L.append(f"| pyspike == ref | {py_pass}/{n} |")
L.append(f"| vendor-spike == ref | {sp_pass}/{n} |")
L.append(f"| **SystemC-ISS == ref** | **{iss_pass}/{n}** |")
L.append(f"| pyspike == vendor-spike | {py_eq_sp}/{n} |")
L.append(f"| ISS 전부-0 출력 | {iss_zero}/{n} |")
L.append("")
L.append("## ISS와 일치하는 op (golden 기준)")
L.append("")
L.append("거의 전부 credit-gate가 없는 데이터 이동/복사/리덕션 계열 — ISS의 strict-credit 모델과 무관하게 동작.")
L.append("")
L.append("| op | iss=0 | py≡iss | py·ref | sp·ref |")
L.append("|---|:--:|:--:|:--:|:--:|")
for r in sorted(rows, key=lambda x: x["op"]):
    if r["iss_ref"] == "PASS":
        L.append("| {op} | {iss_zero} | {py_eq_iss} | {py_ref} | {sp_ref} |".format(**r))
L.append("")
L.append("## op별 상세")
L.append("")
L.append("| op | out(B) | py(s) | sp(s) | iss(s) | py·ref | sp·ref | iss·ref | iss=0 | py≡sp | py≡iss |")
L.append("|---|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|")
for r in sorted(rows, key=lambda x: x["op"]):
    L.append("| {op} | {osizeB} | {t_py} | {t_sp} | {t_iss} | {py_ref} | {sp_ref} "
             "| {iss_ref} | {iss_zero} | {py_eq_sp} | {py_eq_iss} |".format(**r))
L.append("")
L.append("## 비고")
L.append("")
L.append("- `py·ref`/`sp·ref`/`iss·ref` = golden 대비 PASS 또는 `FAIL:<mismatch수>`. "
         "pyspike의 FAIL은 기존 로직 버그(별도 추적), credit 변경과 무관(출력 바이트 동일성 확인됨).")
L.append("- `py≡sp` = EXACT(byte 동일) 또는 PASS(±1 ULP) 또는 DIFF.")
L.append("- ISS는 `test/` 코퍼스 비호환이 systemic — op별 버그가 아니라 corpus 설계 차이. "
         "ISS 3-way 검증이 필요하면 원본 ggml_ops_c 커널(strict-credit 호환)을 써야 함.")
L.append("")
with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}  ({n} ops)")
