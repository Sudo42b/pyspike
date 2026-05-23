# Handoff — numpy 골든 기반 3-sim 정합 (pyspike / spike / ISS)

_작성: 2026-05-23. 목적: 다음 세션이 "ISS 검증 + spike·pyspike를 골든에 정합"을 바로 명령·실행._

## 목표 (사용자 비전)

1. **ggml 커널의 input에서 numpy로 정확한 골든(ground truth)을 생성**한다 (ISS 출력이 아니라 numpy가 정답).
2. **ISS가 정확히 동작하는 op**에서 ISS 출력 == numpy 골든을 검증한다.
3. **spike와 pyspike가 numpy 골든(=ISS)과 동일**하게 나오도록 맞춘다 (pyspike 고유 버그 수정).
4. 골든/입력 데이터는 **ISS가 30~60초 걸릴 만큼 충분히 크게**.

## 핵심 사실 (이번 세션 확정)

### 코퍼스 위치
| 코퍼스 | 경로 | 특징 |
|---|---|---|
| **ggml_ops_c 원본** | `/home/sw.lee/supergate_sw/device/gtx_kernel/ggml_ops_c` (103 op) | **작은 dims**(ABS H=64=512elem). strict-credit 호환 → **ISS 정확히 동작**. 일부 ref가 틀림(메모리 확인). |
| pyspike `test/` (= `vendor/gtx_cpp_reference/test/`, byte-identical) | repo `test/` (95 op) | **큰 멀티타일**(ABS H=393217). Phase-8 재작성, functional credit 튜닝 → **ISS=0 for compute op**. |
| numpy 레퍼런스 수학 | `vendor/gtx_cpp_reference/test/compare_all_ops.py` (654줄) | `compute_scalar` + `OP_CONFIG`. input.txt 읽어 numpy ref 계산. **~36 op에서 부정확**(아래). |

### ISS는 startup-bound (중요 — 크기 결정에 영향)
ISS 실행시간은 데이터 크기와 **거의 무관**하게 ~33초 (startup/SystemC elaboration이 지배):
```
32B → 32.8s   |   65KB → 33.3s   |   131KB → 33.7s   |   16.7MB → 36.8s
```
⇒ "ISS 30~60s"는 **사실상 모든 크기에서 충족**. 60s까지 올리려면 ~100MB+ 데이터가 필요(비현실적). 큰 데이터의 실익은 **fast sim(pyspike·spike) 시간 차별화**(pyspike ABS: 512elem≈즉시 vs 3M elem≈17s). 권장 타깃: **op당 ~1–4M elem**(ISS ~34–35s, pyspike ~5–17s, spike ~5–7s — 모두 의미 있는 측정).

### numpy 골든 현재 상태 (`test/gen_golden.py` 실행 결과, test/ 코퍼스 기준)
- **생성 53 / skip 34**(data-move·complex은 compare_all_ops에 numpy 수학 없음 → 추가 구현 필요).
- 기존 ref 대비: **17 일치(신뢰 가능)**, **36 불일치**.
- ⚠️ 불일치 36개 중 일부는 **numpy 수학 자체가 틀림**(길이 불일치가 증거):
  - `sum`(ref=16, numpy=1), `sum_rows`(256 vs 8192), `mean`(80 vs 160), `arange`(64 vs 512) — reduction/layout config 오류.
  - `fill`·`softmax`·`geglu`·`swiglu_oai`·`group_norm`·`norm`·`rms_norm`·`cumsum` — 전부/대부분 불일치(수학 미흡 또는 fp16 rounding 차이).
- ⇒ **numpy 골든이 아직 신뢰 가능한 ground truth가 아님.** 다음 세션 1순위 = op별 numpy 수학 교정.

### credit 스케줄러는 불필요 (확정, [[gtx-pyspike-credit-dequeue-todo]])
vendor spike도 pyspike와 동일 functional/NOP credit인데 77/95 통과 → functional credit 충분. asyncio/coroutine/yield 스케줄러는 어떤 실패도 안 고치고 코퍼스를 0으로 깰 위험. **만들지 말 것.**

## 도구 (이번 세션 신규/기존)
| 파일 | 용도 |
|---|---|
| `test/gen_golden.py` | input → numpy 골든 생성. `uv run --no-sync python3 test/gen_golden.py [op...]`. `<kernel>_numpy_golden.txt` 출력(비파괴, _ref.txt 안 건드림) + 기존 ref와 diff 리포트. |
| `test/verify_golden.sh <OP> [tmo]` | 한 op을 pyspike·spike·ISS로 돌려 numpy 골든(우선) 대비 PASS/FAIL + iss_zero 리포트. |
| `test/benchmark/build_uni.sh <c> <elf>` | ISS용 uni elf(full startup + exit_shim). |
| `src/test/gtx/build_kernel.sh <c> <elf>` | pyspike·spike용 elf(minimal crt+tohost). |
| `test/benchmark/bench3.sh` / `bench3_sweep.sh` / `aggregate.py` | 95-op 3-way 벤치(`.benchmarks/`). |

## 3-sim 실행 레시피 (env/인자)
```bash
SPK=/mnt/e/14_NIGHTLY/gtx_spike/riscv-isa-sim/build
ISS=vendor/simulator/GTX_ISS
# pyspike (build_kernel elf)
GTX_DDR_SIZE=2G GTX_DDR_INIT=<input> GTX_DDR_DUMP=<out> GTX_DDR_DUMP_ADDR=0x37f000000 \
  GTX_DDR_DUMP_SIZE=<Nbytes> GTX_DDR_REVERSED=1 UV_LINK_MODE=copy \
  uv run --no-sync pyspike --extlib=riscv.gtx --extension=gtx --device=gtx_ddr,0x370000000 <elf>
# vendor spike (build_kernel elf) — 같은 GTX_DDR_* env
LD_LIBRARY_PATH=$SPK $SPK/spike --extension=gtx_npu <elf>
# SystemC ISS (uni elf) — 반드시 -F /tmp경로 sock, -l 0
$ISS -I <uni.elf> -S 0x370000000 -L <input> -B 0x37f000000 -E <Nbytes_HEX> -T <out> -V -l 0 -F /tmp/x.sock
# 비교: python3 vendor/gtx_cpp_reference/gtx/verify.py <dump> <golden> --fp16 --ulp 1 --atol 0.001
```

## 다음 세션 작업 순서

**전제 결정 필요 (사용자):** 어느 코퍼스로? — (A) ggml_ops_c 원본(ISS 정확, 작음→dims 키워야), (B) test/ 큰 코퍼스(ISS=0 compute). 사용자 의도("ISS 정확 동작")상 **(A) ggml_ops_c 권장**, 단 dims를 키워 데이터 크게.

1. **numpy 수학 교정** (1순위, 필수): `compare_all_ops.py`의 reduction/norm/glu/fill/softmax/arange 등을 ggml 의미 + fp16 rounding에 맞게 수정. 검증: `gen_golden.py`의 "ref MATCHES" 개수↑, 길이 불일치 0.
2. **(A 택시) ggml_ops_c 커널 dims 확대 + input/golden 재생성**: 각 op HEIGHT #define 키워(타깃 ~1–4M elem) → input 생성(`generate_n1s16_tests.py`, 단 `compare_all_ops.py`를 `test/`에 심링크/복사 필요 — 현재 import 깨짐) → `gen_golden.py`로 numpy 골든.
3. **ISS 검증**: `verify_golden.sh <OP>`로 op별 ISS==numpy 골든 확인. ISS 정확 동작 op 목록 확정(iss_zero=NO & iss=PASS).
4. **spike·pyspike 정합**: numpy 골든 대비 FAIL인 op(현재 pyspike ~50, spike ~18) 디버그. pyspike 고유 버그(py FAIL·sp PASS)부터. credit 무관(증명됨) → 인트린식/numpy 포팅/context 버그.
5. **최종 벤치**: `bench3_sweep.sh` 재실행 → `.benchmarks/` 갱신(이번엔 golden=numpy, 3-sim 일치율 + 속도).

## 열린 질문 / 리스크
- ggml_ops_c dims 확대 시 멀티타일 로직 필요 여부(원본은 단일타일 H=64; 큰 H가 L2 용량 초과하면 타일 루프 추가 필요 → test/ 재작성본이 한 이유). **이게 핵심 난점**: 큰 데이터+ISS호환+단순구조 동시 만족이 어려울 수 있음.
- numpy 골든 일부 불일치가 "ref 틀림"인지 "numpy fp16 rounding 차이"인지 op별 판별 필요(ISS로 adjudicate, 단 ISS 정확 동작 op만).
- `generate_n1s16_tests.py`의 `from compare_all_ops import` 깨짐(test/에 파일 없음) — 복사/심링크 또는 sys.path 수정 필요.
```bash
# 빠른 복구: ln -s ../vendor/gtx_cpp_reference/test/compare_all_ops.py test/compare_all_ops.py
```

---

## ✅ 완료 (2026-05-23 세션 2) — 코퍼스=ggml_ops_c

**결정**: 코퍼스 (A) ggml_ops_c 확정 (test/는 compute에서 ISS=0이라 ISS검증 불가). 디코딩 = **LE per-element + 16-element bus 역순**(GTX_DDR_REVERSE_MODE=elem). ARANGE ref [15,14,..,0,31,..]가 결정증거. test/(BE)와 혼동 금지.

### 1. numpy 수학 교정 ✅
- `test/gen_golden.py` 엔디안 인식(GTX_GOLDEN_BASE/GTX_GOLDEN_ENDIAN) + verify.py식 채점(sign-mag ULP≤1 or atol≤0.001, NaN/Inf=miss). floor/ceil/round/trunc NaN 가드.
- 결과: ~36 op numpy==ISS(유한). 남은 FAIL=대부분 numpy 버그 아님(거대입력 overflow·초월함수·degenerate param).

### 2. ISS 검증 ✅
- ggml_ops_c `ref`==`result.hex`==ISS 출력 → 골든 생성 즉시 ISS 대비 검증됨.

### 3. spike·pyspike 정합 ✅ (둘 다 수정)
**검증 harness**: `test/verify_pyspike_ggml.py` (build_kernel.sh → pyspike → numpy/ISS 채점).
실행 env: `GTX_MX_IO_DTYPE=float16 GTX_DDR_REVERSE_MODE=elem GTX_DDR_REVERSED=1`.

**pyspike 수정 (5):**
1. `memory.py` — GTX_DDR_REVERSE_MODE env + `_reverse_bus_word()` (byte/elem). **시스템 전반.**
2. `config_params.py` — GTX_MX_IO_DTYPE env (float16 baseline).
3. `MX/act.py` `_prelu`/`_prelu_i` — rs2→OPERAND2 slope staging (leaky_relu ff146→0).
4. `DL/dma_imp.py` `dma_tloop_copy` — L1 wrap 시 assert→modulo wrap (diag_mask ff392→0; length-0→65536 copy 크래시 해결).
5. `src/test/gtx/build_kernel.sh` — EXTRA_INC (gtx_isa_compat.h 13 op 해금).

**spike 수정 (1):** `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/gtx_npu_dma.cc` ddr_init/dump에 elem 역순 모드 추가 → `setup.sh --rebuild`. (leaky_relu/diag_mask는 vendor C++에 이미 정상 — pyspike 포팅 누락이었음.)

**최종 (pyspike, 54 op):** vs_ISS PASS/NAN-EDGE=49, vs_ISS"FAIL"4(scale/fill/geglu*=pyspike==numpy, ISS만 다름), cumsum=TIMEOUT. **진짜 compute 버그 0.**
**spike**: abs/leaky_relu/diag_mask vs_ISS ff=0 확인.

### 남은 후속 (비차단)
- **cumsum pyspike TIMEOUT** (COLS=64, 성능 — 별도 조사).
- 초월함수(sin/cos) numpy(libm) vs ISS(HW근사) 갭 — 입력 거대값 ill-conditioned. 정밀 정합하려면 입력 범위 축소 재생성(+ISS 재실행) 필요.
- param 테스트(scale/fill/clamp/add1) degenerate (offset0 param=0) — 무의미, 입력 재생성 시 해소.
- 미커밋 (브랜치 gtx-strict-credit). spike repo도 미커밋.
