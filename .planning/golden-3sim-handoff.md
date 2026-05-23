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
