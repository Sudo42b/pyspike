# GTX 3-Simulator Benchmark — `test/` corpus

_2026-05-25 · 95 ops · pyspike vs vendor-spike vs SystemC-ISS_

## TL;DR

- **ISS now runs compute** (credit.chk funct7 0x53 통일 fix 이후): ISS가 golden과 일치하는 op이 **78/95** — 이전 벤치(15/95, compute 전부 0)에서 대폭 개선. ISS-compatible 커널(ABS식)은 **py·spike·ISS byte-identical 수렴**.
- **pyspike == vendor-spike**: 94/95 출력 동일. golden 기준 pyspike 79/95, spike 79/95 PASS. **pyspike == ISS**: 86/95.
- **남은 분기 분류** (수정 대상):
  - **A. pyspike 버그** (1개): ISS·spike 정답, pyspike만 오답 → pyspike를 ISS에 맞춰 수정.
  - **B. 구식 커널** (0개): ISS·pyspike=0, spike만 정답 → test/ 커널을 ABS식 ISS호환으로 재작성하면 셋 다 정답 수렴.
  - **D. golden 의심/기타** (15개): ISS도 golden과 불일치(또는 spike도 오답). golden 재검증 또는 spike 수정 대상.
  - **E. ISS 미실행** (2개): ISS timeout/nodump.
- **속도(합계):** pyspike **492s**, vendor-spike **433s**, ISS **3427s** (중앙값/op: py 2.1s · sp 4.4s · iss 34.1s)

## 속도 요약

| 시뮬레이터 | 합계(s) | 중앙값/op(s) | 평균/op(s) | 최소 | 최대 | TIMEOUT |
|---|--:|--:|--:|--:|--:|--:|
| pyspike | 492 | 2.1 | 5.2 | 1.6 | 107.0 | 1 |
| vendor-spike | 433 | 4.4 | 4.6 | 3.7 | 8.4 | 1 |
| SystemC-ISS | 3427 | 34.1 | 36.9 | 31.6 | 121.7 | 2 |

> elf: pyspike·spike = `build_kernel.sh`(minimal crt+tohost); ISS = `build_uni.sh`(full gtx-firmware startup + exit_shim). 동일 커널·입력.

## 출력 일치 요약 (golden ref, FP16 ulp=1)

| 비교 | 일치 op 수 |
|---|--:|
| pyspike == ref | 79/95 |
| vendor-spike == ref | 79/95 |
| **SystemC-ISS == ref** | **78/95** |
| pyspike == vendor-spike | 94/95 |
| pyspike == SystemC-ISS | 86/95 |

## A. pyspike 버그 — ISS·spike 정답, pyspike 수정 대상 (1개)

`LOG`

## D. golden 의심/기타 — ISS도 golden 불일치 (spike도 오답 가능) (15개)

`CONV_TRANSPOSE_2D`, `CUMSUM`, `CUMSUM_v2`, `FLASH_ATTN_EXT`, `GROUP_NORM`, `NORM`, `RMS_NORM`, `RWKV_WKV7`, `SOLVE_TRI`, `SSM_SCAN`, `SUM`, `SUM_ROWS`, `SWIGLU_OAI`, `TIMESTEP_EMBEDDING`, `XIELU`

## E. ISS 미실행 (timeout/nodump) (2개)

`ARGSORT`, `UPSCALE`

## C. 수렴 & 정답 — 세 시뮬레이터 모두 PASS (77개)

`ABS`, `ACC`, `ADD`, `ADD1`, `ADD_ID`, `ADD_REL_POS`, `ARANGE`, `ARGMAX`, `CEIL`, `CLAMP`, `CONCAT`, `CONV_2D`, `CONV_2D_DW`, `CONV_3D`, `CONV_TRANSPOSE_1D`, `COS`, `COUNT_EQUAL`, `CPY`, `DIAG`, `DIAG_MASK_INF`, `DIAG_MASK_ZERO`, `DIV`, `DUP`, `ELU`, `EXP`, `EXPM1`, `FILL`, `FLOOR`, `GATED_LINEAR_ATTN`, `GEGLU`, `GEGLU_ERF`, `GEGLU_QUICK`, `GELU`, `GELU_ERF`, `GELU_QUICK`, `GET_ROWS`, `HARDSIGMOID`, `HARDSWISH`, `IM2COL`, `IM2COL_3D`, `L2_NORM`, `LEAKY_RELU`, `MEAN`, `MUL`, `MUL_MAT`, `MUL_MAT_ID`, `NEG`, `OUT_PROD`, `PAD`, `PAD_REFLECT_1D`, `POOL_1D`, `POOL_2D`, `REGLU`, `RELU`, `REPEAT`, `ROLL`, `ROPE`, `ROUND`, `RWKV_WKV6`, `SCALE`, `SET`, `SET_ROWS`, `SGN`, `SIGMOID`, `SILU`, `SIN`, `SOFTPLUS`, `SOFT_MAX`, `SQR`, `SQRT`, `SSM_CONV`, `STEP`, `SUB`, `TANH`, `TOP_K`, `TRI`, `TRUNC`

## op별 상세

| op | cls | out(B) | py(s) | sp(s) | iss(s) | py·ref | sp·ref | iss·ref | iss=0 | py≡sp | py≡iss |
|---|:--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ABS | C | 6291488 | 16.0 | 5.4 | 77.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ACC | C | 16764960 | 7.0 | 8.4 | 45.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD | C | 2097152 | 13.4 | 4.7 | 55.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD1 | C | 16764960 | 6.2 | 8.3 | 43.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD_ID | C | 6839392 | 3.6 | 5.6 | 39.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD_REL_POS | C | 12160 | 2.0 | 4.6 | 34.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ARANGE | C | 128 | 3.1 | 4.5 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ARGMAX | C | 1024 | 2.0 | 4.5 | 34.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ARGSORT | E | 262144 | 105.4 | 4.6 | TIMEOUT | PASS | PASS | NODUMP | NODUMP | EXACT | DIFF |
| CEIL | C | 131072 | 2.2 | 4.2 | 33.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CLAMP | C | 8192 | 1.7 | 4.2 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONCAT | C | 2106400 | 2.0 | 4.2 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_2D | C | 2208 | 2.0 | 4.2 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_2D_DW | C | 128 | 2.0 | 5.5 | 32.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_3D | C | 1600 | 4.0 | 6.0 | 38.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_TRANSPOSE_1D | C | 65536 | 5.8 | 4.8 | 38.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_TRANSPOSE_2D | D | 1024 | 2.4 | 4.2 | 32.7 | FAIL:1 | FAIL:1 | FAIL:1 | NO | EXACT | EXACT |
| COS | C | 8192 | 2.1 | 3.7 | 33.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| COUNT_EQUAL | C | 32 | 7.2 | 4.7 | 41.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CPY | C | 65472 | 1.8 | 4.4 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CUMSUM | D | 14336 | 2.1 | 4.1 | 34.0 | FAIL:1236 | FAIL:1236 | FAIL:1236 | NO | EXACT | EXACT |
| CUMSUM_v2 | D | 32768 | 2.0 | 4.4 | 34.3 | FAIL:2741 | FAIL:2741 | FAIL:2741 | NO | EXACT | EXACT |
| DIAG | C | 65536 | 1.8 | 5.7 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DIAG_MASK_INF | C | 8192 | 1.7 | 4.1 | 34.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DIAG_MASK_ZERO | C | 8192 | 2.3 | 5.9 | 33.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DIV | C | 4160 | 4.2 | 4.9 | 32.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DUP | C | 65280 | 3.8 | 4.2 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ELU | C | 2560 | 1.8 | 4.7 | 34.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| EXP | C | 32992 | 2.4 | 4.8 | 35.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| EXPM1 | C | 2048 | 1.9 | 5.3 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| FILL | C | 65408 | 1.7 | 4.0 | 34.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| FLASH_ATTN_EXT | D | 131072 | 1.8 | 6.2 | 34.6 | FAIL:64264 | FAIL:64264 | FAIL:64264 | NO | EXACT | EXACT |
| FLOOR | C | 8192 | 1.7 | 4.2 | 34.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GATED_LINEAR_ATTN | C | 96 | 2.3 | 4.0 | 34.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GEGLU | C | 4096 | 1.8 | 4.2 | 34.0 | PASS | PASS | PASS | NO | PASS | DIFF |
| GEGLU_ERF | C | 4096 | 1.8 | 4.2 | 34.3 | PASS | PASS | PASS | NO | PASS | DIFF |
| GEGLU_QUICK | C | 262144 | 2.1 | 4.5 | 35.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GELU | C | 61440 | 2.4 | 4.4 | 34.1 | PASS | PASS | PASS | NO | PASS | DIFF |
| GELU_ERF | C | 4194304 | 2.8 | 4.9 | 38.0 | PASS | PASS | PASS | NO | PASS | DIFF |
| GELU_QUICK | C | 65024 | 2.3 | 4.3 | 35.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GET_ROWS | C | 262144 | 1.8 | 4.7 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GROUP_NORM | D | 65280 | 1.8 | 5.9 | 33.8 | FAIL:74 | FAIL:74 | FAIL:74 | NO | EXACT | EXACT |
| HARDSIGMOID | C | 7904 | 1.9 | 3.9 | 34.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| HARDSWISH | C | 8192 | 1.8 | 4.0 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| IM2COL | C | 4608 | 1.7 | 4.2 | 33.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| IM2COL_3D | C | 94400 | 2.1 | 4.3 | 35.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| L2_NORM | C | 8192 | 2.7 | 4.7 | 34.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| LEAKY_RELU | C | 63776 | 1.9 | 4.3 | 33.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| LOG | A | 16256 | TIMEOUT | TIMEOUT | 34.2 | FAIL:8121 | FAIL:8121 | PASS | NO | EXACT | DIFF |
| MEAN | C | 160 | 2.3 | 4.5 | 34.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| MUL | C | 2097152 | 15.6 | 4.6 | 53.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| MUL_MAT | C | 1048576 | 2.4 | 4.8 | 37.1 | PASS | PASS | PASS | NO | PASS | DIFF |
| MUL_MAT_ID | C | 2048 | 1.7 | 4.6 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| NEG | C | 34816 | 1.8 | 3.7 | 34.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| NORM | D | 16448 | 2.2 | 4.1 | 35.3 | FAIL:41 | FAIL:41 | FAIL:41 | NO | EXACT | EXACT |
| OUT_PROD | C | 1024 | 1.8 | 4.1 | 33.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| PAD | C | 65024 | 2.3 | 3.9 | 34.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| PAD_REFLECT_1D | C | 49152 | 3.9 | 4.0 | 33.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| POOL_1D | C | 4096 | 1.8 | 4.0 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| POOL_2D | C | 1664 | 1.7 | 6.2 | 31.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| REGLU | C | 4096 | 1.9 | 4.0 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RELU | C | 524288 | 1.8 | 3.9 | 34.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| REPEAT | C | 1091808 | 2.5 | 4.7 | 32.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RMS_NORM | D | 2097152 | 2.5 | 4.7 | 36.0 | FAIL:772 | FAIL:756 | FAIL:772 | NO | FAIL:81 | EXACT |
| ROLL | C | 2105344 | 2.5 | 6.7 | 34.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ROPE | C | 2048 | 2.5 | 4.5 | 34.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ROUND | C | 32768 | 2.1 | 4.1 | 36.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RWKV_WKV6 | C | 32 | 1.6 | 3.7 | 33.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RWKV_WKV7 | D | 2112 | 2.6 | 4.4 | 33.7 | FAIL:169 | FAIL:169 | FAIL:169 | NO | EXACT | EXACT |
| SCALE | C | 8192 | 1.8 | 4.1 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SET | C | 2105344 | 2.1 | 4.2 | 34.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SET_ROWS | C | 2097152 | 2.1 | 4.4 | 36.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SGN | C | 7936 | 2.3 | 4.6 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SIGMOID | C | 8192 | 1.9 | 4.9 | 34.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SILU | C | 131072 | 1.9 | 4.4 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SIN | C | 131072 | 4.8 | 4.3 | 35.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SOFTPLUS | C | 131072 | 1.8 | 4.1 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SOFT_MAX | C | 131072 | 2.7 | 4.8 | 35.7 | PASS | PASS | PASS | NO | PASS | DIFF |
| SOLVE_TRI | D | 65536 | 107.0 | 4.5 | 121.7 | FAIL:3311 | FAIL:3311 | FAIL:3311 | NO | EXACT | EXACT |
| SQR | C | 131072 | 1.8 | 4.3 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SQRT | C | 131072 | 2.3 | 4.6 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SSM_CONV | C | 131072 | 16.6 | 4.3 | 69.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SSM_SCAN | D | 1536 | 2.6 | 5.0 | 35.7 | FAIL:635 | FAIL:635 | FAIL:635 | NO | EXACT | EXACT |
| STEP | C | 131072 | 1.8 | 3.8 | 34.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SUB | C | 131072 | 1.7 | 4.7 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SUM | D | 32 | 1.7 | 4.5 | 32.9 | FAIL:1 | FAIL:1 | FAIL:1 | NO | EXACT | EXACT |
| SUM_ROWS | D | 512 | 1.8 | 4.5 | 34.0 | FAIL:28 | FAIL:28 | FAIL:28 | NO | EXACT | EXACT |
| SWIGLU_OAI | D | 65536 | 1.7 | 4.1 | 33.1 | FAIL:4 | FAIL:4 | FAIL:4 | NO | EXACT | EXACT |
| TANH | C | 131072 | 1.7 | 3.9 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TIMESTEP_EMBEDDING | D | 131072 | 4.2 | 4.5 | 34.7 | FAIL:9170 | FAIL:9170 | FAIL:9170 | NO | EXACT | EXACT |
| TOP_K | C | 4096 | 3.8 | 4.9 | 35.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TRI | C | 131072 | 1.8 | 4.3 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TRUNC | C | 131072 | 1.8 | 4.0 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| UPSCALE | E | 131072 | 5.9 | 4.2 | TIMEOUT | PASS | PASS | NODUMP | NODUMP | EXACT | DIFF |
| XIELU | D | 131072 | 2.3 | 4.2 | 34.1 | FAIL:31 | FAIL:31 | FAIL:31 | NO | EXACT | EXACT |

## 비고

- `cls` = 분류(A/B/C/D/E, TL;DR 참조). `py·ref`/`sp·ref`/`iss·ref` = golden 대비 PASS 또는 `FAIL:<mismatch수>`. `py≡sp`/`py≡iss` = EXACT(byte 동일)/PASS(±1 ULP)/DIFF.
- ISS = HW 레퍼런스. 수정 방침: Class A는 pyspike를 ISS에 맞춤, Class B는 커널 재작성, Class D는 golden 재검증 또는 spike 수정. (2026-05-24 사용자 확정)

