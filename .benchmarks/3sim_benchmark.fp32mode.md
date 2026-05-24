# GTX 3-Simulator Benchmark — `test/` corpus

_2026-05-24 · 95 ops · pyspike vs vendor-spike vs SystemC-ISS_

## TL;DR

- **ISS now runs compute** (credit.chk funct7 0x53 통일 fix 이후): ISS가 golden과 일치하는 op이 **76/95** — 이전 벤치(15/95, compute 전부 0)에서 대폭 개선. ISS-compatible 커널(ABS식)은 **py·spike·ISS byte-identical 수렴**.
- **pyspike == vendor-spike**: 49/95 출력 동일. golden 기준 pyspike 45/95, spike 77/95 PASS. **pyspike == ISS**: 35/95.
- **남은 분기 분류** (수정 대상):
  - **A. pyspike 버그** (34개): ISS·spike 정답, pyspike만 오답 → pyspike를 ISS에 맞춰 수정.
  - **B. 구식 커널** (2개): ISS·pyspike=0, spike만 정답 → test/ 커널을 ABS식 ISS호환으로 재작성하면 셋 다 정답 수렴.
  - **D. golden 의심/기타** (15개): ISS도 golden과 불일치(또는 spike도 오답). golden 재검증 또는 spike 수정 대상.
  - **E. ISS 미실행** (2개): ISS timeout/nodump.
- **속도(합계):** pyspike **481s**, vendor-spike **457s**, ISS **3326s** (중앙값/op: py 2.3s · sp 4.7s · iss 33.4s)

## 속도 요약

| 시뮬레이터 | 합계(s) | 중앙값/op(s) | 평균/op(s) | 최소 | 최대 | TIMEOUT |
|---|--:|--:|--:|--:|--:|--:|
| pyspike | 481 | 2.3 | 5.1 | 1.6 | 109.8 | 1 |
| vendor-spike | 457 | 4.7 | 4.9 | 3.5 | 11.0 | 1 |
| SystemC-ISS | 3326 | 33.4 | 35.8 | 32.1 | 121.3 | 2 |

> elf: pyspike·spike = `build_kernel.sh`(minimal crt+tohost); ISS = `build_uni.sh`(full gtx-firmware startup + exit_shim). 동일 커널·입력.

## 출력 일치 요약 (golden ref, FP16 ulp=1)

| 비교 | 일치 op 수 |
|---|--:|
| pyspike == ref | 45/95 |
| vendor-spike == ref | 77/95 |
| **SystemC-ISS == ref** | **76/95** |
| pyspike == vendor-spike | 49/95 |
| pyspike == SystemC-ISS | 35/95 |

## A. pyspike 버그 — ISS·spike 정답, pyspike 수정 대상 (34개)

`ADD_ID`, `ADD_REL_POS`, `ARANGE`, `CLAMP`, `CONV_2D`, `CONV_2D_DW`, `CONV_3D`, `CONV_TRANSPOSE_1D`, `COUNT_EQUAL`, `DIAG_MASK_INF`, `DIAG_MASK_ZERO`, `GATED_LINEAR_ATTN`, `GEGLU`, `GEGLU_ERF`, `GEGLU_QUICK`, `HARDSIGMOID`, `HARDSWISH`, `IM2COL`, `IM2COL_3D`, `L2_NORM`, `LOG`, `MEAN`, `MUL_MAT`, `MUL_MAT_ID`, `OUT_PROD`, `POOL_1D`, `POOL_2D`, `REGLU`, `ROPE`, `RWKV_WKV6`, `SCALE`, `SIN`, `SOFT_MAX`, `SSM_CONV`

## B. 구식 커널 — ISS·pyspike=0, 커널 ISS호환 재작성 대상 (2개)

`ADD`, `MUL`

## D. golden 의심/기타 — ISS도 golden 불일치 (spike도 오답 가능) (15개)

`CONV_TRANSPOSE_2D`, `CUMSUM`, `CUMSUM_v2`, `FLASH_ATTN_EXT`, `GROUP_NORM`, `NORM`, `RMS_NORM`, `RWKV_WKV7`, `SOLVE_TRI`, `SSM_SCAN`, `SUM`, `SUM_ROWS`, `SWIGLU_OAI`, `TIMESTEP_EMBEDDING`, `XIELU`

## E. ISS 미실행 (timeout/nodump) (2개)

`ARGSORT`, `UPSCALE`

## C. 수렴 & 정답 — 세 시뮬레이터 모두 PASS (42개)

`ABS`, `ACC`, `ADD1`, `ARGMAX`, `CEIL`, `CONCAT`, `COS`, `CPY`, `DIAG`, `DIV`, `DUP`, `ELU`, `EXP`, `EXPM1`, `FILL`, `FLOOR`, `GELU`, `GELU_ERF`, `GELU_QUICK`, `GET_ROWS`, `LEAKY_RELU`, `NEG`, `PAD`, `PAD_REFLECT_1D`, `RELU`, `REPEAT`, `ROLL`, `ROUND`, `SET`, `SET_ROWS`, `SGN`, `SIGMOID`, `SILU`, `SOFTPLUS`, `SQR`, `SQRT`, `STEP`, `SUB`, `TANH`, `TOP_K`, `TRI`, `TRUNC`

## op별 상세

| op | cls | out(B) | py(s) | sp(s) | iss(s) | py·ref | sp·ref | iss·ref | iss=0 | py≡sp | py≡iss |
|---|:--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ABS | C | 6291488 | 16.7 | 5.2 | 77.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ACC | C | 16764960 | 9.6 | 7.2 | 45.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD | B | 2097152 | 3.7 | 4.6 | 34.0 | FAIL:1047308 | PASS | FAIL:1047308 | YES | FAIL:1047308 | EXACT |
| ADD1 | C | 16764960 | 5.4 | 9.3 | 41.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ADD_ID | A | 6839392 | 3.7 | 6.9 | 37.4 | FAIL:3416163 | PASS | PASS | NO | FAIL:3415660 | DIFF |
| ADD_REL_POS | A | 12160 | 2.7 | 4.3 | 35.3 | FAIL:6074 | PASS | PASS | NO | FAIL:6074 | DIFF |
| ARANGE | A | 128 | 2.0 | 4.7 | 33.2 | FAIL:64 | PASS | PASS | NO | FAIL:64 | DIFF |
| ARGMAX | C | 1024 | 3.2 | 4.7 | 34.3 | PASS | PASS | PASS | NO | PASS | DIFF |
| ARGSORT | E | 262144 | 109.8 | 5.8 | TIMEOUT | PASS | PASS | NODUMP | NODUMP | PASS | DIFF |
| CEIL | C | 131072 | 2.8 | 4.2 | 34.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CLAMP | A | 8192 | 2.0 | 3.6 | 33.1 | FAIL:4089 | PASS | PASS | NO | FAIL:4089 | DIFF |
| CONCAT | C | 2106400 | 2.4 | 3.8 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_2D | A | 2208 | 2.1 | 3.5 | 33.5 | FAIL:1089 | PASS | PASS | NO | FAIL:1089 | DIFF |
| CONV_2D_DW | A | 128 | 1.8 | 4.2 | 34.2 | FAIL:56 | PASS | PASS | NO | FAIL:56 | DIFF |
| CONV_3D | A | 1600 | 4.4 | 5.1 | 39.0 | FAIL:799 | PASS | PASS | NO | FAIL:799 | DIFF |
| CONV_TRANSPOSE_1D | A | 65536 | 8.1 | 4.3 | 38.4 | FAIL:32767 | PASS | PASS | NO | FAIL:32767 | DIFF |
| CONV_TRANSPOSE_2D | D | 1024 | 3.2 | 4.9 | 34.7 | FAIL:511 | FAIL:1 | FAIL:1 | NO | FAIL:511 | DIFF |
| COS | C | 8192 | 2.5 | 4.7 | 33.3 | PASS | PASS | PASS | NO | PASS | DIFF |
| COUNT_EQUAL | A | 32 | 1.8 | 11.0 | 41.0 | FAIL:1 | FAIL:1 | PASS | NO | EXACT | DIFF |
| CPY | C | 65472 | 1.7 | 4.2 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CUMSUM | D | 14336 | 2.2 | 4.7 | 33.7 | FAIL:7053 | FAIL:1236 | FAIL:1236 | NO | FAIL:7054 | DIFF |
| CUMSUM_v2 | D | 32768 | 2.5 | 4.7 | 33.6 | FAIL:16099 | FAIL:2741 | FAIL:2741 | NO | FAIL:16100 | DIFF |
| DIAG | C | 65536 | 1.7 | 4.7 | 33.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DIAG_MASK_INF | A | 8192 | 1.9 | 4.2 | 32.7 | FAIL:474 | PASS | PASS | NO | FAIL:474 | DIFF |
| DIAG_MASK_ZERO | A | 8192 | 2.1 | 4.9 | 32.8 | FAIL:1388 | PASS | PASS | NO | FAIL:1388 | DIFF |
| DIV | C | 4160 | 1.8 | 4.3 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| DUP | C | 65280 | 2.7 | 4.0 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ELU | C | 2560 | 2.4 | 4.4 | 33.0 | PASS | PASS | PASS | NO | PASS | DIFF |
| EXP | C | 32992 | 2.0 | 5.5 | 32.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| EXPM1 | C | 2048 | 1.9 | 3.9 | 33.5 | PASS | PASS | PASS | NO | PASS | DIFF |
| FILL | C | 65408 | 1.6 | 4.9 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| FLASH_ATTN_EXT | D | 131072 | 1.9 | 4.4 | 33.0 | FAIL:64264 | FAIL:64264 | FAIL:64264 | NO | EXACT | EXACT |
| FLOOR | C | 8192 | 1.8 | 4.3 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GATED_LINEAR_ATTN | A | 96 | 2.0 | 4.4 | 33.0 | FAIL:36 | PASS | PASS | NO | FAIL:36 | DIFF |
| GEGLU | A | 4096 | 2.0 | 4.4 | 32.9 | FAIL:2018 | PASS | PASS | NO | FAIL:2018 | DIFF |
| GEGLU_ERF | A | 4096 | 2.7 | 4.4 | 32.9 | FAIL:2018 | PASS | PASS | NO | FAIL:2018 | DIFF |
| GEGLU_QUICK | A | 262144 | 2.1 | 4.4 | 33.5 | FAIL:129203 | PASS | PASS | NO | FAIL:129199 | DIFF |
| GELU | C | 61440 | 2.4 | 5.9 | 33.3 | PASS | PASS | PASS | NO | PASS | DIFF |
| GELU_ERF | C | 4194304 | 2.9 | 4.8 | 35.0 | PASS | PASS | PASS | NO | PASS | DIFF |
| GELU_QUICK | C | 65024 | 2.5 | 4.5 | 33.7 | PASS | PASS | PASS | NO | PASS | DIFF |
| GET_ROWS | C | 262144 | 1.8 | 4.7 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GROUP_NORM | D | 65280 | 1.6 | 4.7 | 33.8 | FAIL:32618 | FAIL:32618 | FAIL:74 | NO | EXACT | DIFF |
| HARDSIGMOID | A | 7904 | 2.7 | 4.8 | 33.1 | FAIL:3937 | PASS | PASS | NO | FAIL:3937 | DIFF |
| HARDSWISH | A | 8192 | 1.9 | 4.4 | 32.9 | FAIL:4085 | PASS | PASS | NO | FAIL:4085 | DIFF |
| IM2COL | A | 4608 | 1.7 | 5.2 | 33.6 | FAIL:2040 | PASS | PASS | NO | FAIL:2040 | DIFF |
| IM2COL_3D | A | 94400 | 3.5 | 4.3 | 34.6 | FAIL:44518 | PASS | PASS | NO | FAIL:44518 | DIFF |
| L2_NORM | A | 8192 | 2.7 | 4.5 | 33.9 | FAIL:4087 | PASS | PASS | NO | FAIL:4087 | DIFF |
| LEAKY_RELU | C | 63776 | 1.9 | 4.4 | 33.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| LOG | A | 16256 | TIMEOUT | TIMEOUT | 33.7 | FAIL:8121 | FAIL:8121 | PASS | NO | EXACT | DIFF |
| MEAN | A | 160 | 2.4 | 4.6 | 33.3 | FAIL:80 | PASS | PASS | NO | FAIL:80 | DIFF |
| MUL | B | 2097152 | 2.3 | 5.0 | 34.2 | FAIL:1040245 | PASS | FAIL:1040245 | YES | FAIL:1040245 | EXACT |
| MUL_MAT | A | 1048576 | 2.2 | 5.3 | 36.7 | FAIL:524261 | PASS | PASS | NO | FAIL:524261 | DIFF |
| MUL_MAT_ID | A | 2048 | 2.7 | 4.6 | 33.3 | FAIL:1024 | PASS | PASS | NO | FAIL:1024 | DIFF |
| NEG | C | 34816 | 2.7 | 4.9 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| NORM | D | 16448 | 1.9 | 4.3 | 32.9 | FAIL:8221 | FAIL:8219 | FAIL:41 | NO | FAIL:5494 | DIFF |
| OUT_PROD | A | 1024 | 1.9 | 5.3 | 32.9 | FAIL:256 | PASS | PASS | NO | FAIL:256 | DIFF |
| PAD | C | 65024 | 2.3 | 4.6 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| PAD_REFLECT_1D | C | 49152 | 1.8 | 5.4 | 32.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| POOL_1D | A | 4096 | 2.0 | 4.4 | 32.8 | FAIL:2045 | PASS | PASS | NO | FAIL:2045 | DIFF |
| POOL_2D | A | 1664 | 1.7 | 4.1 | 33.0 | FAIL:821 | PASS | PASS | NO | FAIL:821 | DIFF |
| REGLU | A | 4096 | 1.8 | 4.3 | 32.1 | FAIL:1014 | PASS | PASS | NO | FAIL:1014 | DIFF |
| RELU | C | 524288 | 2.4 | 4.9 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| REPEAT | C | 1091808 | 2.0 | 4.3 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RMS_NORM | D | 2097152 | 2.6 | 4.5 | 34.8 | FAIL:1047945 | FAIL:756 | FAIL:772 | NO | FAIL:1047945 | DIFF |
| ROLL | C | 2105344 | 3.0 | 5.7 | 34.0 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ROPE | A | 2048 | 2.4 | 4.8 | 33.4 | FAIL:1024 | PASS | PASS | NO | FAIL:1024 | DIFF |
| ROUND | C | 32768 | 2.1 | 5.3 | 33.9 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RWKV_WKV6 | A | 32 | 2.0 | 4.0 | 32.9 | FAIL:8 | PASS | PASS | NO | FAIL:8 | DIFF |
| RWKV_WKV7 | D | 2112 | 2.9 | 5.4 | 32.9 | FAIL:1055 | FAIL:169 | FAIL:169 | NO | FAIL:1054 | DIFF |
| SCALE | A | 8192 | 1.7 | 5.9 | 33.0 | FAIL:2043 | PASS | PASS | NO | FAIL:2043 | DIFF |
| SET | C | 2105344 | 2.9 | 4.8 | 34.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SET_ROWS | C | 2097152 | 2.6 | 5.2 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SGN | C | 7936 | 1.8 | 4.3 | 32.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SIGMOID | C | 8192 | 2.9 | 4.3 | 32.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SILU | C | 131072 | 1.8 | 4.5 | 33.1 | PASS | PASS | PASS | NO | PASS | DIFF |
| SIN | A | 131072 | 5.5 | 4.6 | 35.0 | FAIL:32733 | FAIL:65467 | PASS | NO | FAIL:32989 | DIFF |
| SOFTPLUS | C | 131072 | 2.0 | 4.5 | 32.9 | PASS | PASS | PASS | NO | PASS | DIFF |
| SOFT_MAX | A | 131072 | 2.8 | 4.5 | 33.4 | FAIL:28416 | PASS | PASS | NO | FAIL:28416 | DIFF |
| SOLVE_TRI | D | 65536 | 107.1 | 4.6 | 121.3 | FAIL:268 | FAIL:3311 | FAIL:3311 | NO | FAIL:1468 | DIFF |
| SQR | C | 131072 | 1.7 | 4.6 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SQRT | C | 131072 | 2.8 | 5.0 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SSM_CONV | A | 131072 | 18.0 | 4.3 | 69.6 | FAIL:65515 | PASS | PASS | NO | FAIL:65516 | DIFF |
| SSM_SCAN | D | 1536 | 2.9 | 5.6 | 33.8 | FAIL:768 | FAIL:635 | FAIL:635 | NO | FAIL:768 | DIFF |
| STEP | C | 131072 | 2.3 | 5.0 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SUB | C | 131072 | 1.8 | 5.2 | 32.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SUM | D | 32 | 2.1 | 4.3 | 32.7 | FAIL:1 | FAIL:1 | FAIL:1 | NO | FAIL:1 | DIFF |
| SUM_ROWS | D | 512 | 1.8 | 5.9 | 33.4 | FAIL:256 | FAIL:28 | FAIL:28 | NO | FAIL:256 | DIFF |
| SWIGLU_OAI | D | 65536 | 1.7 | 4.6 | 34.3 | FAIL:32394 | FAIL:4 | FAIL:4 | NO | FAIL:32392 | DIFF |
| TANH | C | 131072 | 2.6 | 5.6 | 33.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TIMESTEP_EMBEDDING | D | 131072 | 2.8 | 4.8 | 34.4 | FAIL:65143 | FAIL:9170 | FAIL:9170 | NO | FAIL:65129 | DIFF |
| TOP_K | C | 4096 | 3.3 | 5.1 | 35.3 | PASS | PASS | PASS | NO | PASS | DIFF |
| TRI | C | 131072 | 1.9 | 4.8 | 33.2 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TRUNC | C | 131072 | 2.5 | 4.8 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| UPSCALE | E | 131072 | 3.8 | 4.5 | TIMEOUT | PASS | PASS | NODUMP | NODUMP | EXACT | DIFF |
| XIELU | D | 131072 | 2.6 | 4.8 | 33.3 | PASS | FAIL:31 | FAIL:31 | NO | PASS | DIFF |

## 비고

- `cls` = 분류(A/B/C/D/E, TL;DR 참조). `py·ref`/`sp·ref`/`iss·ref` = golden 대비 PASS 또는 `FAIL:<mismatch수>`. `py≡sp`/`py≡iss` = EXACT(byte 동일)/PASS(±1 ULP)/DIFF.
- ISS = HW 레퍼런스. 수정 방침: Class A는 pyspike를 ISS에 맞춤, Class B는 커널 재작성, Class D는 golden 재검증 또는 spike 수정. (2026-05-24 사용자 확정)

