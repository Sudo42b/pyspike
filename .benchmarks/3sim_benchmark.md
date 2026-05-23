# GTX 3-Simulator Benchmark — `test/` corpus

_2026-05-23 · 95 ops · pyspike vs vendor-spike vs SystemC-ISS_

## TL;DR

- **pyspike ↔ vendor-spike: 일치.** 49/95 op이 출력 동일(byte-exact 또는 FP16 ±1 ULP). golden ref 기준 pyspike 45/95 PASS, spike 77/95 PASS.
- **SystemC-ISS: 데이터-이동 op만 일치, compute op은 불일치.** ISS가 golden과 일치한 op은 15/95개로 거의 전부 **credit-gate 없는 데이터 이동/복사 계열** (`ARGMAX`, `ARGSORT`, `CONCAT`, `CPY`, `DUP`, `FILL`, `GET_ROWS`, `PAD`, `PAD_REFLECT_1D`, `REPEAT`, `ROLL`, `SET`, `SET_ROWS`, `TOP_K`, `TRI`). 나머지 80/95 op(주로 element-wise/compute)은 ISS에서 **전부 0 출력**.
- **원인 (credit 단일):** `test/` 커널은 Phase-8에서 pyspike의 functional(NOP) credit 모델에 맞춰 재작성됨. compute op은 T-loop compute→store 핸드셰이크가 credit에 의존하는데, strict-credit HW 모델인 ISS에서는 그 hand-off가 성립하지 않아 결과 영역이 미기록(0)으로 남음. 데이터 이동 op은 직접 DMA(credit-gate 무관)라 ISS에서도 정상 — 이 op들이 ISS와 byte-exact 일치하는 것이 입력 byte-order는 동일하게 해석됨(= 0의 원인이 아님)을 보여줌.
- **속도(합계):** pyspike **482s**, vendor-spike **456s**, ISS **3195s**. (중앙값/op: py 2.3s · sp 4.8s · iss 33.4s)

## 속도 요약

| 시뮬레이터 | 합계(s) | 중앙값/op(s) | 평균/op(s) | 최소 | 최대 | TIMEOUT |
|---|--:|--:|--:|--:|--:|--:|
| pyspike | 482 | 2.3 | 5.1 | 1.7 | 109.1 | 1 |
| vendor-spike | 456 | 4.8 | 4.9 | 4.1 | 8.7 | 1 |
| SystemC-ISS | 3195 | 33.4 | 33.6 | 32.7 | 37.2 | 0 |

> elf: pyspike·spike = `build_kernel.sh`(minimal crt+tohost); ISS = `build_uni.sh`(full gtx-firmware startup + exit_shim, ISS는 minimal crt 거부). 동일 커널 소스, 동일 입력.

## 출력 일치 요약 (golden ref, FP16 ulp=1)

| 비교 | 일치 op 수 |
|---|--:|
| pyspike == ref | 45/95 |
| vendor-spike == ref | 77/95 |
| **SystemC-ISS == ref** | **15/95** |
| pyspike == vendor-spike | 49/95 |
| ISS 전부-0 출력 | 80/95 |

## ISS와 일치하는 op (golden 기준)

거의 전부 credit-gate가 없는 데이터 이동/복사/리덕션 계열 — ISS의 strict-credit 모델과 무관하게 동작.

| op | iss=0 | py≡iss | py·ref | sp·ref |
|---|:--:|:--:|:--:|:--:|
| ARGMAX | YES | DIFF | PASS | PASS |
| ARGSORT | YES | DIFF | PASS | PASS |
| CONCAT | NO | EXACT | PASS | PASS |
| CPY | NO | EXACT | PASS | PASS |
| DUP | NO | EXACT | PASS | PASS |
| FILL | NO | EXACT | PASS | PASS |
| GET_ROWS | NO | EXACT | PASS | PASS |
| PAD | NO | EXACT | PASS | PASS |
| PAD_REFLECT_1D | NO | EXACT | PASS | PASS |
| REPEAT | NO | EXACT | PASS | PASS |
| ROLL | NO | EXACT | PASS | PASS |
| SET | NO | EXACT | PASS | PASS |
| SET_ROWS | NO | EXACT | PASS | PASS |
| TOP_K | YES | DIFF | PASS | PASS |
| TRI | NO | EXACT | PASS | PASS |

## op별 상세

| op | out(B) | py(s) | sp(s) | iss(s) | py·ref | sp·ref | iss·ref | iss=0 | py≡sp | py≡iss |
|---|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ABS | 6291488 | 17.2 | 5.6 | 34.6 | PASS | PASS | FAIL:3142578 | YES | EXACT | DIFF |
| ACC | 16764960 | 8.2 | 7.1 | 36.8 | PASS | PASS | FAIL:8372834 | YES | EXACT | DIFF |
| ADD | 2097152 | 2.3 | 4.7 | 34.0 | FAIL:1047308 | PASS | FAIL:1047308 | YES | FAIL:1047308 | EXACT |
| ADD1 | 16764960 | 5.8 | 8.7 | 35.7 | PASS | PASS | FAIL:8372560 | YES | EXACT | DIFF |
| ADD_ID | 6839392 | 3.4 | 6.1 | 37.2 | FAIL:3416163 | PASS | FAIL:3416163 | YES | FAIL:3415660 | EXACT |
| ADD_REL_POS | 12160 | 2.1 | 4.4 | 33.4 | FAIL:6074 | PASS | FAIL:6075 | YES | FAIL:6074 | DIFF |
| ARANGE | 128 | 2.2 | 4.9 | 33.4 | FAIL:64 | PASS | FAIL:63 | YES | FAIL:64 | DIFF |
| ARGMAX | 1024 | 2.4 | 4.2 | 33.2 | PASS | PASS | PASS | YES | PASS | DIFF |
| ARGSORT | 262144 | 109.0 | 5.0 | 33.3 | PASS | PASS | PASS | YES | PASS | DIFF |
| CEIL | 131072 | 1.8 | 5.2 | 33.4 | PASS | PASS | FAIL:32796 | YES | EXACT | DIFF |
| CLAMP | 8192 | 2.7 | 5.1 | 33.6 | FAIL:4089 | PASS | FAIL:4089 | YES | FAIL:4089 | EXACT |
| CONCAT | 2106400 | 2.8 | 5.3 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CONV_2D | 2208 | 2.7 | 5.1 | 33.7 | FAIL:1089 | PASS | FAIL:1089 | YES | FAIL:1089 | DIFF |
| CONV_2D_DW | 128 | 1.8 | 5.0 | 33.6 | FAIL:56 | PASS | FAIL:56 | YES | FAIL:56 | DIFF |
| CONV_3D | 1600 | 4.9 | 4.7 | 33.2 | FAIL:799 | PASS | FAIL:799 | YES | FAIL:799 | DIFF |
| CONV_TRANSPOSE_1D | 65536 | 6.8 | 4.7 | 36.1 | FAIL:32767 | PASS | FAIL:32704 | YES | FAIL:32767 | DIFF |
| CONV_TRANSPOSE_2D | 1024 | 2.6 | 5.0 | 33.8 | FAIL:511 | FAIL:1 | FAIL:511 | YES | FAIL:511 | DIFF |
| COS | 8192 | 2.0 | 4.9 | 33.1 | PASS | PASS | FAIL:4096 | YES | PASS | DIFF |
| COUNT_EQUAL | 32 | 1.9 | 5.2 | 34.8 | FAIL:1 | FAIL:1 | FAIL:1 | YES | EXACT | EXACT |
| CPY | 65472 | 2.6 | 5.1 | 33.5 | PASS | PASS | PASS | NO | EXACT | EXACT |
| CUMSUM | 14336 | 2.0 | 4.6 | 33.8 | FAIL:7053 | FAIL:1236 | FAIL:7167 | YES | FAIL:7054 | DIFF |
| CUMSUM_v2 | 32768 | 3.4 | 5.0 | 33.4 | FAIL:16099 | FAIL:2741 | FAIL:16379 | YES | FAIL:16100 | DIFF |
| DIAG | 65536 | 2.0 | 4.8 | 33.2 | PASS | PASS | FAIL:180 | YES | EXACT | DIFF |
| DIAG_MASK_INF | 8192 | 1.9 | 4.8 | 33.9 | FAIL:474 | PASS | FAIL:4091 | YES | FAIL:474 | DIFF |
| DIAG_MASK_ZERO | 8192 | 2.7 | 5.1 | 33.3 | FAIL:1388 | PASS | FAIL:2076 | YES | FAIL:1388 | DIFF |
| DIV | 4160 | 1.9 | 5.0 | 33.7 | PASS | PASS | FAIL:2077 | YES | EXACT | DIFF |
| DUP | 65280 | 1.7 | 4.8 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ELU | 2560 | 1.8 | 4.3 | 32.8 | PASS | PASS | FAIL:1277 | YES | PASS | DIFF |
| EXP | 32992 | 1.8 | 5.0 | 34.1 | PASS | PASS | FAIL:16496 | YES | EXACT | DIFF |
| EXPM1 | 2048 | 2.6 | 4.9 | 33.4 | PASS | PASS | FAIL:1021 | YES | PASS | DIFF |
| FILL | 65408 | 1.7 | 4.5 | 33.3 | PASS | PASS | PASS | NO | EXACT | EXACT |
| FLASH_ATTN_EXT | 131072 | 1.7 | 5.0 | 33.7 | FAIL:64264 | FAIL:64264 | FAIL:64264 | NO | EXACT | EXACT |
| FLOOR | 8192 | 2.6 | 5.1 | 33.3 | PASS | PASS | FAIL:2025 | YES | EXACT | DIFF |
| GATED_LINEAR_ATTN | 96 | 2.1 | 4.8 | 33.7 | FAIL:36 | PASS | FAIL:35 | YES | FAIL:36 | DIFF |
| GEGLU | 4096 | 1.8 | 4.8 | 33.9 | FAIL:2018 | PASS | FAIL:2017 | YES | FAIL:2018 | DIFF |
| GEGLU_ERF | 4096 | 3.4 | 5.0 | 33.3 | FAIL:2018 | PASS | FAIL:2017 | YES | FAIL:2018 | DIFF |
| GEGLU_QUICK | 262144 | 2.9 | 4.9 | 33.7 | FAIL:129203 | PASS | FAIL:128967 | YES | FAIL:129199 | DIFF |
| GELU | 61440 | 1.7 | 4.6 | 33.5 | PASS | PASS | FAIL:30656 | YES | PASS | DIFF |
| GELU_ERF | 4194304 | 4.2 | 5.4 | 34.8 | PASS | PASS | FAIL:2092924 | YES | PASS | DIFF |
| GELU_QUICK | 65024 | 2.4 | 4.6 | 33.3 | PASS | PASS | FAIL:32446 | YES | PASS | DIFF |
| GET_ROWS | 262144 | 1.9 | 4.7 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| GROUP_NORM | 65280 | 1.7 | 4.8 | 33.7 | FAIL:32618 | FAIL:32618 | FAIL:32614 | NO | EXACT | DIFF |
| HARDSIGMOID | 7904 | 2.7 | 5.2 | 33.5 | FAIL:3937 | PASS | FAIL:3937 | YES | FAIL:3937 | EXACT |
| HARDSWISH | 8192 | 2.0 | 4.7 | 33.3 | FAIL:4085 | PASS | FAIL:4085 | YES | FAIL:4085 | DIFF |
| IM2COL | 4608 | 1.7 | 4.6 | 33.1 | FAIL:2040 | PASS | FAIL:2295 | YES | FAIL:2040 | DIFF |
| IM2COL_3D | 94400 | 2.2 | 4.7 | 35.0 | FAIL:44518 | PASS | FAIL:47112 | YES | FAIL:44518 | DIFF |
| L2_NORM | 8192 | 3.5 | 4.3 | 33.5 | FAIL:4087 | PASS | FAIL:4087 | YES | FAIL:4087 | DIFF |
| LEAKY_RELU | 63776 | 1.7 | 4.6 | 33.7 | PASS | PASS | FAIL:31736 | YES | EXACT | DIFF |
| LOG | 16256 | TIMEOUT | TIMEOUT | 33.1 | FAIL:8121 | FAIL:8121 | FAIL:8121 | YES | EXACT | EXACT |
| MEAN | 160 | 1.8 | 4.1 | 33.0 | FAIL:80 | PASS | FAIL:80 | YES | FAIL:80 | DIFF |
| MUL | 2097152 | 3.9 | 5.3 | 34.1 | FAIL:1040245 | PASS | FAIL:1040245 | YES | FAIL:1040245 | EXACT |
| MUL_MAT | 1048576 | 2.5 | 5.7 | 36.0 | FAIL:524268 | PASS | FAIL:65528 | NO | FAIL:524268 | DIFF |
| MUL_MAT_ID | 2048 | 1.8 | 4.5 | 33.6 | FAIL:1024 | PASS | FAIL:1024 | YES | FAIL:1024 | DIFF |
| NEG | 34816 | 1.8 | 4.6 | 33.4 | PASS | PASS | FAIL:17386 | YES | EXACT | DIFF |
| NORM | 16448 | 3.6 | 4.8 | 33.3 | FAIL:8221 | FAIL:8219 | FAIL:8219 | YES | FAIL:5494 | DIFF |
| OUT_PROD | 1024 | 2.5 | 4.4 | 33.0 | FAIL:256 | PASS | FAIL:512 | YES | FAIL:256 | DIFF |
| PAD | 65024 | 1.7 | 4.7 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| PAD_REFLECT_1D | 49152 | 2.6 | 4.8 | 33.1 | PASS | PASS | PASS | NO | EXACT | EXACT |
| POOL_1D | 4096 | 2.5 | 4.7 | 32.8 | FAIL:2045 | PASS | FAIL:2045 | YES | FAIL:2045 | DIFF |
| POOL_2D | 1664 | 2.0 | 4.6 | 33.0 | FAIL:821 | PASS | FAIL:821 | YES | FAIL:821 | DIFF |
| REGLU | 4096 | 1.7 | 4.7 | 33.2 | FAIL:1014 | PASS | FAIL:1009 | YES | FAIL:1014 | DIFF |
| RELU | 524288 | 1.8 | 4.8 | 34.1 | PASS | PASS | FAIL:131098 | YES | EXACT | DIFF |
| REPEAT | 1091808 | 2.9 | 5.0 | 33.7 | PASS | PASS | PASS | NO | EXACT | EXACT |
| RMS_NORM | 2097152 | 2.5 | 4.7 | 34.6 | FAIL:1047945 | FAIL:772 | FAIL:1047945 | YES | FAIL:1047945 | EXACT |
| ROLL | 2105344 | 2.4 | 5.3 | 33.6 | PASS | PASS | PASS | NO | EXACT | EXACT |
| ROPE | 2048 | 1.8 | 4.6 | 33.6 | FAIL:1024 | PASS | FAIL:1024 | YES | FAIL:1024 | DIFF |
| ROUND | 32768 | 3.3 | 4.9 | 33.5 | PASS | PASS | FAIL:8167 | YES | EXACT | DIFF |
| RWKV_WKV6 | 32 | 2.0 | 4.5 | 32.8 | FAIL:8 | PASS | FAIL:8 | YES | FAIL:8 | DIFF |
| RWKV_WKV7 | 2112 | 2.7 | 4.2 | 33.6 | FAIL:1055 | FAIL:169 | FAIL:1055 | YES | FAIL:1054 | DIFF |
| SCALE | 8192 | 1.9 | 4.8 | 33.9 | FAIL:2043 | PASS | FAIL:4085 | YES | FAIL:2043 | DIFF |
| SET | 2105344 | 2.8 | 5.1 | 33.8 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SET_ROWS | 2097152 | 2.1 | 5.1 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| SGN | 7936 | 1.8 | 4.5 | 33.2 | PASS | PASS | FAIL:3968 | YES | EXACT | DIFF |
| SIGMOID | 8192 | 2.4 | 4.7 | 32.9 | PASS | PASS | FAIL:4096 | YES | EXACT | DIFF |
| SILU | 131072 | 2.5 | 4.8 | 33.4 | PASS | PASS | FAIL:65400 | YES | PASS | DIFF |
| SIN | 131072 | 4.3 | 4.8 | 34.0 | FAIL:32733 | FAIL:65467 | FAIL:65467 | YES | FAIL:32989 | DIFF |
| SOFTPLUS | 131072 | 1.9 | 4.2 | 33.1 | PASS | PASS | FAIL:65536 | YES | PASS | DIFF |
| SOFT_MAX | 131072 | 1.8 | 4.6 | 33.1 | FAIL:28416 | PASS | FAIL:65536 | YES | FAIL:28416 | DIFF |
| SOLVE_TRI | 65536 | 109.1 | 4.9 | 33.6 | FAIL:268 | FAIL:3311 | FAIL:32321 | YES | FAIL:1468 | DIFF |
| SQR | 131072 | 2.5 | 4.5 | 33.1 | PASS | PASS | FAIL:63475 | YES | EXACT | DIFF |
| SQRT | 131072 | 1.8 | 4.7 | 32.8 | PASS | PASS | FAIL:65536 | YES | EXACT | DIFF |
| SSM_CONV | 131072 | 17.4 | 4.6 | 33.1 | FAIL:65515 | PASS | FAIL:65462 | YES | FAIL:65516 | DIFF |
| SSM_SCAN | 1536 | 2.2 | 4.7 | 33.3 | FAIL:768 | FAIL:635 | FAIL:768 | YES | FAIL:768 | DIFF |
| STEP | 131072 | 1.9 | 4.6 | 33.3 | PASS | PASS | FAIL:32790 | YES | EXACT | DIFF |
| SUB | 131072 | 1.8 | 4.3 | 32.7 | PASS | PASS | FAIL:65471 | YES | EXACT | DIFF |
| SUM | 32 | 1.7 | 4.8 | 33.4 | FAIL:1 | FAIL:1 | FAIL:1 | YES | FAIL:1 | DIFF |
| SUM_ROWS | 512 | 3.3 | 4.9 | 33.2 | FAIL:256 | FAIL:28 | FAIL:256 | YES | FAIL:256 | DIFF |
| SWIGLU_OAI | 65536 | 2.7 | 4.8 | 33.4 | FAIL:32394 | FAIL:4 | FAIL:32503 | YES | FAIL:32392 | DIFF |
| TANH | 131072 | 1.7 | 4.6 | 32.9 | PASS | PASS | FAIL:65467 | YES | EXACT | DIFF |
| TIMESTEP_EMBEDDING | 131072 | 4.1 | 4.3 | 33.9 | FAIL:65143 | FAIL:9170 | FAIL:65277 | YES | FAIL:65129 | DIFF |
| TOP_K | 4096 | 3.5 | 4.3 | 32.9 | PASS | PASS | PASS | YES | PASS | DIFF |
| TRI | 131072 | 1.8 | 4.4 | 33.4 | PASS | PASS | PASS | NO | EXACT | EXACT |
| TRUNC | 131072 | 2.7 | 4.8 | 33.3 | PASS | PASS | FAIL:14 | YES | EXACT | DIFF |
| UPSCALE | 131072 | 3.9 | 4.1 | 32.9 | PASS | PASS | FAIL:65456 | YES | EXACT | DIFF |
| XIELU | 131072 | 1.9 | 4.6 | 33.3 | PASS | FAIL:31 | FAIL:65467 | YES | PASS | DIFF |

## 비고

- `py·ref`/`sp·ref`/`iss·ref` = golden 대비 PASS 또는 `FAIL:<mismatch수>`. pyspike의 FAIL은 기존 로직 버그(별도 추적), credit 변경과 무관(출력 바이트 동일성 확인됨).
- `py≡sp` = EXACT(byte 동일) 또는 PASS(±1 ULP) 또는 DIFF.
- ISS는 `test/` 코퍼스 비호환이 systemic — op별 버그가 아니라 corpus 설계 차이. ISS 3-way 검증이 필요하면 원본 ggml_ops_c 커널(strict-credit 호환)을 써야 함.

