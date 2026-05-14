# GtxNpu FSM(Finite State Machine)기반의 시뮬레이터 완성.

## 소개.

GTX Context RISC-V기반의 Custom0/Custom1 Opcode로 총 4가지 Section에 존재함.
명령어 세부 종류는 다음과 같음: **Source of truth**: `src/main/python/riscv/context_map.yaml` 참고.

| State | 정의 (context_map.yaml) |
|---|---|
| **C1** `PLAN_OUTSIDE` | plan outside — before `START_P` (또는 `END_P` 후 복귀) |
| **C4** `PLAN_INSIDE` | plan inside, shared/thread outside — inside `START_P`, outside `S/T` |
| **C2** `SHARED` | plan inside + shared inside — inside `START_P + START_S` |
| **C3** `THREAD` | plan inside + thread inside — inside `START_P + START_T` |

초기 상태: **C1** (NPU reset 직후)

### INIT
    -> 레지스터 메모리 초기화

### IDLE
    -> cpu에서 명령을 넘어오기 전까지 기다림. 

### DECODE 
    -> custom0/1인지 판단 -> 
         연산종류(mm, conv, scalar, vector, format, activation, DL, MC, SN)
         7bit FN7[25:31] 
                                세부연산.
                                -> 3bit FN7[25:27] 
                                세부 연산(mm인 경우, mm.v, mm, mm.t ... etc)
                                -> FN3(3bit) ... 으로 분리하는데 
        여기서는 한번에 찾음. 
        (fn7, fn3 조합으로 찾는다)

### Dispatch
    -> 각 알맞은 CSR, 함수 적용 전

### Excute 
   -> 실행. 
   
### WriteBack
   -> 정리할거 정리하고, IDLE로 복귀

## 2. 제약사항.
   -> 무조건 pytorch-cuda를 활용하고, 메모리는 최대한 결합하여 silcing이 되도록 함.
   -> uv package manager는 uv를 활용
   -> 궁금하거나 불분명한건 반드시 물어봐야함.
   -> 다수의 Nest및 SPU는 초기화 시, 메모리를 연속되게 만들어 Silcing이 유리하게 되도록 함.
   -> DDR은 CPU, 나머지 메모리 계층은 반드시 cuda로 설정.

## 3. 참고

- `src/main/python/riscv/context_map.yaml` — 9 그룹 + 4 context, 132 instruction 매핑
- `gtx_doxygen/src/intrinsics/1.1.4.1/mainpage.md` — 모드별 사용 가능 인트린식 (cross-ref)
- `gtx-risc-vp/vp/src/platform/gtx/nsu.cpp` — ISS dispatch (vendor 동작 검증용)
- `src/main/python/riscv/gtx/ops/control.py` — `begin_p`/`end_p` 등 warp 마커 핸들러 (현재 위치)
