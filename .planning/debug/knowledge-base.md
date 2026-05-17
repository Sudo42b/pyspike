# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## todo-marked-functions-causing-regressions — gtx extension registration silently broken by 5-defect refactor cascade
- **Date:** 2026-05-18
- **Error patterns:** couldn't find extension 'gtx', libcustomext.so, ImportError, GTX_ISS_F7_RDSPR_ISS, ACT_OPS_REVERSED, torch.uint16, is_reversed assertion, rc=255, act.py:298, refactor casualty
- **Root cause:** Five-defect cascade from refactor cleanup arc (639ddb4..b464bb4, 2026-05-12..05-15). D1: encoding.py:14-15 GTX_F7_WRSPR/RDSPR value swap (0x49/0x48 should be 0x00/0x01 per vendor gtx_npu.h:266-267 firmware-path values; 0x49/0x48 are GTX_ISS_F7_WRSPR_ISS/RDSPR_ISS). D2: encoding.py missing GTX_ISS_F7_RDSPR_ISS/WRSPR_ISS constants → ImportError at spr.py import → riscv/gtx/__init__.py:62-68 silently swallows it as ImportWarning → register_extension never called → rc=255 universal. D3: act.py:30 commented out ACT_OPS_REVERSED import + constant undefined anywhere → would NameError on first GELU/PRELU/TANH/SIGMOID dispatch. D4: spr.py wrong relative import depths (3-dot config_params should be 4-dot; 2-dot csr should be 3-dot — unit/ package layer insertion casualty). D5: _verify.py:43-44 botched np→torch port — torch.uint16(int) is calling a dtype as constructor → TypeError at fp16 compare.
- **Fix:** D1: GTX_F7_WRSPR=0x00, GTX_F7_RDSPR=0x01. D2: re-introduce GTX_ISS_F7_RDSPR_ISS=0x48, GTX_ISS_F7_WRSPR_ISS=0x49; delete dead GTX_ISS_F7_MM/MMC aliases. D3: add ACT_OPS_REVERSED frozenset({PRELU,GELU,TANH,SIGMOID}) per vendor gtx_npu_act.cc:37-42; uncomment act.py:30 import. D4: spr.py imports → 4-dot config_params, 3-dot csr. D5: replace torch.uint16(r_raw).tobytes() with r_raw.to_bytes(2,'little'); replace torch.isnan(float) with NaN-safe x != x. Surgical refactor companion: inline rd_spr/wr_spr from spr_router.py into spr.py, delete spr_router.py; delete control.py P2-skeleton custom0 funct7=0x02..0x07 dispatch stubs (superseded); rename DMA mnemonics to vendor-canonical dot form (load.svr, store.svr, mcast.g2s, mcast.s2l, mcast.s2s, copy.mem); delete _load_svr_l1/_store_svr_l1 alias handlers (0x43/0x45 are aliases of 0x41 funct3=0/1).
- **Files changed:** src/main/python/riscv/gtx/unit/ins/encoding.py, src/main/python/riscv/gtx/unit/ins/ops/act.py, src/main/python/riscv/gtx/unit/ins/ops/spr.py, src/main/python/riscv/gtx/_verify.py, src/main/python/riscv/gtx/unit/context/dma.py, src/main/python/riscv/gtx/unit/context/control.py, src/main/python/riscv/gtx/unit/context/spr_router.py (DELETED)
---

