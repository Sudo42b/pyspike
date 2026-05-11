from typing import Optional

import torch

from .params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
)

from . import DEVICE

class GtxMemory:
    """GTX NPU memory layer — L0/L1/L2 torch tensor + DDR lazy alloc + SPR dict."""
    def __init__(self) -> None:
        # byte arrays for L0/L1/L2. Indexed by (nest, spu) for L0/L1 and by (nest) for L2.
        self._l0_bytes: bytearray = bytearray(GTX_NEST_NUM * GTX_SPU_NUM * GTX_L0_SIZE_BYTES)
        self._l1_bytes: bytearray = bytearray(GTX_NEST_NUM * GTX_SPU_NUM * GTX_L1_SIZE_BYTES)
        self._l2_bytes: bytearray = bytearray(GTX_NEST_NUM * GTX_L2_SIZE_BYTES)
        # spr 총 32개.
        self.spr: dict[int, int] = {}
        self._ddr_bytes: Optional[bytearray] = None

    def l0_byte(self, nest: int, spu: int) -> bytearray:
        start = (nest * GTX_SPU_NUM + spu) * GTX_L0_SIZE_BYTES
        end = start + GTX_L0_SIZE_BYTES
        return self._l0_bytes[start:end]

    def l1_byte(self, nest: int, spu: int) -> bytearray:
        start = (nest * GTX_SPU_NUM + spu) * GTX_L1_SIZE_BYTES
        end = start + GTX_L1_SIZE_BYTES
        return self._l1_bytes[start:end]

    def l2_byte(self, nest: int) -> bytearray:
        start = nest * GTX_L2_SIZE_BYTES
        end = start + GTX_L2_SIZE_BYTES
        return self._l2_bytes[start:end]

    def l0_f16(self, nest: int, spu: int) -> torch.Tensor:
        # using l0_bytes 
        byte = self.l0_byte(nest, spu)
        return torch.tensor(byte).to(torch.float16).to(DEVICE)

    def l1_f16(self, nest: int, spu: int) -> torch.Tensor:
        byte = self.l1_byte(nest, spu)
        return torch.tensor(byte).to(torch.float16).to(DEVICE)

    def l2_f16(self, nest: int) -> torch.Tensor:
        byte = self.l2_byte(nest)
        return torch.tensor(byte).to(torch.float16).to(DEVICE)