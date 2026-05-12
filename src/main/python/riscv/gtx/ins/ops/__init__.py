"""GTX op modules -- handlers registered via @gtx._registry.handler.

Importing each submodule triggers its @handler decorators (Pitfall 6).
Plan 02 fills `spr`; plan 03 fills `control`. Plans 04-05 add `dma`/`mm`/etc.
"""
from . import spr   # noqa: F401  -- triggers SPR @handler decorators
from ...context import control  # noqa: F401  -- triggers warp/control @handler decorators
from ...context import dma   # noqa: F401  -- triggers DMA @handler decorators (Plan 02)
from . import mm   # noqa: F401  -- triggers MM @handler decorators (Plan 04)
from . import vec  # noqa: F401  -- triggers VEC @handler decorators (P5 Plan 02)
from . import act  # noqa: F401  -- triggers ACT @handler decorators (P5 Plans 03 + 04)

__all__ = ["spr", "control", "dma", "mm", "vec", "act"]
