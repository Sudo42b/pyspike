"""GTX op modules — handlers registered via :func:`gtx._registry.handler`.

Importing each submodule triggers its ``@handler`` decorators (Pitfall 6).
"""
from . import spr      # noqa: F401  -- SPR @handler decorators
from ...context import control  # noqa: F401  -- warp / control handlers
from ...context import dma      # noqa: F401  -- DMA handlers
from . import mm       # noqa: F401  -- MM handlers
from . import vec      # noqa: F401  -- VEC handlers
from . import act      # noqa: F401  -- ACT handlers

__all__ = ["spr", "control", "dma", "mm", "vec", "act"]
