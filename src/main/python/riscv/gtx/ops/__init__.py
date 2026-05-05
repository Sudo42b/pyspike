#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""GTX op modules -- handlers registered via @gtx._registry.handler.

Importing each submodule triggers its @handler decorators (Pitfall 6).
Plan 02 fills `spr`; plan 03 fills `control`. Plans 04-05 add `dma`/`mm`/etc.
"""
from . import spr   # noqa: F401  -- triggers SPR @handler decorators
from . import control  # noqa: F401  -- triggers warp/control @handler decorators
from . import dma   # noqa: F401  -- triggers DMA @handler decorators (Plan 02)

__all__ = ["spr", "control", "dma"]
