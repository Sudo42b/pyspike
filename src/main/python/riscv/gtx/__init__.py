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
"""GTX NPU subpackage marker.

Phase 1 only exposes the FP16/FP32 conversion helpers (`fp.py`). Other modules
(`memory.py`, `ddr.py`, `params.py`, `encoding.py`) are populated by sibling
plans in this phase. Public re-exports (e.g. `GtxNpu`) land in Phase 2.
"""
