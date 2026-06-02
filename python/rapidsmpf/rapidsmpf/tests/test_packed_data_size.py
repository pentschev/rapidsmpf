# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData

if TYPE_CHECKING:
    import rmm


def test_packed_data_data_size_reports_payload_bytes(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr)
    packed = PackedData.from_host_bytes(b"abcdef", br)
    assert packed.data_size() == 6
