# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pyarrow as pa

import pylibcudf as plc
from rapidsmpf.integrations.cudf.partition import partition_and_pack, unpack_and_concat
from rapidsmpf.memory.buffer_resource import BufferResource


def _dictionary_table() -> plc.Table:
    values = pa.array(
        ["north", "south", "north", "west", "south", "north", "west", "east"]
    ).dictionary_encode()
    keys = pa.array([0, 1, 0, 2, 1, 0, 2, 3], type=pa.int32())
    table = pa.table({"region": values, "key": keys})
    return plc.Table.from_arrow(table)


def test_partition_pack_round_trips_dictionary_column(device_mr, stream) -> None:
    br = BufferResource(device_mr)
    table = _dictionary_table()
    packed = partition_and_pack(
        table=table,
        columns_to_hash=[1],
        num_partitions=4,
        stream=stream,
        br=br,
    )

    out = unpack_and_concat(
        list(packed.values()),
        stream=stream,
        br=br,
    )

    assert out.num_columns() == 2
    assert out.num_rows() == table.num_rows()
    assert (
        out.to_arrow().column("region").type
        == table.to_arrow().column("region").type
    )
