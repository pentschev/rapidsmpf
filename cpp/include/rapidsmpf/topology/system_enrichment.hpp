/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <rapidsmpf/topology/types.hpp>

namespace rapidsmpf::topology {

/**
 * @brief Discover CPU topology for every NUMA node on the system.
 *
 * For each NUMA node in `[0, num_numa_nodes)`:
 *   - Reads the core list from `/sys/devices/system/node/node<N>/cpulist`.
 *   - Picks one core from that list and looks up its `model name` in
 *     `/proc/cpuinfo`.
 *
 * @param num_numa_nodes  Number of NUMA nodes to enumerate (as reported by
 *                        cuCascade or by counting directories under
 *                        `/sys/devices/system/node/`).
 *
 * @return One `cpu_topology_info` per NUMA node, ordered by node ID.
 *         Model name and core count are left empty / zero when the
 *         corresponding sysfs or procfs entries cannot be read.
 */
[[nodiscard]] std::vector<cpu_topology_info> discover_cpus(int num_numa_nodes);

/**
 * @brief Discover the hardware model name of a network interface card.
 *
 * Tries the following sources in order:
 *   1. InfiniBand sysfs: reads
 *      `/sys/class/infiniband/<nic_name>/board_id` and maps known Mellanox
 *      board IDs to human-friendly names (e.g., "ConnectX-7").
 *   2. PCI sysfs: reads `vendor` and `device` from
 *      `/sys/bus/pci/devices/<pci_bus_id>/` and maps known Mellanox / NVIDIA
 *      device IDs to model names.
 *
 * @param nic_name    Kernel device name (e.g., "mlx5_0").
 * @param pci_bus_id  PCI bus ID of the device (e.g., "0000:05:00.0").
 *
 * @return Human-friendly model name, or an empty string if the device
 *         cannot be identified.
 */
[[nodiscard]] std::string discover_nic_model(
    std::string const& nic_name, std::string const& pci_bus_id
);

}  // namespace rapidsmpf::topology
