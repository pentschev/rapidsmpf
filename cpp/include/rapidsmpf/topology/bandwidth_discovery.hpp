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
 * @brief Discover PCIe link properties for a PCI device.
 *
 * Reads `max_link_speed` and `max_link_width` from
 * `/sys/bus/pci/devices/<pci_bus_id>/` and computes the unidirectional
 * bandwidth in GB/s, accounting for PCIe encoding overhead (128b/130b for
 * Gen3+).
 *
 * @param pci_bus_id  PCI bus ID in the form "DDDD:BB:DD.F" (as reported by
 *                    NVML or lspci).
 *
 * @return A `pcie_info` struct.  All fields are zero if the sysfs entries
 *         cannot be read (e.g., running inside a container without sysfs
 *         access).
 */
[[nodiscard]] pcie_info discover_pcie_info(std::string const& pci_bus_id);

/**
 * @brief Discover active NVLink connections for a GPU.
 *
 * Iterates over all possible NVLink lanes on the device (up to the
 * hardware maximum), queries NVML for link state, version, and remote PCI
 * info, then aggregates lanes that connect to the same peer GPU into a
 * single `nvlink_connection` entry.
 *
 * @param gpu_index  NVML device index (matches `gpu_topology_info::id`).
 *
 * @return A vector of `nvlink_connection`, one per unique peer GPU.
 *         Empty if NVML is unavailable, the device has no NVLink, or all
 *         links are inactive.
 *
 * @note NVML must have been initialized before calling this function (e.g.,
 *       via `nvmlInit_v2()`).  The caller is responsible for NVML lifetime.
 */
[[nodiscard]] std::vector<nvlink_connection> discover_nvlink_connections(
    unsigned int gpu_index
);

/**
 * @brief Discover the link speed of a network device.
 *
 * Tries the following sources in order:
 *   1. InfiniBand: `/sys/class/infiniband/<nic_name>/ports/1/rate`
 *   2. Ethernet:   find the associated netdev and read
 *                  `/sys/class/net/<iface>/speed`
 *
 * @param nic_name  Kernel device name as it appears in
 *                  `network_device_info::name` (e.g., "mlx5_0").
 *
 * @return Link speed in Gb/s (e.g., 400.0 for NDR InfiniBand).
 *         Returns 0 if the speed cannot be determined.
 */
[[nodiscard]] double discover_nic_speed(std::string const& nic_name);

}  // namespace rapidsmpf::topology
