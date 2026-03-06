/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rapidsmpf::topology {

/**
 * @brief PCIe link properties for a device.
 *
 * Captures the negotiated (or maximum) PCIe generation, lane width, and the
 * resulting unidirectional bandwidth.  Discovered from sysfs
 * (`/sys/bus/pci/devices/<bus_id>/max_link_speed` and `max_link_width`).
 */
struct pcie_info {
    int generation{0};  ///< PCIe generation (e.g., 4, 5, 6). 0 = unknown.
    int width{0};  ///< Lane count (e.g., 16). 0 = unknown.
    double bandwidth_gbps{0};  ///< Unidirectional bandwidth in GB/s (e.g., ~31.5 for
                               ///< Gen4 x16). 0 = unknown.
};

/**
 * @brief A single NVLink connection from one GPU to a peer GPU.
 *
 * Represents the aggregate of all NVLink lanes between a GPU pair.
 * Discovered via NVML (`nvmlDeviceGetNvLinkState`, `nvmlDeviceGetNvLinkVersion`,
 * `nvmlDeviceGetNvLinkRemotePciInfo_v2`).
 */
struct nvlink_connection {
    unsigned int peer_gpu_id{0};  ///< Device index of the peer GPU.
    int link_count{0};  ///< Number of active NVLink lanes to this peer.
    int nvlink_version{0};  ///< NVLink version (e.g., 3, 4, 5). 0 = unknown.
    double bandwidth_gbps{0};  ///< Aggregate unidirectional bandwidth in GB/s
                               ///< (sum of all links to this peer, one direction;
                               ///< e.g., 25 GB/s/link for NVLink 3/4). 0 = unknown.
};

/**
 * @brief CPU information for a NUMA node.
 *
 * One entry per NUMA node.  Model name is parsed from `/proc/cpuinfo` and
 * the core list from `/sys/devices/system/node/node<N>/cpulist`.
 */
struct cpu_topology_info {
    int numa_node{-1};  ///< NUMA node this CPU belongs to. -1 = unknown.
    std::string model_name;  ///< e.g., "AMD EPYC 9654 96-Core Processor"
    int core_count{0};  ///< Physical core count on this NUMA node.
    std::string cpu_affinity_list;  ///< Kernel-format CPU list, e.g. "0-47,96-143".
};

/**
 * @brief Extended GPU topology information.
 *
 * Superset of `cucascade::memory::gpu_topology_info` — every field from
 * cuCascade is present, plus PCIe and NVLink details.
 */
struct gpu_topology_info {
    unsigned int id{0};  ///< NVML device index.
    std::string name;  ///< NVML device name (e.g., "NVIDIA H100 80GB HBM3").
    std::string pci_bus_id;  ///< Domain:Bus:Device.Function, e.g. "00000000:06:00.0".
    std::string uuid;  ///< GPU UUID string.
    int numa_node{-1};  ///< NUMA node this GPU is attached to. -1 = unknown.
    std::string cpu_affinity_list;  ///< Kernel-format CPU list for this GPU's affinity.
    std::vector<int> cpu_cores;  ///< Individual core IDs.
    std::vector<int> memory_binding;  ///< NUMA nodes for memory binding.
    std::vector<std::string> network_devices;  ///< NIC names local to this GPU.

    pcie_info pcie;  ///< PCIe link to the CPU / root complex.
    std::vector<nvlink_connection> nvlink_peers;  ///< NVLink connections to peer GPUs.
};

/**
 * @brief Extended network device information.
 *
 * Superset of `cucascade::memory::network_device_info` — adds link speed
 * and hardware model name.
 */
struct network_device_info {
    std::string name;  ///< Kernel device name (e.g., "mlx5_0").
    int numa_node{-1};  ///< NUMA node this device is attached to. -1 = unknown.
    std::string pci_bus_id;  ///< PCI bus ID (e.g., "0000:05:00.0").

    std::string model_name;  ///< Hardware model (e.g., "ConnectX-7"). Empty if unknown.
    double bandwidth_gbps{0};  ///< Unidirectional bandwidth in GB/s (e.g., 50.0 for
                               ///< a 400 Gb/s link: 400 / 8). 0 = unknown.
    pcie_info pcie;  ///< PCIe link to the CPU / root complex.
};

/**
 * @brief Complete enriched system topology.
 *
 * Produced by `topology_viz::discover()` or loaded from JSON via
 * `topology_viz::load_json()`.  All "new" fields (bandwidth, model names)
 * default to zero / empty so that partially-populated JSON files are valid.
 */
struct system_topology {
    std::string hostname;  ///< System hostname.
    unsigned int num_gpus{0};  ///< Total number of GPUs.
    int num_numa_nodes{0};  ///< Total number of NUMA nodes.
    int num_network_devices{0};  ///< Total number of network devices.

    std::vector<cpu_topology_info> cpus;  ///< Per-NUMA-node CPU information.
    std::vector<gpu_topology_info> gpus;  ///< Per-GPU topology information.
    std::vector<network_device_info> network_devices;  ///< Per-NIC information.
};

}  // namespace rapidsmpf::topology
