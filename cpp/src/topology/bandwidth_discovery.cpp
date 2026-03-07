/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>

#include <nvml.h>

#include <rapidsmpf/topology/bandwidth_discovery.hpp>

namespace rapidsmpf::topology {

namespace {

std::string read_sysfs(std::string const& path) {
    std::ifstream f{path};
    if (!f.is_open())
        return {};
    std::string line;
    std::getline(f, line);
    while (!line.empty()
           && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
    {
        line.pop_back();
    }
    return line;
}

std::string normalize_pci_bus_id(std::string const& bus_id) {
    std::string lower = bus_id;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    // cuCascade uses "00000000:06:00.0" (8-char domain); sysfs uses "0000:06:00.0".
    // Normalize to the 4-char domain form used by sysfs.
    if (lower.size() > 12 && lower[4] == '0' && lower[5] == '0' && lower[6] == '0'
        && lower[7] == '0' && lower[8] == ':')
    {
        lower = lower.substr(4);
    }
    return lower;
}

// Map GT/s string to PCIe generation and raw GT/s rate.
struct pcie_speed_entry {
    double gt_per_s;
    int generation;
};

pcie_speed_entry parse_link_speed(std::string const& speed_str) {
    // Format: "16.0 GT/s PCIe" or "16 GT/s"
    double gt = 0;
    std::sscanf(speed_str.c_str(), "%lf", &gt);

    int gen = 0;
    if (gt >= 63.0)
        gen = 6;
    else if (gt >= 31.0)
        gen = 5;
    else if (gt >= 15.0)
        gen = 4;
    else if (gt >= 7.0)
        gen = 3;
    else if (gt >= 4.0)
        gen = 2;
    else if (gt >= 2.0)
        gen = 1;

    return {gt, gen};
}

double compute_pcie_bandwidth(double gt_per_s, int width, int generation) {
    // Gen1/Gen2: 8b/10b encoding → factor = 0.8
    // Gen3+:     128b/130b encoding → factor ≈ 0.9846
    double encoding = (generation <= 2) ? 0.8 : (128.0 / 130.0);
    return gt_per_s * width * encoding / 8.0;
}

constexpr int kMaxNvLinks = 32;

double nvlink_per_link_bandwidth(int version) {
    // Unidirectional GB/s per sub-link
    switch (version) {
    case 1:
        return 20.0;
    case 2:
        return 25.0;
    case 3:
        return 25.0;
    case 4:
        return 25.0;
    case 5:
        return 50.0;
    default:
        return 25.0;
    }
}

}  // namespace

pcie_info discover_pcie_info(std::string const& pci_bus_id) {
    pcie_info info;
    std::string sysfs_id = normalize_pci_bus_id(pci_bus_id);
    std::string base = "/sys/bus/pci/devices/" + sysfs_id + "/";

    std::string speed_str = read_sysfs(base + "max_link_speed");
    std::string width_str = read_sysfs(base + "max_link_width");

    if (speed_str.empty() || width_str.empty())
        return info;

    auto [gt_per_s, gen] = parse_link_speed(speed_str);
    int width = 0;
    try {
        width = std::stoi(width_str);
    } catch (...) {
        return info;
    }

    info.generation = gen;
    info.width = width;
    info.bandwidth_gbps = compute_pcie_bandwidth(gt_per_s, width, gen);
    return info;
}

std::vector<nvlink_connection> discover_nvlink_connections(unsigned int gpu_index) {
    nvmlDevice_t device{};
    if (nvmlDeviceGetHandleByIndex_v2(gpu_index, &device) != NVML_SUCCESS) {
        return {};
    }

    // Collect per-link info keyed by remote PCI bus ID
    struct peer_info {
        std::string pci_bus_id;
        int link_count{0};
        int nvlink_version{0};
    };

    std::map<std::string, peer_info> peers;

    for (int link = 0; link < kMaxNvLinks; ++link) {
        nvmlEnableState_t active{};
        if (nvmlDeviceGetNvLinkState(device, static_cast<unsigned int>(link), &active)
            != NVML_SUCCESS)
        {
            break;
        }
        if (active != NVML_FEATURE_ENABLED)
            continue;

        nvmlPciInfo_t remote_pci{};
        if (nvmlDeviceGetNvLinkRemotePciInfo_v2(
                device, static_cast<unsigned int>(link), &remote_pci
            )
            != NVML_SUCCESS)
        {
            continue;
        }

        unsigned int version = 0;
        nvmlDeviceGetNvLinkVersion(device, static_cast<unsigned int>(link), &version);

        std::string remote_bus{remote_pci.busId};
        auto& p = peers[normalize_pci_bus_id(remote_bus)];
        p.pci_bus_id = remote_bus;
        p.link_count++;
        if (static_cast<int>(version) > p.nvlink_version) {
            p.nvlink_version = static_cast<int>(version);
        }
    }

    // Resolve remote PCI bus IDs to GPU indices
    unsigned int device_count = 0;
    nvmlDeviceGetCount_v2(&device_count);

    std::map<std::string, unsigned int> pci_to_gpu;
    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t dev{};
        if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS)
            continue;
        nvmlPciInfo_t pci{};
        if (nvmlDeviceGetPciInfo_v3(dev, &pci) != NVML_SUCCESS)
            continue;
        pci_to_gpu[normalize_pci_bus_id(std::string{pci.busId})] = i;
    }

    std::vector<nvlink_connection> result;
    for (auto const& [pci_id, info] : peers) {
        nvlink_connection conn;
        auto it = pci_to_gpu.find(pci_id);
        if (it != pci_to_gpu.end()) {
            conn.peer_gpu_id = it->second;
        }
        conn.link_count = info.link_count;
        conn.nvlink_version = info.nvlink_version;
        conn.bandwidth_gbps =
            info.link_count * nvlink_per_link_bandwidth(info.nvlink_version);
        result.push_back(conn);
    }
    return result;
}

namespace {

bool is_pci_bdf(std::string const& name) {
    // Match DDDD:BB:DD.F pattern (e.g., "0000:06:00.0")
    static std::regex const re{
        R"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])"
    };
    return std::regex_match(name, re);
}

std::string basename_of(std::string const& path) {
    auto pos = path.rfind('/');
    if (pos == std::string::npos)
        return path;
    return path.substr(pos + 1);
}

std::string dirname_of(std::string const& path) {
    auto pos = path.rfind('/');
    if (pos == std::string::npos)
        return ".";
    if (pos == 0)
        return "/";
    return path.substr(0, pos);
}

bool is_pci_bridge(std::string const& bdf) {
    std::string cls = read_sysfs("/sys/bus/pci/devices/" + bdf + "/class");
    // PCI-to-PCI bridge class: 0x0604XX
    return cls.size() >= 6 && cls.substr(0, 6) == "0x0604";
}

// Walk sysfs from a PCI endpoint up to find the switch upstream port.
// Returns the BDF of the switch upstream port, or empty if none found.
//
// Typical sysfs path:
//   .../root_port/switch_upstream/switch_downstream/device
// Bridge ancestors (device toward root): [downstream, upstream, root_port]
// The switch upstream port is bridges[1] (second from device).
// If fewer than 2 bridges exist, the device is directly on a root port.
std::string find_parent_switch(std::string const& pci_bus_id) {
    std::string norm = normalize_pci_bus_id(pci_bus_id);
    std::string device_link = "/sys/bus/pci/devices/" + norm;

    char resolved[PATH_MAX];
    if (realpath(device_link.c_str(), resolved) == nullptr)
        return {};

    std::string current{resolved};

    std::vector<std::string> bridges;
    while (true) {
        current = dirname_of(current);
        std::string name = basename_of(current);

        if (!is_pci_bdf(name))
            break;

        if (is_pci_bridge(name)) {
            bridges.push_back(name);
        }
    }

    // bridges[0] = downstream port (closest to device)
    // bridges[1] = upstream port (the switch we want)
    // bridges[last] = root port
    if (bridges.size() >= 2) {
        return bridges[1];
    }
    return {};
}

}  // namespace

std::vector<pcie_switch_info> discover_pcie_switches(
    std::vector<gpu_topology_info> const& gpus,
    std::vector<network_device_info> const& nics
) {
    // Map switch BDF -> (gpu_ids, nic_names)
    std::map<std::string, std::pair<std::vector<unsigned int>, std::vector<std::string>>>
        switch_devices;

    for (auto const& gpu : gpus) {
        if (gpu.pci_bus_id.empty())
            continue;
        std::string sw = find_parent_switch(gpu.pci_bus_id);
        if (!sw.empty()) {
            switch_devices[sw].first.push_back(gpu.id);
        }
    }

    for (auto const& nic : nics) {
        if (nic.pci_bus_id.empty())
            continue;
        std::string sw = find_parent_switch(nic.pci_bus_id);
        if (!sw.empty()) {
            switch_devices[sw].second.push_back(nic.name);
        }
    }

    std::vector<pcie_switch_info> result;
    for (auto& [bdf, devices] : switch_devices) {
        pcie_switch_info info;
        info.pci_bus_id = bdf;
        info.gpu_ids = std::move(devices.first);
        info.nic_names = std::move(devices.second);

        // Read NUMA node
        std::string numa_str = read_sysfs("/sys/bus/pci/devices/" + bdf + "/numa_node");
        if (!numa_str.empty()) {
            try {
                info.numa_node = std::stoi(numa_str);
            } catch (...) {
            }
        }

        // Discover upstream PCIe link bandwidth
        info.pcie = discover_pcie_info(bdf);

        result.push_back(std::move(info));
    }

    return result;
}

double discover_nic_speed(std::string const& nic_name) {
    // Try InfiniBand rate first
    std::string ib_rate =
        read_sysfs("/sys/class/infiniband/" + nic_name + "/ports/1/rate");
    if (!ib_rate.empty()) {
        // Format: "400 Gb/sec" or "200 Gb/sec (4X HDR)"
        double gbps = 0;
        if (std::sscanf(ib_rate.c_str(), "%lf", &gbps) == 1 && gbps > 0) {
            return gbps / 8.0;
        }
    }

    // Try to find associated Ethernet netdev
    std::string net_dir = "/sys/class/infiniband/" + nic_name + "/device/net/";
    // Read the first entry in the net directory by checking common netdev names
    // via the device symlink
    std::string device_path = "/sys/class/infiniband/" + nic_name + "/device";

    // Alternative: look for /sys/class/net/*/device -> same PCI device
    // For now, try reading speed from the IB device's associated net interface
    // by iterating known patterns
    for (int port = 0; port < 2; ++port) {
        std::string speed_str = read_sysfs(
            "/sys/class/infiniband/" + nic_name + "/ports/" + std::to_string(port + 1)
            + "/rate"
        );
        if (!speed_str.empty()) {
            double gbps = 0;
            if (std::sscanf(speed_str.c_str(), "%lf", &gbps) == 1 && gbps > 0) {
                return gbps / 8.0;
            }
        }
    }

    return 0;
}

}  // namespace rapidsmpf::topology
