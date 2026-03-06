/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include <rapidsmpf/topology/system_enrichment.hpp>

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

std::string trim(std::string s) {
    auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

int count_cpus_in_list(std::string const& cpulist) {
    if (cpulist.empty())
        return 0;
    int count = 0;
    std::istringstream stream{cpulist};
    std::string token;
    while (std::getline(stream, token, ',')) {
        auto dash = token.find('-');
        if (dash != std::string::npos) {
            int lo = std::stoi(token.substr(0, dash));
            int hi = std::stoi(token.substr(dash + 1));
            count += hi - lo + 1;
        } else {
            count += 1;
        }
    }
    return count;
}

int first_cpu_in_list(std::string const& cpulist) {
    if (cpulist.empty())
        return -1;
    auto dash = cpulist.find('-');
    auto comma = cpulist.find(',');
    std::string first = cpulist.substr(0, std::min(dash, comma));
    try {
        return std::stoi(first);
    } catch (...) {
        return -1;
    }
}

std::string normalize_pci_bus_id(std::string const& bus_id) {
    std::string lower = bus_id;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (lower.size() > 12 && lower[4] == '0' && lower[5] == '0' && lower[6] == '0'
        && lower[7] == '0' && lower[8] == ':')
    {
        lower = lower.substr(4);
    }
    return lower;
}

struct known_device {
    unsigned int device_id;
    char const* name;
};

// Mellanox/NVIDIA ConnectX device IDs (from PCI ID database)
constexpr known_device kMellanoxDevices[] = {
    {0x101b, "ConnectX-6 Dx"},
    {0x101d, "ConnectX-6 Dx"},
    {0x101f, "ConnectX-6 Lx"},
    {0x1021, "ConnectX-7"},
    {0x1023, "ConnectX-7"},
    {0xa2d6, "ConnectX-4"},
    {0xa2dc, "ConnectX-4"},
    {0x1013, "ConnectX-5"},
    {0x1015, "ConnectX-5"},
    {0x1017, "ConnectX-5 Ex"},
    {0x1019, "ConnectX-5 Ex"},
    {0x101b, "ConnectX-6"},
};

// Mellanox vendor IDs
constexpr unsigned int kMellanoxVendor = 0x15b3;

}  // namespace

std::vector<cpu_topology_info> discover_cpus(int num_numa_nodes) {
    // Parse /proc/cpuinfo to build a map of core ID → model name
    std::map<int, std::string> core_model;
    {
        std::ifstream cpuinfo{"/proc/cpuinfo"};
        if (cpuinfo.is_open()) {
            std::string line;
            int current_processor = -1;
            while (std::getline(cpuinfo, line)) {
                if (line.rfind("processor", 0) == 0) {
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        current_processor = std::stoi(trim(line.substr(colon + 1)));
                    }
                } else if (line.rfind("model name", 0) == 0 && current_processor >= 0) {
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        core_model[current_processor] = trim(line.substr(colon + 1));
                    }
                }
            }
        }
    }

    std::vector<cpu_topology_info> result;
    for (int node = 0; node < num_numa_nodes; ++node) {
        cpu_topology_info cpu;
        cpu.numa_node = node;

        std::string cpulist = read_sysfs(
            "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"
        );
        cpu.cpu_affinity_list = cpulist;
        cpu.core_count = count_cpus_in_list(cpulist);

        int first = first_cpu_in_list(cpulist);
        if (first >= 0) {
            auto it = core_model.find(first);
            if (it != core_model.end()) {
                cpu.model_name = it->second;
            }
        }
        result.push_back(std::move(cpu));
    }
    return result;
}

std::string discover_nic_model(
    std::string const& nic_name, std::string const& pci_bus_id
) {
    // Try InfiniBand board_id
    std::string board_id = read_sysfs("/sys/class/infiniband/" + nic_name + "/board_id");
    if (!board_id.empty()) {
        // board_id often encodes the device type, e.g., "MT_2180110032"
        // Map known Mellanox board ID prefixes
        if (board_id.find("MT_0000000") != std::string::npos
            || board_id.find("MT_") != std::string::npos)
        {
            // Fall through to PCI-based detection for more accuracy
        }
    }

    // Try PCI vendor/device IDs
    std::string sysfs_id = normalize_pci_bus_id(pci_bus_id);
    std::string base = "/sys/bus/pci/devices/" + sysfs_id + "/";

    std::string vendor_str = read_sysfs(base + "vendor");
    std::string device_str = read_sysfs(base + "device");

    if (!vendor_str.empty() && !device_str.empty()) {
        unsigned int vendor = 0;
        unsigned int device = 0;
        std::sscanf(vendor_str.c_str(), "%x", &vendor);
        std::sscanf(device_str.c_str(), "%x", &device);

        if (vendor == kMellanoxVendor) {
            for (auto const& known : kMellanoxDevices) {
                if (known.device_id == device) {
                    return known.name;
                }
            }
            return "Mellanox NIC";
        }
    }

    // Try the InfiniBand hca_type
    std::string hca_type = read_sysfs("/sys/class/infiniband/" + nic_name + "/hca_type");
    if (!hca_type.empty()) {
        return hca_type;
    }

    return {};
}

}  // namespace rapidsmpf::topology
