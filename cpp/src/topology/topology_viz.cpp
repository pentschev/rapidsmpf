/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvml.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/topology/bandwidth_discovery.hpp>
#include <rapidsmpf/topology/json_io.hpp>
#include <rapidsmpf/topology/system_enrichment.hpp>
#include <rapidsmpf/topology/topology_viz.hpp>

namespace rapidsmpf::topology {

namespace {

std::string read_sysfs_line(std::filesystem::path const& path) {
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

int read_numa_node(std::filesystem::path const& device_path) {
    std::string numa = read_sysfs_line(device_path / "numa_node");
    if (numa.empty())
        return -1;
    try {
        return std::stoi(numa);
    } catch (...) {
        return -1;
    }
}

bool has_network_device(
    system_topology const& topo, std::string const& name, std::string const& pci_bus_id
) {
    return std::any_of(
        topo.network_devices.begin(),
        topo.network_devices.end(),
        [&](network_device_info const& dev) {
            return dev.name == name
                   || (!pci_bus_id.empty() && dev.pci_bus_id == pci_bus_id);
        }
    );
}

void add_infiniband_devices(system_topology& topo) {
    namespace fs = std::filesystem;
    fs::path const ib_dir{"/sys/class/infiniband"};
    std::error_code ec;
    if (!fs::exists(ib_dir, ec) || !fs::is_directory(ib_dir, ec)) {
        return;
    }

    std::vector<fs::path> devices;
    for (auto const& entry : fs::directory_iterator{ib_dir, ec}) {
        if (ec)
            break;
        devices.push_back(entry.path());
    }
    std::sort(devices.begin(), devices.end());

    for (auto const& path : devices) {
        std::string name = path.filename().string();
        if (name.empty())
            continue;

        fs::path device_path = path / "device";
        fs::path resolved = fs::canonical(device_path, ec);
        if (ec) {
            ec.clear();
            continue;
        }

        std::string pci_bus_id = resolved.filename().string();
        if (has_network_device(topo, name, pci_bus_id))
            continue;

        network_device_info dev;
        dev.name = name;
        dev.pci_bus_id = pci_bus_id;
        dev.numa_node = read_numa_node(device_path);
        topo.network_devices.push_back(std::move(dev));
    }

    topo.num_network_devices = static_cast<int>(topo.network_devices.size());
}

system_topology convert_from_cucascade(
    cucascade::memory::system_topology_info const& base
) {
    system_topology topo;
    topo.hostname = base.hostname;
    topo.num_gpus = base.num_gpus;
    topo.num_numa_nodes = base.num_numa_nodes;
    topo.num_network_devices = base.num_network_devices;

    for (auto const& g : base.gpus) {
        gpu_topology_info gpu;
        gpu.id = g.id;
        gpu.name = g.name;
        gpu.pci_bus_id = g.pci_bus_id;
        gpu.uuid = g.uuid;
        gpu.numa_node = g.numa_node;
        gpu.cpu_affinity_list = g.cpu_affinity_list;
        gpu.cpu_cores = g.cpu_cores;
        gpu.memory_binding = g.memory_binding;
        gpu.network_devices = g.network_devices;
        topo.gpus.push_back(std::move(gpu));
    }

    for (auto const& n : base.network_devices) {
        network_device_info dev;
        dev.name = n.name;
        dev.numa_node = n.numa_node;
        dev.pci_bus_id = n.pci_bus_id;
        topo.network_devices.push_back(std::move(dev));
    }

    return topo;
}

void enrich_topology(system_topology& topo) {
    // NVML should already be initialized by cuCascade, but ensure it is
    nvmlInit_v2();

    add_infiniband_devices(topo);

    for (auto& gpu : topo.gpus) {
        if (gpu.pcie.generation == 0 && !gpu.pci_bus_id.empty()) {
            gpu.pcie = discover_pcie_info(gpu.pci_bus_id);
        }
        if (gpu.nvlink_peers.empty()) {
            gpu.nvlink_peers = discover_nvlink_connections(gpu.id);
        }
    }

    for (auto& dev : topo.network_devices) {
        if (dev.bandwidth_gbps == 0 && !dev.name.empty()) {
            dev.bandwidth_gbps = discover_nic_speed(dev.name);
        }
        if (dev.model_name.empty() && !dev.name.empty()) {
            dev.model_name = discover_nic_model(dev.name, dev.pci_bus_id);
        }
        if (dev.pcie.generation == 0 && !dev.pci_bus_id.empty()) {
            dev.pcie = discover_pcie_info(dev.pci_bus_id);
        }
    }

    if (topo.pcie_switches.empty()) {
        topo.pcie_switches = discover_pcie_switches(topo.gpus, topo.network_devices);
    }

    if (topo.cpus.empty() && topo.num_numa_nodes > 0) {
        topo.cpus = discover_cpus(topo.num_numa_nodes);
    }

    nvmlShutdown();
}

}  // namespace

bool topology_viz::discover() {
    cucascade::memory::topology_discovery base_discovery;
    if (!base_discovery.discover()) {
        return false;
    }

    topology_ = convert_from_cucascade(base_discovery.get_topology());
    enrich_topology(*topology_);
    return true;
}

bool topology_viz::load_json(std::string const& json_str) {
    try {
        topology_ = from_json(json_str);
        return true;
    } catch (std::exception const& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }
}

bool topology_viz::load_json_file(std::string const& path) {
    try {
        topology_ = from_json_file(path);
        return true;
    } catch (std::exception const& e) {
        std::cerr << "Failed to load JSON file: " << e.what() << std::endl;
        return false;
    }
}

bool topology_viz::enrich() {
    if (!topology_.has_value())
        return false;
    enrich_topology(*topology_);
    return true;
}

bool topology_viz::is_ready() const noexcept {
    return topology_.has_value();
}

system_topology const& topology_viz::get_topology() const {
    if (!topology_.has_value()) {
        throw std::runtime_error("No topology loaded or discovered");
    }
    return *topology_;
}

std::string topology_viz::to_json(int indent) const {
    if (!topology_.has_value()) {
        throw std::runtime_error("No topology loaded or discovered");
    }
    return rapidsmpf::topology::to_json(*topology_, indent);
}

}  // namespace rapidsmpf::topology
