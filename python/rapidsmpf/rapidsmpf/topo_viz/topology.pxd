# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as bool_t
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "<rapidsmpf/topology/types.hpp>" nogil:
    cdef struct cpp_pcie_info "rapidsmpf::topology::pcie_info":
        int generation
        int width
        double bandwidth_gbps

    cdef struct cpp_nvlink_connection "rapidsmpf::topology::nvlink_connection":
        unsigned int peer_gpu_id
        int link_count
        int nvlink_version
        double bandwidth_gbps

    cdef struct cpp_cpu_topology_info "rapidsmpf::topology::cpu_topology_info":
        int numa_node
        string model_name
        int core_count
        string cpu_affinity_list

    cdef struct cpp_gpu_topology_info "rapidsmpf::topology::gpu_topology_info":
        unsigned int id
        string name
        string pci_bus_id
        string uuid
        int numa_node
        string cpu_affinity_list
        vector[int] cpu_cores
        vector[int] memory_binding
        vector[string] network_devices
        cpp_pcie_info pcie
        vector[cpp_nvlink_connection] nvlink_peers

    cdef struct cpp_network_device_info \
            "rapidsmpf::topology::network_device_info":
        string name
        int numa_node
        string pci_bus_id
        string model_name
        double bandwidth_gbps
        cpp_pcie_info pcie

    cdef struct cpp_pcie_switch_info \
            "rapidsmpf::topology::pcie_switch_info":
        string pci_bus_id
        int numa_node
        cpp_pcie_info pcie
        vector[unsigned int] gpu_ids
        vector[string] nic_names

    cdef struct cpp_system_topology "rapidsmpf::topology::system_topology":
        string hostname
        unsigned int num_gpus
        int num_numa_nodes
        int num_network_devices
        vector[cpp_cpu_topology_info] cpus
        vector[cpp_gpu_topology_info] gpus
        vector[cpp_network_device_info] network_devices
        vector[cpp_pcie_switch_info] pcie_switches


cdef extern from "<rapidsmpf/topology/topology_viz.hpp>" nogil:
    cdef cppclass cpp_topology_viz "rapidsmpf::topology::topology_viz":
        cpp_topology_viz()
        bool_t discover() except +
        bool_t load_json(const string& json_str) except +
        bool_t load_json_file(const string& path) except +
        bool_t enrich() except +
        bool_t is_ready() noexcept
        const cpp_system_topology& get_topology() except +
        string to_json(int indent) except +


cdef extern from "<rapidsmpf/topology/json_io.hpp>" nogil:
    string cpp_to_json "rapidsmpf::topology::to_json"(
        const cpp_system_topology& topology, int indent
    ) except +
    cpp_system_topology cpp_from_json "rapidsmpf::topology::from_json"(
        const string& json_str
    ) except +
    cpp_system_topology cpp_from_json_file \
        "rapidsmpf::topology::from_json_file"(
            const string& path
        ) except +


cdef class TopologyViz:
    cdef cpp_topology_viz _handle
