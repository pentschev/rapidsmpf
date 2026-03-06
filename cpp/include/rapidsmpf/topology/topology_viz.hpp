/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>

#include <rapidsmpf/topology/types.hpp>

namespace rapidsmpf::topology {

/**
 * @brief Enriched topology discovery and visualization entry point.
 *
 * `topology_viz` is the main C++ API for obtaining a fully enriched system
 * topology.  It combines cuCascade's base topology discovery with additional
 * bandwidth and naming enrichment, and exposes the result as a
 * `system_topology` struct (or as JSON).
 *
 * Typical usage:
 * @code
 *   rapidsmpf::topology::topology_viz viz;
 *   if (viz.discover()) {
 *       // Use the topology programmatically...
 *       auto const& topo = viz.get_topology();
 *       for (auto const& gpu : topo.gpus) { ... }
 *
 *       // ...or serialize to JSON.
 *       std::string json = viz.to_json();
 *   }
 * @endcode
 *
 * Alternatively, load from a previously-saved JSON file:
 * @code
 *   rapidsmpf::topology::topology_viz viz;
 *   viz.load_json_file("topology.json");
 *   auto const& topo = viz.get_topology();
 * @endcode
 */
class topology_viz {
  public:
    /**
     * @brief Discover the full enriched topology of the local system.
     *
     * Performs the following steps:
     *   1. Runs cuCascade's `topology_discovery::discover()` for base GPU,
     *      NUMA, and network-device topology.
     *   2. Enriches every GPU with PCIe info and NVLink peer connections.
     *   3. Enriches every network device with link speed, model name, and
     *      PCIe info.
     *   4. Discovers per-NUMA-node CPU information (model name, core count).
     *
     * On success the enriched topology is stored internally and can be
     * accessed via `get_topology()` or serialized via `to_json()`.
     *
     * @return `true` on success, `false` if the base cuCascade discovery
     *         fails.  Partial enrichment failures (e.g., sysfs not
     *         accessible) are not fatal — the corresponding fields are left
     *         at their defaults.
     */
    [[nodiscard]] bool discover();

    /**
     * @brief Load topology from a JSON string.
     *
     * Accepts both the enriched format produced by `to_json()` and the
     * original cuCascade JSON format.  Missing enrichment fields are left
     * at their defaults (zero / empty).
     *
     * If called on a live system the caller may optionally call
     * `enrich()` afterwards to fill in missing bandwidth and naming data.
     *
     * @param json_str  A UTF-8 JSON string.
     *
     * @return `true` on success, `false` if parsing fails.
     */
    [[nodiscard]] bool load_json(std::string const& json_str);

    /**
     * @brief Load topology from a JSON file.
     *
     * Convenience wrapper around `load_json()`.
     *
     * @param path  Filesystem path to a JSON file.
     *
     * @return `true` on success, `false` if the file cannot be opened or
     *         parsed.
     */
    [[nodiscard]] bool load_json_file(std::string const& path);

    /**
     * @brief Re-enrich an already-loaded topology with live system data.
     *
     * Useful after `load_json()` / `load_json_file()` when running on the
     * same (or compatible) hardware: fills in any zero / empty bandwidth and
     * naming fields by querying sysfs and NVML.
     *
     * No-op if no topology has been loaded yet.
     *
     * @return `true` if a topology was present and enrichment ran (even if
     *         some individual queries failed), `false` if no topology is
     *         loaded.
     */
    [[nodiscard]] bool enrich();

    /**
     * @brief Check whether a topology has been loaded or discovered.
     *
     * @return `true` if a topology is available, `false` otherwise.
     */
    [[nodiscard]] bool is_ready() const noexcept;

    /**
     * @brief Access the enriched topology.
     *
     * @pre `is_ready()` returns `true`.
     *
     * @return Const reference to the stored `system_topology`.
     *
     * @throws std::runtime_error if no topology has been loaded or
     *         discovered.
     */
    [[nodiscard]] system_topology const& get_topology() const;

    /**
     * @brief Serialize the stored topology to a JSON string.
     *
     * @param indent  Number of spaces per indentation level (default 2).
     *
     * @pre `is_ready()` returns `true`.
     *
     * @return A UTF-8 JSON string.
     *
     * @throws std::runtime_error if no topology has been loaded or
     *         discovered.
     */
    [[nodiscard]] std::string to_json(int indent = 2) const;

  private:
    std::optional<system_topology> topology_;
};

}  // namespace rapidsmpf::topology
