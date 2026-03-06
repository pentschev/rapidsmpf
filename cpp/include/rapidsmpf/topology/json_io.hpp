/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <rapidsmpf/topology/types.hpp>

namespace rapidsmpf::topology {

/**
 * @brief Serialize a system topology to a JSON string.
 *
 * The output format is a superset of cuCascade's `topology_discovery` JSON:
 * all original fields are present in their original positions, and new fields
 * (`pcie`, `nvlink_peers`, `cpus`, NIC `model_name`/`bandwidth_gbps`, etc.) are
 * appended.  Fields with default / unknown values (0, empty string) are still
 * emitted so that the schema is self-documenting.
 *
 * @param topology  The topology to serialize.
 * @param indent    Number of spaces per indentation level (default 2).
 *                  Set to 0 for compact (single-line) output.
 *
 * @return A UTF-8 JSON string.
 */
[[nodiscard]] std::string to_json(system_topology const& topology, int indent = 2);

/**
 * @brief Deserialize a system topology from a JSON string.
 *
 * Accepts both the enriched format produced by `to_json()` and the
 * original cuCascade format (missing fields are left at their defaults).
 * This makes it possible to load a JSON file saved by the existing
 * `topology_discovery` tool and still produce a (partial) visualization.
 *
 * @param json_str  A UTF-8 JSON string.
 *
 * @return The parsed `system_topology`.
 *
 * @throws std::runtime_error if the string is not valid JSON or required
 *         top-level keys (`system`, `gpus`, `network_devices`) are missing.
 */
[[nodiscard]] system_topology from_json(std::string const& json_str);

/**
 * @brief Load a system topology from a JSON file.
 *
 * Convenience wrapper: reads the entire file into memory and delegates to
 * `from_json()`.
 *
 * @param path  Filesystem path to a JSON file.
 *
 * @return The parsed `system_topology`.
 *
 * @throws std::runtime_error if the file cannot be opened or parsed.
 */
[[nodiscard]] system_topology from_json_file(std::string const& path);

}  // namespace rapidsmpf::topology
