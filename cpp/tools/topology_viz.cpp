/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <rapidsmpf/topology/topology_viz.hpp>

void usage(char const* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "\n"
              << "Discover system topology with bandwidth and naming enrichment,\n"
              << "or load from a JSON file.  Outputs enriched JSON.\n"
              << "\n"
              << "Options:\n"
              << "  --json <file>    Load topology from a JSON file instead of\n"
              << "                   live discovery\n"
              << "  --enrich         Re-enrich a loaded JSON with live system data\n"
              << "  --output <file>  Write output to a file instead of stdout\n"
              << "  --compact        Output compact JSON (no indentation)\n"
              << "  --help           Show this help message\n";
}

int main(int argc, char** argv) {
    std::string json_file;
    std::string output_file;
    bool do_enrich = false;
    bool compact = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_file = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--enrich") == 0) {
            do_enrich = true;
        } else if (std::strcmp(argv[i], "--compact") == 0) {
            compact = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n\n";
            usage(argv[0]);
            return 1;
        }
    }

    rapidsmpf::topology::topology_viz viz;

    if (!json_file.empty()) {
        if (!viz.load_json_file(json_file)) {
            std::cerr << "Failed to load JSON file: " << json_file << std::endl;
            return 1;
        }
        if (do_enrich) {
            static_cast<void>(viz.enrich());
        }
    } else {
        if (!viz.discover()) {
            std::cerr << "Failed to discover system topology" << std::endl;
            return 1;
        }
    }

    int indent = compact ? 0 : 2;
    std::string json = viz.to_json(indent);

    if (!output_file.empty()) {
        std::ofstream out{output_file};
        if (!out.is_open()) {
            std::cerr << "Cannot open output file: " << output_file << std::endl;
            return 1;
        }
        out << json;
    } else {
        std::cout << json;
    }

    return 0;
}
