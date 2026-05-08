/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <rapidsmpf/communicator/ucxx.hpp>

namespace rapidsmpf::benchmark::dashboard {

class JsonlEventSink {
  public:
    explicit JsonlEventSink(std::filesystem::path path);

    [[nodiscard]] std::filesystem::path const& path() const noexcept;

    void clear() const;
    void publish_raw(std::string const& json) const;
    void publish_rank(
        rapidsmpf::Rank rank,
        rapidsmpf::Rank nranks,
        std::string const& hostname,
        int cuda_device,
        std::string const& gpu_pci_bus_id
    ) const;
    void publish_topology(std::string const& topology_json) const;
    void publish_topology_error(std::string const& message) const;
    void publish_transfer(rapidsmpf::ucxx::UCXX::TelemetryEvent const& event) const;

  private:
    std::filesystem::path path_;
};

class Server {
  public:
    Server(std::filesystem::path event_file, std::uint16_t requested_port);
    ~Server();

    Server(Server const&) = delete;
    Server& operator=(Server const&) = delete;
    Server(Server&&) = delete;
    Server& operator=(Server&&) = delete;

    [[nodiscard]] std::uint16_t port() const noexcept;
    [[nodiscard]] std::string url() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string default_event_file();
[[nodiscard]] std::string hostname();

}  // namespace rapidsmpf::benchmark::dashboard
