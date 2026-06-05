/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

enum class CopyMode : std::uint8_t {
    Get,
    Put
};
enum class CopyApi : std::uint8_t {
    Generic,
    DtoD,
    Peer
};
enum class StreamMode : std::uint8_t {
    Single,
    PerPeer,
    PerCopy
};
enum class PostOrder : std::uint8_t {
    DeviceMajor,
    Balanced
};

struct Args {
    std::size_t message_bytes{1 << 20};
    std::uint64_t num_ops{1};
    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{1};
    std::vector<int> devices;
    std::optional<std::size_t> num_devices;
    CopyMode copy_mode{CopyMode::Get};
    CopyApi copy_api{CopyApi::DtoD};
    StreamMode stream_mode{StreamMode::PerPeer};
    PostOrder post_order{PostOrder::DeviceMajor};
    bool gate_launch{true};
    bool enable_peer_access{true};
};

struct DeviceState {
    int ordinal{};
    CUdevice device{};
    CUcontext context{};
    CUdeviceptr send{0};
    CUdeviceptr recv{0};
    int* release_flag_host{nullptr};
    std::vector<CUstream> streams;
    std::vector<std::size_t> active_streams;
    CUstream release_stream{};
    std::string name;
    std::array<char, 32> pci_bus_id{};
};

__global__ void wait_for_release(int const* release_flag) {
    auto const* flag = static_cast<int volatile const*>(release_flag);
    while (*flag == 0) {
        __nanosleep(256);
    }
}

[[noreturn]] void fail(std::string const& message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, char const* expr, char const* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream ss;
    ss << file << ":" << line << ": " << expr
       << " failed: " << cudaGetErrorString(status);
    fail(ss.str());
}

void check_cu(CUresult status, char const* expr, char const* file, int line) {
    if (status == CUDA_SUCCESS) {
        return;
    }
    char const* name = nullptr;
    char const* desc = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &desc);
    std::ostringstream ss;
    ss << file << ":" << line << ": " << expr
       << " failed: " << (name == nullptr ? "unknown" : name);
    if (desc != nullptr) {
        ss << " (" << desc << ")";
    }
    fail(ss.str());
}

#define CUDA_RT(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUDA_DRV(call) check_cu((call), #call, __FILE__, __LINE__)

std::string lower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::uint64_t parse_u64(std::string const& value) {
    if (value.empty()) {
        fail("invalid integer: " + value);
    }
    if (!std::ranges::all_of(value, [](unsigned char c) { return std::isdigit(c); })) {
        fail("invalid integer: " + value);
    }
    std::size_t pos = 0;
    auto const parsed = std::stoull(value, &pos, 10);
    if (pos != value.size()) {
        fail("invalid integer: " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::size_t checked_to_size(std::uint64_t value, std::string_view what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        fail(std::string{what} + " exceeds size_t range");
    }
    return static_cast<std::size_t>(value);
}

int checked_to_int(std::uint64_t value, std::string_view what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        fail(std::string{what} + " exceeds int range");
    }
    return static_cast<int>(value);
}

std::size_t parse_size(std::string value) {
    value = lower(value);
    auto const suffix_pos = value.find_first_not_of("0123456789");
    auto const number = value.substr(0, suffix_pos);
    if (number.empty()) {
        fail("invalid size: " + value);
    }

    auto suffix =
        suffix_pos == std::string::npos ? std::string{} : value.substr(suffix_pos);
    std::uint64_t multiplier = 1;
    if (suffix.empty() || suffix == "b") {
        multiplier = 1;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        multiplier = 1ull << 10;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        multiplier = 1ull << 20;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        multiplier = 1ull << 30;
    } else if (suffix == "t" || suffix == "tb" || suffix == "tib") {
        multiplier = 1ull << 40;
    } else {
        fail("invalid size suffix: " + suffix);
    }

    auto const count = parse_u64(number);
    if (count > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        fail("size overflow: " + value);
    }
    auto const bytes = count * multiplier;
    return checked_to_size(bytes, "size");
}

std::string next_arg(int& i, int argc, char** argv, std::string const& option) {
    if (i + 1 >= argc) {
        fail("missing value for " + option);
    }
    return argv[++i];
}

std::vector<int> parse_devices(std::string const& value) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start <= value.size()) {
        auto const comma = value.find(',', start);
        auto const token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start
        );
        if (token.empty()) {
            fail("invalid --devices list: " + value);
        }
        result.push_back(checked_to_int(parse_u64(token), "device ordinal"));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return result;
}

void usage(char const* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "Options:\n"
        << "  -n, --bytes <size>       Message size per peer/op (default: 1MiB)\n"
        << "  -p, --ops <num>          Concurrent all-to-all operations (default: 1)\n"
        << "  -r, --runs <num>         Measured runs (default: 1)\n"
        << "  -w, --warmups <num>      Warmup runs using the same buffers (default: 1)\n"
        << "  --devices <list>         Comma-separated CUDA device ordinals "
           "(default: all visible)\n"
        << "  --num-devices <num>      Use devices 0..num-1 instead of all visible "
           "devices\n"
        << "  --mode <get|put>         GET reads peer send buffers; PUT writes peer recv "
           "buffers (default: get)\n"
        << "  --copy-api <d2d|generic|peer>\n"
        << "                           Driver copy API to enqueue (default: d2d)\n"
        << "  --streams <single|per-peer|per-copy>\n"
        << "                           CUDA stream assignment per device "
           "(default: per-peer)\n"
        << "  --post-order <device-major|balanced>\n"
        << "                           Copy posting order (default: device-major)\n"
        << "  --no-gate                Do not gate streams with a device-side release "
           "flag\n"
        << "  --no-peer-enable         Do not call cuCtxEnablePeerAccess\n"
        << "  -h, --help               Show this help\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "-n" || arg == "--bytes") {
            args.message_bytes = parse_size(next_arg(i, argc, argv, arg));
        } else if (arg == "-p" || arg == "--ops") {
            args.num_ops = parse_u64(next_arg(i, argc, argv, arg));
        } else if (arg == "-r" || arg == "--runs") {
            args.num_runs = parse_u64(next_arg(i, argc, argv, arg));
        } else if (arg == "-w" || arg == "--warmups") {
            args.num_warmups = parse_u64(next_arg(i, argc, argv, arg));
        } else if (arg == "--devices") {
            args.devices = parse_devices(next_arg(i, argc, argv, arg));
        } else if (arg == "--num-devices") {
            args.num_devices = checked_to_size(
                parse_u64(next_arg(i, argc, argv, arg)), "number of devices"
            );
        } else if (arg == "--mode") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "get") {
                args.copy_mode = CopyMode::Get;
            } else if (mode == "put") {
                args.copy_mode = CopyMode::Put;
            } else {
                fail("invalid --mode: " + mode);
            }
        } else if (arg == "--copy-api") {
            auto const api = lower(next_arg(i, argc, argv, arg));
            if (api == "generic") {
                args.copy_api = CopyApi::Generic;
            } else if (api == "d2d") {
                args.copy_api = CopyApi::DtoD;
            } else if (api == "peer") {
                args.copy_api = CopyApi::Peer;
            } else {
                fail("invalid --copy-api: " + api);
            }
        } else if (arg == "--streams") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "single") {
                args.stream_mode = StreamMode::Single;
            } else if (mode == "per-peer") {
                args.stream_mode = StreamMode::PerPeer;
            } else if (mode == "per-copy") {
                args.stream_mode = StreamMode::PerCopy;
            } else {
                fail("invalid --streams: " + mode);
            }
        } else if (arg == "--post-order") {
            auto const order = lower(next_arg(i, argc, argv, arg));
            if (order == "device-major" || order == "major") {
                args.post_order = PostOrder::DeviceMajor;
            } else if (order == "balanced" || order == "round-robin") {
                args.post_order = PostOrder::Balanced;
            } else {
                fail("invalid --post-order: " + order);
            }
        } else if (arg == "--no-gate") {
            args.gate_launch = false;
        } else if (arg == "--no-peer-enable") {
            args.enable_peer_access = false;
        } else {
            fail("unknown option: " + arg);
        }
    }
    if (args.message_bytes == 0) {
        fail("message size must be greater than zero");
    }
    if (args.num_ops == 0) {
        fail("number of operations must be greater than zero");
    }
    if (args.num_runs == 0) {
        fail("number of measured runs must be greater than zero");
    }
    if (!args.devices.empty() && args.num_devices.has_value()) {
        fail("--devices and --num-devices are mutually exclusive");
    }
    return args;
}

std::string to_string(CopyMode mode) {
    return mode == CopyMode::Get ? "get" : "put";
}

std::string to_string(CopyApi api) {
    switch (api) {
    case CopyApi::Generic:
        return "generic";
    case CopyApi::DtoD:
        return "d2d";
    case CopyApi::Peer:
        return "peer";
    }
    fail("unknown copy API");
}

std::string to_string(StreamMode mode) {
    switch (mode) {
    case StreamMode::Single:
        return "single";
    case StreamMode::PerPeer:
        return "per-peer";
    case StreamMode::PerCopy:
        return "per-copy";
    }
    fail("unknown stream mode");
}

std::string to_string(PostOrder order) {
    switch (order) {
    case PostOrder::DeviceMajor:
        return "device-major";
    case PostOrder::Balanced:
        return "balanced";
    }
    fail("unknown post order");
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        fail("size overflow");
    }
    return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        fail("size overflow");
    }
    return lhs + rhs;
}

std::size_t buffer_offset(
    Args const& args, std::size_t num_devices, std::uint64_t op, std::size_t peer_index
) {
    return checked_mul(
        args.message_bytes,
        checked_add(
            checked_mul(checked_to_size(op, "operation index"), num_devices), peer_index
        )
    );
}

std::size_t stream_count(Args const& args, std::size_t num_devices) {
    switch (args.stream_mode) {
    case StreamMode::Single:
        return 1;
    case StreamMode::PerPeer:
        return num_devices;
    case StreamMode::PerCopy:
        return checked_mul(
            checked_to_size(args.num_ops, "number of operations"), num_devices - 1
        );
    }
    fail("unknown stream mode");
}

std::size_t stream_index(
    Args const& args, std::size_t peer_index, std::uint64_t copy_index
) {
    switch (args.stream_mode) {
    case StreamMode::Single:
        return 0;
    case StreamMode::PerPeer:
        return peer_index;
    case StreamMode::PerCopy:
        return checked_to_size(copy_index, "copy index");
    }
    fail("unknown stream mode");
}

std::vector<std::size_t> active_stream_indices(
    Args const& args, std::size_t local_index, std::size_t num_devices
) {
    std::vector<std::size_t> result;
    switch (args.stream_mode) {
    case StreamMode::Single:
        result.push_back(0);
        break;
    case StreamMode::PerPeer:
        for (std::size_t peer = 0; peer < num_devices; ++peer) {
            if (peer != local_index) {
                result.push_back(peer);
            }
        }
        break;
    case StreamMode::PerCopy:
        result.resize(checked_mul(
            checked_to_size(args.num_ops, "number of operations"), num_devices - 1
        ));
        std::iota(result.begin(), result.end(), std::size_t{0});
        break;
    }
    return result;
}

void set_current_driver(DeviceState const& device) {
    CUDA_DRV(cuCtxSetCurrent(device.context));
}

void set_current_runtime(DeviceState const& device) {
    CUDA_RT(cudaSetDevice(device.ordinal));
    CUDA_DRV(cuCtxSetCurrent(device.context));
}

std::vector<int> selected_devices(Args const& args, int device_count) {
    if (device_count <= 0) {
        fail("no visible CUDA devices");
    }

    auto const visible_count = static_cast<std::size_t>(device_count);
    std::vector<int> devices = args.devices;
    if (devices.empty()) {
        auto const count = args.num_devices.value_or(visible_count);
        if (count > visible_count) {
            fail("--num-devices exceeds the number of visible CUDA devices");
        }
        devices.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            devices.push_back(
                checked_to_int(static_cast<std::uint64_t>(i), "device ordinal")
            );
        }
    }

    if (devices.size() < 2) {
        fail("bench_cuda_p2p requires at least two devices");
    }

    auto sorted = devices;
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end()) {
        fail("duplicate CUDA device ordinal in selection");
    }
    for (auto ordinal : devices) {
        if (ordinal < 0 || ordinal >= device_count) {
            fail("selected CUDA device is not visible: " + std::to_string(ordinal));
        }
    }
    return devices;
}

void enable_peer_access(std::vector<DeviceState> const& devices) {
    for (auto const& local : devices) {
        for (auto const& peer : devices) {
            if (local.ordinal == peer.ordinal) {
                continue;
            }

            int can_access = 0;
            CUDA_DRV(cuDeviceCanAccessPeer(&can_access, local.device, peer.device));
            if (can_access == 0) {
                std::ostringstream ss;
                ss << "CUDA device " << local.ordinal << " cannot access peer device "
                   << peer.ordinal;
                fail(ss.str());
            }

            set_current_driver(local);
            auto const status = cuCtxEnablePeerAccess(peer.context, 0u);
            if (status == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                continue;
            }
            CUDA_DRV(status);
        }
    }
}

void launch_wait_kernel(DeviceState const& device, CUstream stream) {
    set_current_runtime(device);
    wait_for_release<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        device.release_flag_host
    );
    CUDA_RT(cudaGetLastError());
}

void store_release_flag(DeviceState const& device, int value) {
    *reinterpret_cast<int volatile*>(device.release_flag_host) = value;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void enqueue_copy(
    Args const& args,
    DeviceState const& local,
    DeviceState const& peer,
    std::size_t local_index,
    std::size_t peer_index,
    std::size_t num_devices,
    std::uint64_t op,
    std::uint64_t copy_index
) {
    set_current_driver(local);
    auto const stream = local.streams.at(stream_index(args, peer_index, copy_index));

    CUdeviceptr dst = 0;
    CUdeviceptr src = 0;
    CUcontext dst_context = nullptr;
    CUcontext src_context = nullptr;
    if (args.copy_mode == CopyMode::Get) {
        src = peer.send + buffer_offset(args, num_devices, op, local_index);
        dst = local.recv + buffer_offset(args, num_devices, op, peer_index);
        src_context = peer.context;
        dst_context = local.context;
    } else {
        src = local.send + buffer_offset(args, num_devices, op, peer_index);
        dst = peer.recv + buffer_offset(args, num_devices, op, local_index);
        src_context = local.context;
        dst_context = peer.context;
    }

    switch (args.copy_api) {
    case CopyApi::Generic:
        CUDA_DRV(cuMemcpyAsync(dst, src, args.message_bytes, stream));
        break;
    case CopyApi::DtoD:
        CUDA_DRV(cuMemcpyDtoDAsync(dst, src, args.message_bytes, stream));
        break;
    case CopyApi::Peer:
        CUDA_DRV(cuMemcpyPeerAsync(
            dst, dst_context, src, src_context, args.message_bytes, stream
        ));
        break;
    }
}

using DevicePair = std::pair<std::size_t, std::size_t>;
using PairRound = std::vector<DevicePair>;

std::vector<std::size_t> initial_round_robin_slots(std::size_t num_devices) {
    auto const participant_count = num_devices + (num_devices % 2);
    std::vector<std::size_t> slots;
    slots.reserve(participant_count);

    for (std::size_t i = 0; i < participant_count; i += 2) {
        slots.push_back(i);
    }
    for (std::size_t i = participant_count; i > 1; i -= 2) {
        slots.push_back(i - 1);
    }
    return slots;
}

std::vector<PairRound> round_robin_pair_rounds(std::size_t num_devices) {
    auto const participant_count = num_devices + (num_devices % 2);
    auto slots = initial_round_robin_slots(num_devices);
    std::vector<PairRound> rounds;
    rounds.reserve(participant_count - 1);

    for (std::size_t round = 0; round < participant_count - 1; ++round) {
        PairRound pairs;
        pairs.reserve(participant_count / 2);
        for (std::size_t slot = 0; slot < participant_count / 2; ++slot) {
            auto const first = slots.at(slot);
            auto const second = slots.at(participant_count - 1 - slot);
            if (first < num_devices && second < num_devices) {
                pairs.emplace_back(first, second);
            }
        }
        rounds.push_back(std::move(pairs));

        auto const last = slots.back();
        for (std::size_t i = participant_count - 1; i > 1; --i) {
            slots.at(i) = slots.at(i - 1);
        }
        slots.at(1) = last;
    }
    return rounds;
}

void enqueue_transfer(
    Args const& args,
    std::vector<DeviceState> const& devices,
    std::size_t src_index,
    std::size_t dst_index,
    std::uint64_t op,
    std::vector<std::uint64_t>& copy_indices
) {
    auto const local_index = args.copy_mode == CopyMode::Get ? dst_index : src_index;
    auto const peer_index = args.copy_mode == CopyMode::Get ? src_index : dst_index;
    auto& copy_index = copy_indices.at(local_index);
    enqueue_copy(
        args,
        devices.at(local_index),
        devices.at(peer_index),
        local_index,
        peer_index,
        devices.size(),
        op,
        copy_index
    );
    ++copy_index;
}

void enqueue_device_major_copies(
    Args const& args, std::vector<DeviceState> const& devices
) {
    auto const num_devices = devices.size();
    for (std::size_t local_index = 0; local_index < num_devices; ++local_index) {
        auto const& local = devices.at(local_index);
        std::uint64_t copy_index = 0;
        for (std::uint64_t op = 0; op < args.num_ops; ++op) {
            for (std::size_t peer_index = 0; peer_index < num_devices; ++peer_index) {
                if (peer_index == local_index) {
                    continue;
                }
                enqueue_copy(
                    args,
                    local,
                    devices.at(peer_index),
                    local_index,
                    peer_index,
                    num_devices,
                    op,
                    copy_index
                );
                ++copy_index;
            }
        }
    }
}

void enqueue_balanced_copies(Args const& args, std::vector<DeviceState> const& devices) {
    auto const rounds = round_robin_pair_rounds(devices.size());
    std::vector<std::uint64_t> copy_indices(devices.size(), 0);
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (auto const& round : rounds) {
            for (auto const& pair : round) {
                enqueue_transfer(
                    args, devices, pair.first, pair.second, op, copy_indices
                );
            }
            for (auto const& pair : round) {
                enqueue_transfer(
                    args, devices, pair.second, pair.first, op, copy_indices
                );
            }
        }
    }
}

void enqueue_all_copies(Args const& args, std::vector<DeviceState> const& devices) {
    switch (args.post_order) {
    case PostOrder::DeviceMajor:
        enqueue_device_major_copies(args, devices);
        break;
    case PostOrder::Balanced:
        enqueue_balanced_copies(args, devices);
        break;
    }
}

void reset_release_flags(std::vector<DeviceState> const& devices) {
    for (auto const& device : devices) {
        store_release_flag(device, 0);
    }
}

void synchronize_active_streams(std::vector<DeviceState> const& devices) {
    for (auto const& device : devices) {
        set_current_driver(device);
        for (auto index : device.active_streams) {
            CUDA_DRV(cuStreamSynchronize(device.streams.at(index)));
        }
        CUDA_DRV(cuStreamSynchronize(device.release_stream));
    }
}

double run_once(Args const& args, std::vector<DeviceState> const& devices) {
    reset_release_flags(devices);

    if (args.gate_launch) {
        for (auto const& device : devices) {
            for (auto index : device.active_streams) {
                launch_wait_kernel(device, device.streams.at(index));
            }
        }
        enqueue_all_copies(args, devices);

        auto const t0 = std::chrono::steady_clock::now();
        for (auto const& device : devices) {
            store_release_flag(device, 1);
        }
        synchronize_active_streams(devices);
        auto const t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    auto const t0 = std::chrono::steady_clock::now();
    enqueue_all_copies(args, devices);
    synchronize_active_streams(devices);
    auto const t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void print_config(Args const& args, std::vector<DeviceState> const& devices) {
    std::cout << "Benchmark: single-process CUDA P2P all-to-all\n"
              << "Arguments:\n"
              << "  --bytes " << args.message_bytes << "\n"
              << "  --ops " << args.num_ops << "\n"
              << "  --runs " << args.num_runs << "\n"
              << "  --warmups " << args.num_warmups << "\n"
              << "  --mode " << to_string(args.copy_mode) << "\n"
              << "  --copy-api " << to_string(args.copy_api) << "\n"
              << "  --streams " << to_string(args.stream_mode) << "\n"
              << "  --post-order " << to_string(args.post_order) << "\n"
              << "  gate_launch " << (args.gate_launch ? "true" : "false") << "\n"
              << "  peer_access_enabled " << (args.enable_peer_access ? "true" : "false")
              << "\n"
              << "  devices " << devices.size() << "\n";

    for (std::size_t i = 0; i < devices.size(); ++i) {
        auto const& device = devices.at(i);
        std::cout << "  logical " << i << ": cuda device " << device.ordinal << " pci "
                  << device.pci_bus_id.data() << " " << device.name << "\n";
    }
}

void print_rate(double elapsed, std::size_t bytes, bool warmup) {
    auto const gib =
        static_cast<double>(bytes) / elapsed / static_cast<double>(1ull << 30);
    auto const tib =
        static_cast<double>(bytes) / elapsed / static_cast<double>(1ull << 40);
    std::cout << std::fixed << std::setprecision(6) << "elapsed: " << elapsed << " s"
              << " | aggregate: " << std::setprecision(2) << gib << " GiB/s"
              << " (" << std::setprecision(3) << tib << " TiB/s)";
    if (warmup) {
        std::cout << " (warmup run)";
    }
    std::cout << "\n";
}

void cleanup(std::vector<DeviceState>& devices) {
    for (auto& device : devices) {
        if (device.context != nullptr) {
            set_current_driver(device);
        }
        for (auto& stream : device.streams) {
            if (stream != nullptr) {
                CUDA_DRV(cuStreamDestroy(stream));
                stream = nullptr;
            }
        }
        device.streams.clear();
        if (device.release_stream != nullptr) {
            CUDA_DRV(cuStreamDestroy(device.release_stream));
            device.release_stream = nullptr;
        }
        if (device.release_flag_host != nullptr) {
            CUDA_DRV(cuMemFreeHost(device.release_flag_host));
            device.release_flag_host = nullptr;
        }
        if (device.recv != 0) {
            CUDA_DRV(cuMemFree(device.recv));
            device.recv = 0;
        }
        if (device.send != 0) {
            CUDA_DRV(cuMemFree(device.send));
            device.send = 0;
        }
        if (device.context != nullptr) {
            CUDA_DRV(cuDevicePrimaryCtxRelease(device.device));
            device.context = nullptr;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<DeviceState> devices;
    try {
        auto args = parse_args(argc, argv);

        CUDA_DRV(cuInit(0));
        int visible_devices = 0;
        CUDA_RT(cudaGetDeviceCount(&visible_devices));
        auto const ordinals = selected_devices(args, visible_devices);
        auto const num_devices = ordinals.size();

        auto const slots_per_allocation = checked_mul(
            checked_to_size(args.num_ops, "number of operations"), num_devices
        );
        auto const allocation_bytes =
            checked_mul(args.message_bytes, slots_per_allocation);

        devices.reserve(num_devices);
        for (auto ordinal : ordinals) {
            DeviceState device;
            device.ordinal = ordinal;
            CUDA_DRV(cuDeviceGet(&device.device, ordinal));
            CUDA_RT(cudaSetDevice(ordinal));
            CUDA_RT(cudaFree(nullptr));
            CUDA_DRV(cuDevicePrimaryCtxRetain(&device.context, device.device));
            set_current_driver(device);

            std::array<char, 256> name{};
            CUDA_DRV(
                cuDeviceGetName(name.data(), static_cast<int>(name.size()), device.device)
            );
            device.name = name.data();
            CUDA_RT(cudaDeviceGetPCIBusId(
                device.pci_bus_id.data(),
                static_cast<int>(device.pci_bus_id.size()),
                ordinal
            ));

            CUDA_DRV(cuMemAlloc(&device.send, allocation_bytes));
            CUDA_DRV(cuMemAlloc(&device.recv, allocation_bytes));
            void* release_flag_host = nullptr;
            CUDA_DRV(
                cuMemHostAlloc(&release_flag_host, sizeof(int), CU_MEMHOSTALLOC_PORTABLE)
            );
            device.release_flag_host = static_cast<int*>(release_flag_host);
            CUDA_DRV(cuMemsetD8(
                device.send, static_cast<unsigned char>(0xa5), allocation_bytes
            ));
            CUDA_DRV(
                cuMemsetD8(device.recv, static_cast<unsigned char>(0), allocation_bytes)
            );
            store_release_flag(device, 0);
            CUDA_DRV(cuCtxSynchronize());

            devices.push_back(std::move(device));
        }

        if (args.enable_peer_access) {
            enable_peer_access(devices);
        }

        auto const streams_per_device = stream_count(args, num_devices);
        for (std::size_t i = 0; i < devices.size(); ++i) {
            auto& device = devices.at(i);
            set_current_driver(device);
            device.streams.resize(streams_per_device);
            for (auto& stream : device.streams) {
                CUDA_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
            }
            CUDA_DRV(cuStreamCreate(&device.release_stream, CU_STREAM_NON_BLOCKING));
            device.active_streams = active_stream_indices(args, i, num_devices);
        }

        print_config(args, devices);

        auto const local_bytes = checked_mul(
            checked_mul(
                args.message_bytes, checked_to_size(args.num_ops, "number of operations")
            ),
            num_devices - 1
        );
        auto const aggregate_bytes = checked_mul(local_bytes, num_devices);
        for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
            auto const elapsed = run_once(args, devices);
            print_rate(elapsed, aggregate_bytes, i < args.num_warmups);
        }

        cleanup(devices);
        return 0;
    } catch (std::exception const& e) {
        std::cerr << "bench_cuda_p2p: " << e.what() << "\n";
        try {
            cleanup(devices);
        } catch (...) {
        }
        return 1;
    }
}
