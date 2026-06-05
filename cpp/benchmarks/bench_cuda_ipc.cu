/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>

namespace {

enum class CopyMode : std::uint8_t {
    Get,
    Put
};
enum class StreamMode : std::uint8_t {
    Single,
    PerPeer,
    PerCopy
};
enum class PostOrder : std::uint8_t {
    RankMajor,
    Balanced
};

struct Args {
    std::size_t message_bytes{1 << 20};
    std::uint64_t num_ops{1};
    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{1};
    int device{-1};
    CopyMode copy_mode{CopyMode::Get};
    StreamMode stream_mode{StreamMode::PerPeer};
    PostOrder post_order{PostOrder::RankMajor};
    bool gate_launch{true};
};

__global__ void wait_for_release(int const* release_flag) {
    while (atomicAdd(const_cast<int*>(release_flag), 0) == 0) {
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

void check_mpi(int status, char const* expr, char const* file, int line) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char error[MPI_MAX_ERROR_STRING]{};
    int len = 0;
    MPI_Error_string(status, error, &len);
    std::ostringstream ss;
    ss << file << ":" << line << ": " << expr
       << " failed: " << std::string(error, static_cast<std::size_t>(len));
    fail(ss.str());
}

#define CUDA_RT(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUDA_DRV(call) check_cu((call), #call, __FILE__, __LINE__)
#define MPI_CALL(call) check_mpi((call), #call, __FILE__, __LINE__)

std::string lower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::uint64_t parse_u64(std::string const& value) {
    std::size_t pos = 0;
    auto result = std::stoull(value, &pos, 10);
    if (pos != value.size()) {
        fail("invalid integer: " + value);
    }
    return result;
}

int checked_mpi_count(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        fail("MPI byte count exceeds int range");
    }
    return static_cast<int>(size);
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
    if (count > std::numeric_limits<std::size_t>::max() / multiplier) {
        fail("size overflow: " + value);
    }
    return static_cast<std::size_t>(count * multiplier);
}

std::string next_arg(int& i, int argc, char** argv, std::string const& option) {
    if (i + 1 >= argc) {
        fail("missing value for " + option);
    }
    return argv[++i];
}

void usage(char const* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "Options:\n"
        << "  -n, --bytes <size>       Message size per peer/op (default: 1MiB)\n"
        << "  -p, --ops <num>          Concurrent all-to-all operations (default: 1)\n"
        << "  -r, --runs <num>         Measured runs (default: 1)\n"
        << "  -w, --warmups <num>      Warmup runs using the same buffers (default: 1)\n"
        << "  --mode <get|put>         GET opens peer send buffers; PUT opens peer recv "
           "buffers (default: get)\n"
        << "  --streams <single|per-peer|per-copy>\n"
        << "                           CUDA stream assignment (default: per-peer)\n"
        << "  -P, --post-order <rank-major|balanced>\n"
        << "                           Copy posting order (default: rank-major)\n"
        << "  --device <num>           CUDA device for this rank (default: rank % "
           "visible devices)\n"
        << "  --no-gate                Do not gate streams with a device-side release "
           "flag\n"
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
        } else if (arg == "--mode") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "get") {
                args.copy_mode = CopyMode::Get;
            } else if (mode == "put") {
                args.copy_mode = CopyMode::Put;
            } else {
                fail("invalid --mode: " + mode);
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
        } else if (arg == "-P" || arg == "--post-order" || arg == "--post-mode") {
            auto const order = lower(next_arg(i, argc, argv, arg));
            if (order == "rank-major" || order == "device-major" || order == "major") {
                args.post_order = PostOrder::RankMajor;
            } else if (order == "balanced" || order == "round-robin") {
                args.post_order = PostOrder::Balanced;
            } else {
                fail("invalid --post-order: " + order);
            }
        } else if (arg == "--device") {
            args.device = static_cast<int>(parse_u64(next_arg(i, argc, argv, arg)));
        } else if (arg == "--no-gate") {
            args.gate_launch = false;
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
    return args;
}

std::string to_string(CopyMode mode) {
    return mode == CopyMode::Get ? "get" : "put";
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
    case PostOrder::RankMajor:
        return "rank-major";
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

std::size_t buffer_offset(Args const& args, int nranks, std::uint64_t op, int rank) {
    return checked_mul(
        args.message_bytes,
        checked_mul(static_cast<std::size_t>(op), static_cast<std::size_t>(nranks))
            + static_cast<std::size_t>(rank)
    );
}

std::size_t stream_count(Args const& args, int nranks) {
    switch (args.stream_mode) {
    case StreamMode::Single:
        return 1;
    case StreamMode::PerPeer:
        return static_cast<std::size_t>(nranks);
    case StreamMode::PerCopy:
        return checked_mul(
            static_cast<std::size_t>(args.num_ops), static_cast<std::size_t>(nranks - 1)
        );
    }
    fail("unknown stream mode");
}

std::size_t stream_index(Args const& args, int peer, std::uint64_t copy_index) {
    switch (args.stream_mode) {
    case StreamMode::Single:
        return 0;
    case StreamMode::PerPeer:
        return static_cast<std::size_t>(peer);
    case StreamMode::PerCopy:
        return static_cast<std::size_t>(copy_index);
    }
    fail("unknown stream mode");
}

std::vector<std::size_t> active_stream_indices(Args const& args, int rank, int nranks) {
    std::vector<std::size_t> result;
    switch (args.stream_mode) {
    case StreamMode::Single:
        result.push_back(0);
        break;
    case StreamMode::PerPeer:
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer != rank) {
                result.push_back(static_cast<std::size_t>(peer));
            }
        }
        break;
    case StreamMode::PerCopy:
        result.resize(checked_mul(
            static_cast<std::size_t>(args.num_ops), static_cast<std::size_t>(nranks - 1)
        ));
        std::iota(result.begin(), result.end(), std::size_t{0});
        break;
    }
    return result;
}

class UniqueFd {
  public:
    UniqueFd() = default;

    explicit UniqueFd(int fd) : fd_{fd} {}

    UniqueFd(UniqueFd const&) = delete;
    UniqueFd& operator=(UniqueFd const&) = delete;

    UniqueFd(UniqueFd&& other) noexcept : fd_{std::exchange(other.fd_, -1)} {}

    ~UniqueFd() {
        reset();
    }

    [[nodiscard]] int get() const {
        return fd_;
    }

    [[nodiscard]] explicit operator bool() const {
        return fd_ >= 0;
    }

    void reset(int fd = -1) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = fd;
    }

  private:
    int fd_{-1};
};

void write_all(int fd, void const* data, std::size_t size) {
    auto const* ptr = static_cast<char const*>(data);
    while (size > 0) {
        auto written = ::write(fd, ptr, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("write() failed: " + std::string{std::strerror(errno)});
        }
        if (written == 0) {
            fail("write() made no progress");
        }
        ptr += written;
        size -= static_cast<std::size_t>(written);
    }
}

void read_all(int fd, void* data, std::size_t size) {
    auto* ptr = static_cast<char*>(data);
    while (size > 0) {
        auto read = ::read(fd, ptr, size);
        if (read < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("read() failed: " + std::string{std::strerror(errno)});
        }
        if (read == 0) {
            fail("socket closed unexpectedly");
        }
        ptr += read;
        size -= static_cast<std::size_t>(read);
    }
}

template <typename T>
void write_value(int fd, T const& value) {
    write_all(fd, &value, sizeof(T));
}

template <typename T>
T read_value(int fd) {
    T value{};
    read_all(fd, &value, sizeof(T));
    return value;
}

struct Listener {
    UniqueFd fd;
    std::string address;
};

Listener create_listener(int nranks) {
    UniqueFd fd{::socket(AF_INET, SOCK_STREAM, 0)};
    if (!fd) {
        fail("socket() failed: " + std::string{std::strerror(errno)});
    }

    int opt = 1;
    if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        fail("setsockopt() failed: " + std::string{std::strerror(errno)});
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        fail("bind() failed: " + std::string{std::strerror(errno)});
    }
    if (::listen(fd.get(), nranks + 4) != 0) {
        fail("listen() failed: " + std::string{std::strerror(errno)});
    }

    sockaddr_in bound{};
    socklen_t bound_len = sizeof(bound);
    if (::getsockname(fd.get(), reinterpret_cast<sockaddr*>(&bound), &bound_len) != 0) {
        fail("getsockname() failed: " + std::string{std::strerror(errno)});
    }

    std::ostringstream ss;
    ss << "127.0.0.1:" << ntohs(bound.sin_port);
    return {std::move(fd), ss.str()};
}

UniqueFd connect_to(std::string const& address) {
    auto const colon = address.rfind(':');
    if (colon == std::string::npos) {
        fail("invalid control address: " + address);
    }

    auto const host = address.substr(0, colon);
    auto const port = parse_u64(address.substr(colon + 1));
    if (port > std::numeric_limits<std::uint16_t>::max()) {
        fail("invalid control port: " + address);
    }

    UniqueFd fd{::socket(AF_INET, SOCK_STREAM, 0)};
    if (!fd) {
        fail("socket() failed: " + std::string{std::strerror(errno)});
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        fail("invalid control host: " + host);
    }
    if (::connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        fail("connect() failed: " + std::string{std::strerror(errno)});
    }
    return fd;
}

class Control {
  public:
    static Control init(int* argc, char*** argv) {
        if (rapidsmpf::bootstrap::is_running_with_rrun()) {
            auto ctx =
                rapidsmpf::bootstrap::init(rapidsmpf::bootstrap::BackendType::AUTO);
            return Control{std::move(ctx)};
        }

        int provided = 0;
        MPI_CALL(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided));
        if (provided != MPI_THREAD_MULTIPLE) {
            fail("MPI did not provide MPI_THREAD_MULTIPLE");
        }

        int rank = 0;
        int nranks = 0;
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
        return Control{rank, nranks};
    }

    Control(Control const&) = delete;
    Control& operator=(Control const&) = delete;
    Control(Control&&) noexcept = default;
    Control& operator=(Control&&) noexcept = default;

    [[nodiscard]] int rank() const {
        return rank_;
    }

    [[nodiscard]] int nranks() const {
        return nranks_;
    }

    [[nodiscard]] bool is_root() const {
        return rank_ == 0;
    }

    void barrier() const {
        if (mpi_initialized_) {
            MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        } else {
            rapidsmpf::bootstrap::barrier(*bootstrap_ctx_);
        }
    }

    std::vector<char> allgather(
        std::string const& key, void const* local_data, std::size_t local_size
    ) const {
        if (mpi_initialized_) {
            return mpi_allgather(local_data, local_size);
        }
        return rrun_allgather(key, local_data, local_size);
    }

    double max_double(double value, std::string const& key) const {
        if (mpi_initialized_) {
            double result = 0.0;
            MPI_CALL(
                MPI_Reduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD)
            );
            return result;
        }

        auto values = rrun_allgather(key, &value, sizeof(value));
        double result = 0.0;
        for (int i = 0; i < nranks_; ++i) {
            double rank_value = 0.0;
            std::memcpy(
                &rank_value,
                values.data() + static_cast<std::size_t>(i) * sizeof(double),
                sizeof(double)
            );
            result = std::max(result, rank_value);
        }
        return result;
    }

    void finalize() {
        if (mpi_initialized_) {
            MPI_CALL(MPI_Finalize());
            mpi_initialized_ = false;
        }
    }

    [[noreturn]] void abort(int code) const {
        if (mpi_initialized_) {
            MPI_Abort(MPI_COMM_WORLD, code);
        }
        std::exit(code);
    }

  private:
    explicit Control(rapidsmpf::bootstrap::Context ctx)
        : bootstrap_ctx_{std::move(ctx)},
          rank_{static_cast<int>(bootstrap_ctx_->rank)},
          nranks_{static_cast<int>(bootstrap_ctx_->nranks)} {}

    Control(int rank, int nranks)
        : rank_{rank}, nranks_{nranks}, mpi_initialized_{true} {}

    std::vector<char> mpi_allgather(
        void const* local_data, std::size_t local_size
    ) const {
        std::vector<char> result(static_cast<std::size_t>(nranks_) * local_size);
        auto const count = checked_mpi_count(local_size);
        MPI_CALL(MPI_Allgather(
            local_data, count, MPI_BYTE, result.data(), count, MPI_BYTE, MPI_COMM_WORLD
        ));
        return result;
    }

    std::vector<char> rrun_allgather(
        std::string const& key, void const* local_data, std::size_t local_size
    ) const {
        auto const addr_key = key + "_addr";
        auto const data_key = key + "_data";
        std::vector<char> result(static_cast<std::size_t>(nranks_) * local_size);

        std::optional<Listener> listener;
        if (is_root()) {
            listener.emplace(create_listener(nranks_));
            rapidsmpf::bootstrap::put(*bootstrap_ctx_, addr_key, listener->address);
        }
        rapidsmpf::bootstrap::sync(*bootstrap_ctx_);

        if (is_root()) {
            std::memcpy(result.data(), local_data, local_size);
            for (int i = 1; i < nranks_; ++i) {
                sockaddr_in peer_addr{};
                socklen_t peer_addr_len = sizeof(peer_addr);
                UniqueFd conn{::accept(
                    listener->fd.get(),
                    reinterpret_cast<sockaddr*>(&peer_addr),
                    &peer_addr_len
                )};
                if (!conn) {
                    fail("accept() failed: " + std::string{std::strerror(errno)});
                }

                auto const peer_rank = read_value<std::int32_t>(conn.get());
                auto const payload_size = read_value<std::uint64_t>(conn.get());
                if (peer_rank <= 0 || peer_rank >= nranks_) {
                    fail("invalid peer rank in rrun allgather");
                }
                if (payload_size != local_size) {
                    fail("invalid payload size in rrun allgather");
                }
                read_all(
                    conn.get(),
                    result.data() + static_cast<std::size_t>(peer_rank) * local_size,
                    local_size
                );
            }

            rapidsmpf::bootstrap::put(
                *bootstrap_ctx_, data_key, std::string_view{result.data(), result.size()}
            );
        } else {
            auto addr = rapidsmpf::bootstrap::get(*bootstrap_ctx_, addr_key);
            auto conn = connect_to(addr);
            auto const peer_rank = static_cast<std::int32_t>(rank_);
            auto const payload_size = static_cast<std::uint64_t>(local_size);
            write_value(conn.get(), peer_rank);
            write_value(conn.get(), payload_size);
            write_all(conn.get(), local_data, local_size);
        }

        rapidsmpf::bootstrap::sync(*bootstrap_ctx_);
        if (!is_root()) {
            auto gathered = rapidsmpf::bootstrap::get(*bootstrap_ctx_, data_key);
            if (gathered.size() != result.size()) {
                fail("invalid gathered payload size from rrun bootstrap");
            }
            std::memcpy(result.data(), gathered.data(), gathered.size());
        }
        return result;
    }

    std::optional<rapidsmpf::bootstrap::Context> bootstrap_ctx_;
    int rank_{0};
    int nranks_{0};
    bool mpi_initialized_{false};
};

void launch_wait_kernel(int const* release_flag, CUstream stream) {
    wait_for_release<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(release_flag);
    CUDA_RT(cudaGetLastError());
}

using RankPair = std::pair<int, int>;
using PairRound = std::vector<RankPair>;

std::vector<std::size_t> initial_round_robin_slots(std::size_t nranks) {
    if (nranks == 0) {
        return {};
    }

    auto const participant_count = nranks + (nranks % 2);
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

std::vector<PairRound> round_robin_pair_rounds(int nranks) {
    if (nranks < 2) {
        return {};
    }

    auto const rank_count = static_cast<std::size_t>(nranks);
    auto const participant_count = rank_count + (rank_count % 2);
    auto slots = initial_round_robin_slots(rank_count);
    std::vector<PairRound> rounds;
    rounds.reserve(participant_count - 1);

    for (std::size_t round = 0; round < participant_count - 1; ++round) {
        PairRound pairs;
        pairs.reserve(participant_count / 2);
        for (std::size_t slot = 0; slot < participant_count / 2; ++slot) {
            auto const first = slots.at(slot);
            auto const second = slots.at(participant_count - 1 - slot);
            if (first < rank_count && second < rank_count) {
                pairs.emplace_back(static_cast<int>(first), static_cast<int>(second));
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

void enqueue_copy(
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams,
    std::uint64_t op,
    int peer,
    std::uint64_t& copy_index
) {
    auto const peer_index = static_cast<std::size_t>(peer);
    auto const stream = streams.at(stream_index(args, peer, copy_index));
    if (args.copy_mode == CopyMode::Get) {
        auto const src =
            remote_send.at(peer_index) + buffer_offset(args, nranks, op, rank);
        auto const dst = local_recv + buffer_offset(args, nranks, op, peer);
        CUDA_DRV(cuMemcpyDtoDAsync(dst, src, args.message_bytes, stream));
    } else {
        auto const src = local_send + buffer_offset(args, nranks, op, peer);
        auto const dst =
            remote_recv.at(peer_index) + buffer_offset(args, nranks, op, rank);
        CUDA_DRV(cuMemcpyDtoDAsync(dst, src, args.message_bytes, stream));
    }
    ++copy_index;
}

void enqueue_rank_major_copies(
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams
) {
    std::uint64_t copy_index = 0;
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == rank) {
                continue;
            }
            enqueue_copy(
                args,
                rank,
                nranks,
                local_send,
                local_recv,
                remote_send,
                remote_recv,
                streams,
                op,
                peer,
                copy_index
            );
        }
    }
}

void enqueue_transfer(
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams,
    std::uint64_t op,
    int src,
    int dst,
    std::uint64_t& copy_index
) {
    if (args.copy_mode == CopyMode::Get) {
        if (rank != dst) {
            return;
        }
        enqueue_copy(
            args,
            rank,
            nranks,
            local_send,
            local_recv,
            remote_send,
            remote_recv,
            streams,
            op,
            src,
            copy_index
        );
        return;
    }

    if (rank != src) {
        return;
    }
    enqueue_copy(
        args,
        rank,
        nranks,
        local_send,
        local_recv,
        remote_send,
        remote_recv,
        streams,
        op,
        dst,
        copy_index
    );
}

void enqueue_balanced_copies(
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams
) {
    auto const rounds = round_robin_pair_rounds(nranks);
    std::uint64_t copy_index = 0;
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (auto const& round : rounds) {
            for (auto const& pair : round) {
                enqueue_transfer(
                    args,
                    rank,
                    nranks,
                    local_send,
                    local_recv,
                    remote_send,
                    remote_recv,
                    streams,
                    op,
                    pair.first,
                    pair.second,
                    copy_index
                );
            }
            for (auto const& pair : round) {
                enqueue_transfer(
                    args,
                    rank,
                    nranks,
                    local_send,
                    local_recv,
                    remote_send,
                    remote_recv,
                    streams,
                    op,
                    pair.second,
                    pair.first,
                    copy_index
                );
            }
        }
    }
}

void enqueue_all_copies(
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams
) {
    switch (args.post_order) {
    case PostOrder::RankMajor:
        enqueue_rank_major_copies(
            args, rank, nranks, local_send, local_recv, remote_send, remote_recv, streams
        );
        break;
    case PostOrder::Balanced:
        enqueue_balanced_copies(
            args, rank, nranks, local_send, local_recv, remote_send, remote_recv, streams
        );
        break;
    }
}

double run_once(
    Control const& control,
    Args const& args,
    int rank,
    int nranks,
    CUdeviceptr local_send,
    CUdeviceptr local_recv,
    std::vector<CUdeviceptr> const& remote_send,
    std::vector<CUdeviceptr> const& remote_recv,
    std::vector<CUstream> const& streams,
    std::vector<std::size_t> const& active_streams,
    CUstream release_stream,
    int* release_flag
) {
    CUDA_RT(cudaMemset(release_flag, 0, sizeof(int)));
    CUDA_RT(cudaDeviceSynchronize());
    control.barrier();

    auto const t0 = std::chrono::steady_clock::now();
    if (args.gate_launch) {
        for (auto index : active_streams) {
            launch_wait_kernel(release_flag, streams.at(index));
        }
    }

    enqueue_all_copies(
        args, rank, nranks, local_send, local_recv, remote_send, remote_recv, streams
    );

    if (args.gate_launch) {
        int const one = 1;
        CUDA_DRV(cuMemcpyHtoDAsync(
            reinterpret_cast<CUdeviceptr>(release_flag), &one, sizeof(one), release_stream
        ));
    }

    for (auto index : active_streams) {
        CUDA_DRV(cuStreamSynchronize(streams.at(index)));
    }
    CUDA_DRV(cuStreamSynchronize(release_stream));
    auto const t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void print_config(Args const& args, int nranks) {
    std::cout << "Arguments:\n"
              << "  --bytes " << args.message_bytes << "\n"
              << "  --ops " << args.num_ops << "\n"
              << "  --runs " << args.num_runs << "\n"
              << "  --warmups " << args.num_warmups << "\n"
              << "  --mode " << to_string(args.copy_mode) << "\n"
              << "  --streams " << to_string(args.stream_mode) << "\n"
              << "  --post-order " << to_string(args.post_order) << "\n"
              << "  gate_launch " << (args.gate_launch ? "true" : "false") << "\n"
              << "  nranks " << nranks << "\n";
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

}  // namespace

int main(int argc, char** argv) {
    std::optional<Control> control;
    try {
        control.emplace(Control::init(&argc, &argv));
        auto const rank = control->rank();
        auto const nranks = control->nranks();
        if (nranks < 2) {
            fail("bench_cuda_ipc requires at least two MPI ranks");
        }

        auto const args = parse_args(argc, argv);

        CUDA_DRV(cuInit(0));
        int device_count = 0;
        CUDA_RT(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            fail("no visible CUDA devices");
        }
        auto const device = args.device >= 0 ? args.device : rank % device_count;
        if (device >= device_count) {
            fail("selected CUDA device is not visible");
        }
        CUDA_RT(cudaSetDevice(device));
        CUDA_RT(cudaFree(nullptr));

        auto const slots_per_allocation = checked_mul(
            static_cast<std::size_t>(args.num_ops), static_cast<std::size_t>(nranks)
        );
        auto const allocation_bytes =
            checked_mul(args.message_bytes, slots_per_allocation);
        void* send_ptr = nullptr;
        void* recv_ptr = nullptr;
        int* release_flag = nullptr;
        CUDA_RT(cudaMalloc(&send_ptr, allocation_bytes));
        CUDA_RT(cudaMalloc(&recv_ptr, allocation_bytes));
        CUDA_RT(cudaMalloc(&release_flag, sizeof(int)));

        auto const local_send = reinterpret_cast<CUdeviceptr>(send_ptr);
        auto const local_recv = reinterpret_cast<CUdeviceptr>(recv_ptr);
        CUDA_DRV(cuMemsetD8(local_send, 0xa5, allocation_bytes));
        CUDA_DRV(cuMemsetD8(local_recv, 0, allocation_bytes));
        CUDA_RT(cudaDeviceSynchronize());

        CUipcMemHandle local_send_handle{};
        CUipcMemHandle local_recv_handle{};
        CUDA_DRV(cuIpcGetMemHandle(&local_send_handle, local_send));
        CUDA_DRV(cuIpcGetMemHandle(&local_recv_handle, local_recv));

        std::vector<CUipcMemHandle> send_handles(static_cast<std::size_t>(nranks));
        std::vector<CUipcMemHandle> recv_handles(static_cast<std::size_t>(nranks));
        auto const send_handle_bytes = control->allgather(
            "bench_cuda_ipc_send_handles", &local_send_handle, sizeof(CUipcMemHandle)
        );
        auto const recv_handle_bytes = control->allgather(
            "bench_cuda_ipc_recv_handles", &local_recv_handle, sizeof(CUipcMemHandle)
        );
        std::memcpy(
            send_handles.data(), send_handle_bytes.data(), send_handle_bytes.size()
        );
        std::memcpy(
            recv_handles.data(), recv_handle_bytes.data(), recv_handle_bytes.size()
        );

        std::vector<CUdeviceptr> remote_send(static_cast<std::size_t>(nranks), 0);
        std::vector<CUdeviceptr> remote_recv(static_cast<std::size_t>(nranks), 0);
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == rank) {
                continue;
            }
            auto const peer_index = static_cast<std::size_t>(peer);
            if (args.copy_mode == CopyMode::Get) {
                CUDA_DRV(cuIpcOpenMemHandle(
                    &remote_send.at(peer_index),
                    send_handles.at(peer_index),
                    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
                ));
            } else {
                CUDA_DRV(cuIpcOpenMemHandle(
                    &remote_recv.at(peer_index),
                    recv_handles.at(peer_index),
                    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
                ));
            }
        }

        std::vector<CUstream> streams(stream_count(args, nranks));
        for (auto& stream : streams) {
            CUDA_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        }
        CUstream release_stream = nullptr;
        CUDA_DRV(cuStreamCreate(&release_stream, CU_STREAM_NON_BLOCKING));
        auto const active_streams = active_stream_indices(args, rank, nranks);

        std::array<char, 128> local_info{};
        std::array<char, 32> pci_bus_id{};
        CUDA_RT(cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), device));
        std::snprintf(
            local_info.data(),
            local_info.size(),
            "rank %d: cuda device %d pci %s",
            rank,
            device,
            pci_bus_id.data()
        );
        auto const all_info = control->allgather(
            "bench_cuda_ipc_rank_info", local_info.data(), local_info.size()
        );
        if (control->is_root()) {
            print_config(args, nranks);
            for (int i = 0; i < nranks; ++i) {
                std::cout << "  "
                          << all_info.data()
                                 + static_cast<std::size_t>(i) * local_info.size()
                          << "\n";
            }
        }

        auto const local_bytes = checked_mul(
            checked_mul(args.message_bytes, static_cast<std::size_t>(args.num_ops)),
            static_cast<std::size_t>(nranks - 1)
        );
        auto const aggregate_bytes =
            checked_mul(local_bytes, static_cast<std::size_t>(nranks));
        for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
            auto const local_elapsed = run_once(
                *control,
                args,
                rank,
                nranks,
                local_send,
                local_recv,
                remote_send,
                remote_recv,
                streams,
                active_streams,
                release_stream,
                release_flag
            );
            auto const elapsed = control->max_double(
                local_elapsed, "bench_cuda_ipc_elapsed_" + std::to_string(i)
            );
            bool const warmup = i < args.num_warmups;
            if (control->is_root()) {
                print_rate(elapsed, aggregate_bytes, warmup);
            }
        }

        for (auto stream : streams) {
            CUDA_DRV(cuStreamDestroy(stream));
        }
        CUDA_DRV(cuStreamDestroy(release_stream));
        for (int peer = 0; peer < nranks; ++peer) {
            auto const peer_index = static_cast<std::size_t>(peer);
            if (remote_send.at(peer_index) != 0) {
                CUDA_DRV(cuIpcCloseMemHandle(remote_send.at(peer_index)));
            }
            if (remote_recv.at(peer_index) != 0) {
                CUDA_DRV(cuIpcCloseMemHandle(remote_recv.at(peer_index)));
            }
        }
        CUDA_RT(cudaFree(release_flag));
        CUDA_RT(cudaFree(recv_ptr));
        CUDA_RT(cudaFree(send_ptr));
        control->finalize();
        return 0;
    } catch (std::exception const& e) {
        auto const rank = control.has_value() ? control->rank() : 0;
        std::cerr << "rank " << rank << ": " << e.what() << "\n";
        if (control.has_value()) {
            control->abort(1);
        }
        return 1;
    }
}
