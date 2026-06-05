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
#include <ucp/api/ucp.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>

namespace {

enum class PostOrder : std::uint8_t {
    RankMajor,
    Balanced,
    BalancedInterleaved
};

enum class RmaMode : std::uint8_t {
    Get,
    Put
};

enum class CompletionMode : std::uint8_t {
    Ordered,
    Unordered
};

enum class ProgressDuringPost : std::uint8_t {
    None,
    Request,
    Direction,
    Round,
    Operation
};

enum class MemoryTypeHint : std::uint8_t {
    Unknown,
    Cuda
};

struct Args {
    std::size_t message_bytes{1 << 20};
    std::uint64_t num_ops{1};
    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{1};
    int device{-1};
    RmaMode rma_mode{RmaMode::Put};
    PostOrder post_order{PostOrder::RankMajor};
    CompletionMode completion_mode{CompletionMode::Ordered};
    ProgressDuringPost progress_during_post{ProgressDuringPost::None};
    MemoryTypeHint memory_type_hint{MemoryTypeHint::Unknown};
    bool no_imm_completion{true};
    bool multi_send{false};
    bool use_local_memh{true};
    bool flush_put{true};
};

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

void check_ucs(ucs_status_t status, char const* expr, char const* file, int line) {
    if (status == UCS_OK) {
        return;
    }
    std::ostringstream ss;
    ss << file << ":" << line << ": " << expr << " failed: " << ucs_status_string(status);
    fail(ss.str());
}

#define CUDA_RT(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUDA_DRV(call) check_cu((call), #call, __FILE__, __LINE__)
#define MPI_CALL(call) check_mpi((call), #call, __FILE__, __LINE__)
#define UCS_CALL(call) check_ucs((call), #call, __FILE__, __LINE__)

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
        << "  -w, --warmups <num>      Warmup runs using same buffers (default: 1)\n"
        << "  --mode <get|put>         RMA operation mode (default: put)\n"
        << "  -P, --post-order <rank-major|balanced|balanced-interleaved>\n"
        << "                           Request posting order (default: rank-major)\n"
        << "  -W, --completion-mode <ordered|unordered>\n"
        << "                           Completion polling order (default: ordered)\n"
        << "  -G, --progress-during-post <none|request|direction|round|operation>\n"
        << "                           UCP progress while posting (default: none)\n"
        << "  --memory-type <unknown|cuda>\n"
        << "                           UCP memory type hint (default: unknown)\n"
        << "  --allow-imm-completion   Do not set UCP_OP_ATTR_FLAG_NO_IMM_CMPL\n"
        << "  --multi-send             Set UCP_OP_ATTR_FLAG_MULTI_SEND\n"
        << "  --no-local-memh          Do not pass local memh in RMA request params\n"
        << "  --no-flush               Do not flush after PUT requests complete\n"
        << "  --device <num>           CUDA device for this rank (default: rank % "
           "visible devices)\n"
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
                args.rma_mode = RmaMode::Get;
            } else if (mode == "put") {
                args.rma_mode = RmaMode::Put;
            } else {
                fail("invalid --mode: " + mode);
            }
        } else if (arg == "-P" || arg == "--post-order" || arg == "--post-mode") {
            auto const order = lower(next_arg(i, argc, argv, arg));
            if (order == "rank-major" || order == "device-major" || order == "major") {
                args.post_order = PostOrder::RankMajor;
            } else if (order == "balanced" || order == "round-robin") {
                args.post_order = PostOrder::Balanced;
            } else if (order == "balanced-interleaved" || order == "interleaved") {
                args.post_order = PostOrder::BalancedInterleaved;
            } else {
                fail("invalid --post-order: " + order);
            }
        } else if (arg == "-W" || arg == "--completion-mode" || arg == "--wait-mode") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "ordered" || mode == "vector") {
                args.completion_mode = CompletionMode::Ordered;
            } else if (mode == "unordered" || mode == "map") {
                args.completion_mode = CompletionMode::Unordered;
            } else {
                fail("invalid --completion-mode: " + mode);
            }
        } else if (arg == "-G" || arg == "--progress-during-post") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "none" || mode == "off" || mode == "disabled") {
                args.progress_during_post = ProgressDuringPost::None;
            } else if (mode == "request" || mode == "copy" || mode == "post") {
                args.progress_during_post = ProgressDuringPost::Request;
            } else if (mode == "direction") {
                args.progress_during_post = ProgressDuringPost::Direction;
            } else if (mode == "round") {
                args.progress_during_post = ProgressDuringPost::Round;
            } else if (mode == "operation" || mode == "op") {
                args.progress_during_post = ProgressDuringPost::Operation;
            } else {
                fail("invalid --progress-during-post: " + mode);
            }
        } else if (arg == "--memory-type") {
            auto const mode = lower(next_arg(i, argc, argv, arg));
            if (mode == "unknown") {
                args.memory_type_hint = MemoryTypeHint::Unknown;
            } else if (mode == "cuda" || mode == "device") {
                args.memory_type_hint = MemoryTypeHint::Cuda;
            } else {
                fail("invalid --memory-type: " + mode);
            }
        } else if (arg == "--allow-imm-completion") {
            args.no_imm_completion = false;
        } else if (arg == "--multi-send") {
            args.multi_send = true;
        } else if (arg == "--no-local-memh") {
            args.use_local_memh = false;
        } else if (arg == "--no-flush") {
            args.flush_put = false;
        } else if (arg == "--device") {
            auto const value = parse_u64(next_arg(i, argc, argv, arg));
            if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                fail("device index exceeds int range");
            }
            args.device = static_cast<int>(value);
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

std::string to_string(RmaMode mode) {
    switch (mode) {
    case RmaMode::Get:
        return "get";
    case RmaMode::Put:
        return "put";
    }
    fail("unknown RMA mode");
}

std::string to_string(PostOrder order) {
    switch (order) {
    case PostOrder::RankMajor:
        return "rank-major";
    case PostOrder::Balanced:
        return "balanced";
    case PostOrder::BalancedInterleaved:
        return "balanced-interleaved";
    }
    fail("unknown post order");
}

std::string to_string(CompletionMode mode) {
    switch (mode) {
    case CompletionMode::Ordered:
        return "ordered";
    case CompletionMode::Unordered:
        return "unordered";
    }
    fail("unknown completion mode");
}

std::string to_string(ProgressDuringPost mode) {
    switch (mode) {
    case ProgressDuringPost::None:
        return "none";
    case ProgressDuringPost::Request:
        return "request";
    case ProgressDuringPost::Direction:
        return "direction";
    case ProgressDuringPost::Round:
        return "round";
    case ProgressDuringPost::Operation:
        return "operation";
    }
    fail("unknown progress-during-post mode");
}

std::string to_string(MemoryTypeHint hint) {
    switch (hint) {
    case MemoryTypeHint::Unknown:
        return "unknown";
    case MemoryTypeHint::Cuda:
        return "cuda";
    }
    fail("unknown memory type hint");
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

struct UcpEndpoint {
    ucp_ep_h ep{nullptr};
};

class UcpResources {
  public:
    UcpResources(int rank, int nranks) {
        ucp_config_t* config = nullptr;
        UCS_CALL(ucp_config_read(nullptr, nullptr, &config));

        ucp_params_t params{};
        params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        params.features = UCP_FEATURE_RMA;
        params.estimated_num_eps = static_cast<std::size_t>(nranks);
        auto const status = ucp_init(&params, config, &context_);
        ucp_config_release(config);
        UCS_CALL(status);

        ucp_worker_params_t worker_params{};
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
        UCS_CALL(ucp_worker_create(context_, &worker_params, &worker_));

        endpoints_.resize(static_cast<std::size_t>(nranks));
        rank_ = rank;
    }

    UcpResources(UcpResources const&) = delete;
    UcpResources& operator=(UcpResources const&) = delete;

    ~UcpResources() {
        close_endpoints();
        if (worker_ != nullptr) {
            ucp_worker_destroy(worker_);
            worker_ = nullptr;
        }
        if (context_ != nullptr) {
            ucp_cleanup(context_);
            context_ = nullptr;
        }
    }

    void close_endpoints() {
        ucp_request_param_t close_params{};
        for (auto& endpoint : endpoints_) {
            if (endpoint.ep != nullptr) {
                auto* request = ucp_ep_close_nbx(endpoint.ep, &close_params);
                if (!UCS_PTR_IS_ERR(request) && request != nullptr) {
                    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
                        progress();
                    }
                    ucp_request_free(request);
                }
                endpoint.ep = nullptr;
            }
        }
    }

    [[nodiscard]] ucp_worker_h worker() const {
        return worker_;
    }

    [[nodiscard]] ucp_context_h context() const {
        return context_;
    }

    [[nodiscard]] ucp_ep_h endpoint(int peer) const {
        return endpoints_.at(static_cast<std::size_t>(peer)).ep;
    }

    void progress() const {
        static_cast<void>(ucp_worker_progress(worker_));
    }

    void flush() const {
        ucp_request_param_t params{};
        auto* request = ucp_worker_flush_nbx(worker_, &params);
        if (UCS_PTR_IS_ERR(request)) {
            fail(
                "ucp_worker_flush_nbx failed: "
                + std::string{ucs_status_string(UCS_PTR_STATUS(request))}
            );
        }
        if (request == nullptr) {
            return;
        }
        while (ucp_request_check_status(request) == UCS_INPROGRESS) {
            progress();
        }
        auto const status = ucp_request_check_status(request);
        ucp_request_free(request);
        if (status != UCS_OK) {
            fail("UCP flush failed: " + std::string{ucs_status_string(status)});
        }
    }

    std::vector<char> local_address() const {
        ucp_address_t* address = nullptr;
        std::size_t address_length = 0;
        UCS_CALL(ucp_worker_get_address(worker_, &address, &address_length));
        std::vector<char> result(address_length);
        std::memcpy(result.data(), address, address_length);
        ucp_worker_release_address(worker_, address);
        return result;
    }

    void connect(int peer, char const* address) {
        if (peer == rank_) {
            return;
        }
        ucp_ep_params_t params{};
        params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        params.address = reinterpret_cast<ucp_address_t const*>(address);
        UCS_CALL(ucp_ep_create(
            worker_, &params, &endpoints_.at(static_cast<std::size_t>(peer)).ep
        ));
    }

  private:
    int rank_{0};
    ucp_context_h context_{nullptr};
    ucp_worker_h worker_{nullptr};
    std::vector<UcpEndpoint> endpoints_;
};

std::vector<std::uint64_t> gather_u64_vector(
    Control const& control, std::string const& key, std::uint64_t local_value
) {
    auto const bytes = control.allgather(key, &local_value, sizeof(local_value));
    std::vector<std::uint64_t> values(static_cast<std::size_t>(control.nranks()));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

void connect_ucp_endpoints(Control const& control, UcpResources& ucp) {
    auto const local_address = ucp.local_address();
    auto const local_size = static_cast<std::uint64_t>(local_address.size());
    auto const sizes = gather_u64_vector(control, "bench_ucp_rma_addr_sizes", local_size);
    auto const max_size = *std::ranges::max_element(sizes);
    if (max_size == 0
        || max_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        fail("invalid UCP worker address size");
    }

    std::vector<char> padded(static_cast<std::size_t>(max_size), '\0');
    std::memcpy(padded.data(), local_address.data(), local_address.size());
    auto const all_addresses =
        control.allgather("bench_ucp_rma_addresses", padded.data(), padded.size());

    for (int peer = 0; peer < control.nranks(); ++peer) {
        if (peer == control.rank()) {
            continue;
        }
        auto const offset = static_cast<std::size_t>(peer) * padded.size();
        ucp.connect(peer, all_addresses.data() + offset);
    }
    control.barrier();
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

struct PendingRequest {
    void* request{nullptr};
};

struct RemoteMemory {
    std::uint64_t base_address{0};
    ucp_rkey_h rkey{nullptr};
};

struct PackedBuffers {
    std::vector<std::uint64_t> sizes;
    std::size_t stride{0};
    std::vector<char> buffers;
};

std::size_t checked_to_size(std::uint64_t value, char const* description) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        fail(std::string{description} + " exceeds size_t range");
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t pointer_to_u64(void const* ptr) {
    static_assert(sizeof(std::uintptr_t) <= sizeof(std::uint64_t));
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ptr));
}

std::uint64_t add_remote_offset(std::uint64_t base, std::size_t offset) {
    if (offset > std::numeric_limits<std::uint64_t>::max() - base) {
        fail("remote address overflow");
    }
    return base + static_cast<std::uint64_t>(offset);
}

ucp_mem_h map_memory(
    UcpResources const& ucp, Args const& args, void* ptr, std::size_t bytes
) {
    ucp_mem_map_params_t params{};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = ptr;
    params.length = bytes;
    if (args.memory_type_hint == MemoryTypeHint::Cuda) {
        params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.memory_type = UCS_MEMORY_TYPE_CUDA;
    }

    ucp_mem_h memh = nullptr;
    UCS_CALL(ucp_mem_map(ucp.context(), &params, &memh));
    return memh;
}

void unmap_memory(UcpResources const& ucp, ucp_mem_h& memh) {
    if (memh != nullptr) {
        UCS_CALL(ucp_mem_unmap(ucp.context(), memh));
        memh = nullptr;
    }
}

std::vector<char> pack_remote_key(UcpResources const& ucp, ucp_mem_h memh) {
    void* buffer = nullptr;
    std::size_t size = 0;
    UCS_CALL(ucp_rkey_pack(ucp.context(), memh, &buffer, &size));
    if (buffer == nullptr || size == 0) {
        fail("ucp_rkey_pack returned an empty remote key");
    }

    std::vector<char> result(size);
    std::memcpy(result.data(), buffer, size);
    ucp_rkey_buffer_release(buffer);
    return result;
}

PackedBuffers gather_packed_buffers(
    Control const& control, std::string const& key, std::vector<char> const& local_buffer
) {
    auto const local_size = static_cast<std::uint64_t>(local_buffer.size());
    auto sizes = gather_u64_vector(control, key + "_sizes", local_size);
    auto const max_size = *std::ranges::max_element(sizes);
    if (max_size == 0) {
        fail("empty packed remote key");
    }

    auto const stride = checked_to_size(max_size, "packed remote key size");
    std::vector<char> padded(stride, '\0');
    std::memcpy(padded.data(), local_buffer.data(), local_buffer.size());
    auto buffers = control.allgather(key + "_buffers", padded.data(), padded.size());
    return {std::move(sizes), stride, std::move(buffers)};
}

std::vector<RemoteMemory> unpack_remote_memory(
    Control const& control,
    UcpResources const& ucp,
    std::vector<std::uint64_t> const& base_addresses,
    PackedBuffers const& packed,
    char const* description
) {
    auto const nranks = control.nranks();
    std::vector<RemoteMemory> remotes(static_cast<std::size_t>(nranks));
    if (base_addresses.size() != remotes.size() || packed.sizes.size() != remotes.size())
    {
        fail(std::string{"invalid gathered "} + description + " metadata");
    }

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == control.rank()) {
            continue;
        }

        auto const peer_index = static_cast<std::size_t>(peer);
        auto const size =
            checked_to_size(packed.sizes.at(peer_index), "packed remote key size");
        if (size == 0 || size > packed.stride) {
            fail(std::string{"invalid packed "} + description + " remote key size");
        }

        auto const offset = checked_mul(peer_index, packed.stride);
        ucp_rkey_h rkey = nullptr;
        UCS_CALL(
            ucp_ep_rkey_unpack(ucp.endpoint(peer), packed.buffers.data() + offset, &rkey)
        );
        remotes.at(peer_index) = {base_addresses.at(peer_index), rkey};
    }
    return remotes;
}

void destroy_remote_memory(std::vector<RemoteMemory>& remotes) {
    for (auto& remote : remotes) {
        if (remote.rkey != nullptr) {
            ucp_rkey_destroy(remote.rkey);
            remote.rkey = nullptr;
        }
    }
}

ucp_request_param_t request_params(Args const& args, ucp_mem_h local_memh) {
    ucp_request_param_t params{};
    if (args.no_imm_completion) {
        params.op_attr_mask |= UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    }
    if (args.multi_send) {
        params.op_attr_mask |= UCP_OP_ATTR_FLAG_MULTI_SEND;
    }
    if (args.memory_type_hint == MemoryTypeHint::Cuda) {
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
        params.memory_type = UCS_MEMORY_TYPE_CUDA;
    }
    if (args.use_local_memh && local_memh != nullptr) {
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        params.memh = local_memh;
    }
    return params;
}

void add_request(
    std::vector<PendingRequest>& requests,
    ucs_status_ptr_t status_ptr,
    char const* operation
) {
    if (UCS_PTR_IS_ERR(status_ptr)) {
        std::ostringstream ss;
        ss << operation << " failed: " << ucs_status_string(UCS_PTR_STATUS(status_ptr));
        fail(ss.str());
    }
    if (status_ptr != nullptr) {
        requests.push_back({status_ptr});
    }
}

void progress_after(UcpResources& ucp, Args const& args, ProgressDuringPost point) {
    if (args.progress_during_post == point) {
        ucp.progress();
    }
}

void post_get(
    UcpResources& ucp,
    Args const& args,
    int nranks,
    std::byte* recv_base,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<PendingRequest>& requests,
    std::uint64_t op,
    int src,
    int dst
) {
    auto params = request_params(args, recv_memh);
    auto* buffer = recv_base + buffer_offset(args, nranks, op, src);
    auto const remote_address = add_remote_offset(
        remote_send.at(static_cast<std::size_t>(src)).base_address,
        buffer_offset(args, nranks, op, dst)
    );
    auto* request = ucp_get_nbx(
        ucp.endpoint(src),
        buffer,
        args.message_bytes,
        remote_address,
        remote_send.at(static_cast<std::size_t>(src)).rkey,
        &params
    );
    add_request(requests, request, "ucp_get_nbx");
    progress_after(ucp, args, ProgressDuringPost::Request);
}

void post_put(
    UcpResources& ucp,
    Args const& args,
    int nranks,
    std::byte* send_base,
    ucp_mem_h send_memh,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests,
    std::uint64_t op,
    int src,
    int dst
) {
    auto params = request_params(args, send_memh);
    auto const* buffer = send_base + buffer_offset(args, nranks, op, dst);
    auto const remote_address = add_remote_offset(
        remote_recv.at(static_cast<std::size_t>(dst)).base_address,
        buffer_offset(args, nranks, op, src)
    );
    auto* request = ucp_put_nbx(
        ucp.endpoint(dst),
        buffer,
        args.message_bytes,
        remote_address,
        remote_recv.at(static_cast<std::size_t>(dst)).rkey,
        &params
    );
    add_request(requests, request, "ucp_put_nbx");
    progress_after(ucp, args, ProgressDuringPost::Request);
}

void post_transfer(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests,
    std::uint64_t op,
    int src,
    int dst
) {
    if (src == dst) {
        return;
    }

    switch (args.rma_mode) {
    case RmaMode::Get:
        if (rank == dst) {
            post_get(
                ucp,
                args,
                nranks,
                recv_base,
                recv_memh,
                remote_send,
                requests,
                op,
                src,
                dst
            );
        }
        break;
    case RmaMode::Put:
        if (rank == src) {
            post_put(
                ucp,
                args,
                nranks,
                send_base,
                send_memh,
                remote_recv,
                requests,
                op,
                src,
                dst
            );
        }
        break;
    }
}

void post_rank_major(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests
) {
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == rank) {
                continue;
            }
            auto const src = args.rma_mode == RmaMode::Get ? peer : rank;
            auto const dst = args.rma_mode == RmaMode::Get ? rank : peer;
            post_transfer(
                ucp,
                args,
                rank,
                nranks,
                send_base,
                recv_base,
                send_memh,
                recv_memh,
                remote_send,
                remote_recv,
                requests,
                op,
                src,
                dst
            );
        }
        progress_after(ucp, args, ProgressDuringPost::Direction);
        progress_after(ucp, args, ProgressDuringPost::Operation);
    }
}

void post_balanced_direction(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests,
    std::uint64_t op,
    PairRound const& round,
    std::uint64_t direction
) {
    for (auto const& [first, second] : round) {
        auto const src = direction == 0 ? first : second;
        auto const dst = direction == 0 ? second : first;
        post_transfer(
            ucp,
            args,
            rank,
            nranks,
            send_base,
            recv_base,
            send_memh,
            recv_memh,
            remote_send,
            remote_recv,
            requests,
            op,
            src,
            dst
        );
    }
}

void post_balanced(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests
) {
    auto const rounds = round_robin_pair_rounds(nranks);
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (auto const& round : rounds) {
            post_balanced_direction(
                ucp,
                args,
                rank,
                nranks,
                send_base,
                recv_base,
                send_memh,
                recv_memh,
                remote_send,
                remote_recv,
                requests,
                op,
                round,
                0
            );
            progress_after(ucp, args, ProgressDuringPost::Direction);
            post_balanced_direction(
                ucp,
                args,
                rank,
                nranks,
                send_base,
                recv_base,
                send_memh,
                recv_memh,
                remote_send,
                remote_recv,
                requests,
                op,
                round,
                1
            );
            progress_after(ucp, args, ProgressDuringPost::Direction);
            progress_after(ucp, args, ProgressDuringPost::Round);
        }
        progress_after(ucp, args, ProgressDuringPost::Operation);
    }
}

void post_balanced_interleaved(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests
) {
    auto const rounds = round_robin_pair_rounds(nranks);
    for (std::uint64_t op = 0; op < args.num_ops; ++op) {
        for (std::size_t round_idx = 0; round_idx < rounds.size(); ++round_idx) {
            for (std::uint64_t direction = 0; direction < 2; ++direction) {
                post_balanced_direction(
                    ucp,
                    args,
                    rank,
                    nranks,
                    send_base,
                    recv_base,
                    send_memh,
                    recv_memh,
                    remote_send,
                    remote_recv,
                    requests,
                    op,
                    rounds.at(round_idx),
                    direction
                );
                progress_after(ucp, args, ProgressDuringPost::Direction);
            }
            progress_after(ucp, args, ProgressDuringPost::Round);
        }
        progress_after(ucp, args, ProgressDuringPost::Operation);
    }
}

void post_all(
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv,
    std::vector<PendingRequest>& requests
) {
    switch (args.post_order) {
    case PostOrder::RankMajor:
        post_rank_major(
            ucp,
            args,
            rank,
            nranks,
            send_base,
            recv_base,
            send_memh,
            recv_memh,
            remote_send,
            remote_recv,
            requests
        );
        break;
    case PostOrder::Balanced:
        post_balanced(
            ucp,
            args,
            rank,
            nranks,
            send_base,
            recv_base,
            send_memh,
            recv_memh,
            remote_send,
            remote_recv,
            requests
        );
        break;
    case PostOrder::BalancedInterleaved:
        post_balanced_interleaved(
            ucp,
            args,
            rank,
            nranks,
            send_base,
            recv_base,
            send_memh,
            recv_memh,
            remote_send,
            remote_recv,
            requests
        );
        break;
    }
}

bool complete_request(PendingRequest& request) {
    auto const status = ucp_request_check_status(request.request);
    if (status == UCS_INPROGRESS) {
        return false;
    }
    if (status != UCS_OK) {
        fail("UCP request failed: " + std::string{ucs_status_string(status)});
    }
    ucp_request_free(request.request);
    request.request = nullptr;
    return true;
}

void wait_ordered(UcpResources& ucp, std::vector<PendingRequest>& requests) {
    while (!requests.empty()) {
        ucp.progress();
        std::size_t completed = 0;
        while (completed < requests.size() && complete_request(requests.at(completed))) {
            ++completed;
        }
        if (completed > 0) {
            requests.erase(
                requests.begin(),
                requests.begin() + static_cast<std::ptrdiff_t>(completed)
            );
        }
    }
}

void wait_unordered(UcpResources& ucp, std::vector<PendingRequest>& requests) {
    while (!requests.empty()) {
        ucp.progress();
        for (std::size_t i = 0; i < requests.size();) {
            if (complete_request(requests.at(i))) {
                requests.at(i) = requests.back();
                requests.pop_back();
            } else {
                ++i;
            }
        }
    }
}

void wait_all(
    UcpResources& ucp, Args const& args, std::vector<PendingRequest>& requests
) {
    switch (args.completion_mode) {
    case CompletionMode::Ordered:
        wait_ordered(ucp, requests);
        break;
    case CompletionMode::Unordered:
        wait_unordered(ucp, requests);
        break;
    }
}

double run_once(
    Control const& control,
    UcpResources& ucp,
    Args const& args,
    int rank,
    int nranks,
    std::byte* send_base,
    std::byte* recv_base,
    ucp_mem_h send_memh,
    ucp_mem_h recv_memh,
    std::vector<RemoteMemory> const& remote_send,
    std::vector<RemoteMemory> const& remote_recv
) {
    CUDA_RT(cudaDeviceSynchronize());
    control.barrier();

    auto const t0 = std::chrono::steady_clock::now();
    std::vector<PendingRequest> requests;
    requests.reserve(checked_mul(
        checked_to_size(args.num_ops, "number of operations"),
        static_cast<std::size_t>(nranks - 1)
    ));
    post_all(
        ucp,
        args,
        rank,
        nranks,
        send_base,
        recv_base,
        send_memh,
        recv_memh,
        remote_send,
        remote_recv,
        requests
    );
    wait_all(ucp, args, requests);
    if (args.rma_mode == RmaMode::Put && args.flush_put) {
        ucp.flush();
    }
    auto const t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void print_config(Args const& args, int nranks) {
    std::cout << "Benchmark: raw UCP RMA CUDA all-to-all\n"
              << "Arguments:\n"
              << "  --bytes " << args.message_bytes << "\n"
              << "  --ops " << args.num_ops << "\n"
              << "  --runs " << args.num_runs << "\n"
              << "  --warmups " << args.num_warmups << "\n"
              << "  --mode " << to_string(args.rma_mode) << "\n"
              << "  --post-order " << to_string(args.post_order) << "\n"
              << "  --completion-mode " << to_string(args.completion_mode) << "\n"
              << "  --progress-during-post " << to_string(args.progress_during_post)
              << "\n"
              << "  --memory-type " << to_string(args.memory_type_hint) << "\n"
              << "  no_imm_completion " << (args.no_imm_completion ? "true" : "false")
              << "\n"
              << "  multi_send " << (args.multi_send ? "true" : "false") << "\n"
              << "  use_local_memh " << (args.use_local_memh ? "true" : "false") << "\n"
              << "  flush_put " << (args.flush_put ? "true" : "false") << "\n"
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
            fail("bench_ucp_rma requires at least two ranks");
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
        CUDA_RT(cudaMalloc(&send_ptr, allocation_bytes));
        CUDA_RT(cudaMalloc(&recv_ptr, allocation_bytes));
        CUDA_DRV(
            cuMemsetD8(reinterpret_cast<CUdeviceptr>(send_ptr), 0xa5, allocation_bytes)
        );
        CUDA_DRV(
            cuMemsetD8(reinterpret_cast<CUdeviceptr>(recv_ptr), 0, allocation_bytes)
        );
        CUDA_RT(cudaDeviceSynchronize());

        UcpResources ucp{rank, nranks};
        connect_ucp_endpoints(*control, ucp);

        auto send_memh = map_memory(ucp, args, send_ptr, allocation_bytes);
        auto recv_memh = map_memory(ucp, args, recv_ptr, allocation_bytes);
        auto const packed_send_rkey = pack_remote_key(ucp, send_memh);
        auto const packed_recv_rkey = pack_remote_key(ucp, recv_memh);
        auto const send_base_addresses = gather_u64_vector(
            *control, "bench_ucp_rma_send_base_addresses", pointer_to_u64(send_ptr)
        );
        auto const recv_base_addresses = gather_u64_vector(
            *control, "bench_ucp_rma_recv_base_addresses", pointer_to_u64(recv_ptr)
        );
        auto const packed_send_rkeys =
            gather_packed_buffers(*control, "bench_ucp_rma_send_rkeys", packed_send_rkey);
        auto const packed_recv_rkeys =
            gather_packed_buffers(*control, "bench_ucp_rma_recv_rkeys", packed_recv_rkey);
        auto remote_send = unpack_remote_memory(
            *control, ucp, send_base_addresses, packed_send_rkeys, "send"
        );
        auto remote_recv = unpack_remote_memory(
            *control, ucp, recv_base_addresses, packed_recv_rkeys, "recv"
        );
        control->barrier();

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
            "bench_ucp_rma_rank_info", local_info.data(), local_info.size()
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
        auto* send_base = static_cast<std::byte*>(send_ptr);
        auto* recv_base = static_cast<std::byte*>(recv_ptr);
        for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
            auto const local_elapsed = run_once(
                *control,
                ucp,
                args,
                rank,
                nranks,
                send_base,
                recv_base,
                send_memh,
                recv_memh,
                remote_send,
                remote_recv
            );
            auto const elapsed = control->max_double(
                local_elapsed, "bench_ucp_rma_elapsed_" + std::to_string(i)
            );
            bool const warmup = i < args.num_warmups;
            if (control->is_root()) {
                print_rate(elapsed, aggregate_bytes, warmup);
            }
            control->barrier();
        }

        ucp.flush();
        destroy_remote_memory(remote_send);
        destroy_remote_memory(remote_recv);
        control->barrier();
        ucp.close_endpoints();
        control->barrier();
        unmap_memory(ucp, recv_memh);
        unmap_memory(ucp, send_memh);
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
