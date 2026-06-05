/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <getopt.h>
#include <mpi.h>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_utils.hpp"


using namespace rapidsmpf;

enum class PostOrder : std::uint8_t {
    RankMajor,
    Balanced,
    BalancedInterleaved
};

enum class TagMode : std::uint8_t {
    Constant,
    PerCopy,
    PerRound
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

PostOrder parse_post_order(std::string const& value) {
    if (value == "rank-major" || value == "major" || value == "device-major") {
        return PostOrder::RankMajor;
    }
    if (value == "balanced" || value == "round-robin") {
        return PostOrder::Balanced;
    }
    if (value == "balanced-interleaved" || value == "round-robin-interleaved"
        || value == "interleaved")
    {
        return PostOrder::BalancedInterleaved;
    }
    throw std::invalid_argument(
        "-P/--post-order must be one of {rank-major, balanced, balanced-interleaved}"
    );
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
    RAPIDSMPF_FAIL("unknown post order", std::invalid_argument);
}

TagMode parse_tag_mode(std::string const& value) {
    if (value == "constant" || value == "single") {
        return TagMode::Constant;
    }
    if (value == "per-copy" || value == "copy" || value == "unique") {
        return TagMode::PerCopy;
    }
    if (value == "per-round" || value == "round") {
        return TagMode::PerRound;
    }
    throw std::invalid_argument(
        "-T/--tag-mode must be one of {constant, per-copy, per-round}"
    );
}

std::string to_string(TagMode mode) {
    switch (mode) {
    case TagMode::Constant:
        return "constant";
    case TagMode::PerCopy:
        return "per-copy";
    case TagMode::PerRound:
        return "per-round";
    }
    RAPIDSMPF_FAIL("unknown tag mode", std::invalid_argument);
}

CompletionMode parse_completion_mode(std::string const& value) {
    if (value == "ordered" || value == "vector") {
        return CompletionMode::Ordered;
    }
    if (value == "unordered" || value == "map") {
        return CompletionMode::Unordered;
    }
    throw std::invalid_argument(
        "-W/--completion-mode must be one of {ordered, unordered}"
    );
}

std::string to_string(CompletionMode mode) {
    switch (mode) {
    case CompletionMode::Ordered:
        return "ordered";
    case CompletionMode::Unordered:
        return "unordered";
    }
    RAPIDSMPF_FAIL("unknown completion mode", std::invalid_argument);
}

ProgressDuringPost parse_progress_during_post(std::string const& value) {
    if (value == "none" || value == "off" || value == "disabled") {
        return ProgressDuringPost::None;
    }
    if (value == "request" || value == "copy" || value == "post") {
        return ProgressDuringPost::Request;
    }
    if (value == "direction") {
        return ProgressDuringPost::Direction;
    }
    if (value == "round") {
        return ProgressDuringPost::Round;
    }
    if (value == "operation" || value == "op") {
        return ProgressDuringPost::Operation;
    }
    throw std::invalid_argument(
        "--progress-during-post must be one of {none, request, direction, round, "
        "operation}"
    );
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
    RAPIDSMPF_FAIL("unknown progress-during-post mode", std::invalid_argument);
}

class ArgumentParser {
  public:
    ArgumentParser(int argc, char* const* argv, bool use_mpi = true) {
        int rank = 0;
        int nranks = 1;

        if (use_mpi) {
            RAPIDSMPF_EXPECTS(mpi::is_initialized() == true, "MPI is not initialized");
            RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
            RAPIDSMPF_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
        } else {
            // When not using MPI, expect to be using bootstrap mode (rrun)
            nranks = rapidsmpf::bootstrap::get_nranks();
        }

        try {
            int option;
            struct option long_options[] = {
                {"post-order", required_argument, nullptr, 'P'},
                {"post-mode", required_argument, nullptr, 'P'},
                {"tag-mode", required_argument, nullptr, 'T'},
                {"completion-mode", required_argument, nullptr, 'W'},
                {"wait-mode", required_argument, nullptr, 'W'},
                {"progress-during-post", required_argument, nullptr, 'G'},
                {nullptr, 0, nullptr, 0}
            };
#ifdef RAPIDSMPF_HAVE_CUPTI
            char const* option_string = "hC:O:r:w:n:p:m:P:T:W:G:M:";
#else
            char const* option_string = "hC:O:r:w:n:p:m:P:T:W:G:";
#endif
            while ((option =
                        getopt_long(argc, argv, option_string, long_options, nullptr))
                   != -1)
            {
                switch (option) {
                case 'h':
                    {
                        std::stringstream ss;
                        ss << "Usage: " << argv[0] << " [options]\n"
                           << "Options:\n"
                           << "  -C <comm>  Communicator {mpi, ucxx} (default: mpi)\n"
                           << "             ucxx automatically detects launcher (mpirun "
                              "or rrun)\n"
                           << "  -O <op>    Operation {all-to-all} (default: "
                              "all-to-all)\n"
                           << "  -n <num>   Message size in bytes (default: 1M)\n"
                           << "  -p <num>   Number of concurrent operations, e.g. number"
                              " of  concurrent all-to-all operations (default: 1)\n"
                           << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                              "managed} "
                              "(default: pool)\n"
                           << "  -r <num>   Number of runs (default: 1)\n"
                           << "  -w <num>   Number of warmup runs (default: 0)\n"
                           << "  -P, --post-order <order>\n"
                           << "             Posting order {rank-major, balanced, "
                              "balanced-interleaved} "
                              "(default: rank-major)\n"
                           << "  -T, --tag-mode <mode>\n"
                           << "             Tag assignment {constant, per-copy, "
                              "per-round} "
                              "(default: constant)\n"
                           << "  -W, --completion-mode <mode>\n"
                           << "             Completion polling {ordered, unordered} "
                              "(default: ordered)\n"
                           << "  -G, --progress-during-post <mode>\n"
                           << "             UCXX progress while posting {none, request, "
                              "direction, round, operation} "
                              "(default: none)\n"
#ifdef RAPIDSMPF_HAVE_CUPTI
                           << "  -M <path>  Enable CUPTI memory monitoring and save CSV "
                              "files with given path prefix. For example, /tmp/test will "
                              "write files to /tmp/test_<rank>.csv (default: disabled)\n"
#endif
                           << "  -h         Display this help message\n";
                        if (rank == 0) {
                            std::cerr << ss.str();
                        }
                        if (use_mpi) {
                            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, 0));
                        } else {
                            std::exit(0);
                        }
                    }
                    break;
                case 'C':
                    comm_type = std::string{optarg};
                    if (!(comm_type == "mpi" || comm_type == "ucxx")) {
                        if (rank == 0) {
                            std::cerr << "-C (Communicator) must be one of {mpi, ucxx}"
                                      << std::endl;
                        }
                        if (use_mpi) {
                            RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                        } else {
                            std::exit(-1);
                        }
                    }
                    break;
                case 'O':
                    operation = std::string{optarg};
                    if (operation != "all-to-all") {
                        throw std::invalid_argument(
                            "-O (Operation) must be one of {all-to-all}"
                        );
                    }
                    break;
                case 'n':
                    parse_integer(msg_size, optarg);
                    break;
                case 'p':
                    parse_integer(num_ops, optarg);
                    break;
                case 'm':
                    rmm_mr = std::string{optarg};
                    if (!(rmm_mr == "cuda" || rmm_mr == "pool" || rmm_mr == "async"
                          || rmm_mr == "managed"))
                    {
                        throw std::invalid_argument(
                            "-m (RMM memory resource) must be one of {cuda, pool, async, "
                            "managed}"
                        );
                    }
                    break;
                case 'r':
                    parse_integer(num_runs, optarg);
                    break;
                case 'w':
                    parse_integer(num_warmups, optarg);
                    break;
                case 'P':
                    post_order = parse_post_order(std::string{optarg});
                    break;
                case 'T':
                    tag_mode = parse_tag_mode(std::string{optarg});
                    break;
                case 'W':
                    completion_mode = parse_completion_mode(std::string{optarg});
                    break;
                case 'G':
                    progress_during_post =
                        parse_progress_during_post(std::string{optarg});
                    break;
#ifdef RAPIDSMPF_HAVE_CUPTI
                case 'M':
                    cupti_csv_prefix = std::string{optarg};
                    enable_cupti_monitoring = true;
                    break;
#endif
                case '?':
                    if (use_mpi) {
                        RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
                    } else {
                        std::exit(-1);
                    }
                    break;
                default:
                    RAPIDSMPF_FAIL("unknown option", std::invalid_argument);
                }
            }
            if (optind < argc) {
                RAPIDSMPF_FAIL("unknown option", std::invalid_argument);
            }
        } catch (std::exception const& e) {
            if (rank == 0) {
                std::cerr << "Error parsing arguments: " << e.what() << std::endl;
            }
            if (use_mpi) {
                RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, -1));
            } else {
                std::exit(-1);
            }
        }

        if (rmm_mr == "cuda") {
            if (rank == 0) {
                std::cout << "WARNING: using the default cuda memory resource "
                             "(-m cuda) might leak memory! A limitation in UCX "
                             "means that device memory send through IPC can "
                             "never be freed."
                          << std::endl;
            }
        }
    }

    void pprint(Communicator& comm) const {
        if (comm.rank() > 0) {
            return;
        }
        std::stringstream ss;
        ss << "Arguments:\n";
        ss << "  -C " << comm_type << " (communicator)\n";
        ss << "  -O " << operation << " (operation)\n";
        ss << "  -n " << msg_size << " (message size)\n";
        ss << "  -p " << num_ops << " (number of operations)\n";
        ss << "  -r " << num_runs << " (number of runs)\n";
        ss << "  -w " << num_warmups << " (number of warmup runs)\n";
        ss << "  -P " << to_string(post_order) << " (posting order)\n";
        ss << "  -T " << to_string(tag_mode) << " (tag mode)\n";
        ss << "  -W " << to_string(completion_mode) << " (completion mode)\n";
        ss << "  -G " << to_string(progress_during_post) << " (progress during post)\n";
        ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
        if (enable_cupti_monitoring) {
            ss << "  -M " << cupti_csv_prefix << " (CUPTI memory monitoring enabled)\n";
        }
        comm.logger()->print(ss.str());
    }

    std::uint64_t num_runs{1};
    std::uint64_t num_warmups{0};
    std::string rmm_mr{"pool"};
    std::string comm_type{"mpi"};
    std::string operation{"all-to-all"};
    std::uint64_t msg_size{1 << 20};
    std::uint64_t num_ops{1};
    PostOrder post_order{PostOrder::RankMajor};
    TagMode tag_mode{TagMode::Constant};
    CompletionMode completion_mode{CompletionMode::Ordered};
    ProgressDuringPost progress_during_post{ProgressDuringPost::None};
    bool enable_cupti_monitoring{false};
    std::string cupti_csv_prefix;
};

struct BufferSet {
    std::vector<std::unique_ptr<Buffer>> send_bufs;
    std::vector<std::unique_ptr<Buffer>> recv_bufs;
};

struct FutureSlot {
    enum class Kind : std::uint8_t {
        Send,
        Recv
    };

    Kind kind;
    std::uint64_t index;
};

void barrier(std::shared_ptr<Communicator> const& comm) {
    bool const use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();
    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));
        return;
    }

    auto ucxx_comm = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm);
    RAPIDSMPF_EXPECTS(
        ucxx_comm != nullptr, "rrun benchmark requires a UCXX communicator"
    );
    ucxx_comm->barrier();
}

std::uint64_t buffer_index(Communicator const& comm, std::uint64_t i, Rank rank) {
    return static_cast<std::uint64_t>(rank)
           + i * static_cast<std::uint64_t>(comm.nranks());
}

std::uint64_t rank_pair_index(Rank nranks, Rank src, Rank dst) {
    RAPIDSMPF_EXPECTS(src >= 0 && src < nranks, "invalid source rank");
    RAPIDSMPF_EXPECTS(dst >= 0 && dst < nranks, "invalid destination rank");
    auto const rank_count = rapidsmpf::safe_cast<std::uint64_t>(nranks);
    return rapidsmpf::safe_cast<std::uint64_t>(src) * rank_count
           + rapidsmpf::safe_cast<std::uint64_t>(dst);
}

Tag make_linear_tag(std::uint64_t op_id) {
    auto constexpr max_op_id = std::uint64_t{1} << Tag::op_id_bits;
    RAPIDSMPF_EXPECTS(
        op_id < max_op_id,
        "tag mode requires more distinct tags than RAPIDSMPF Tag can represent",
        std::overflow_error
    );
    return Tag{rapidsmpf::safe_cast<OpID>(op_id), StageID{0}};
}

Tag make_message_tag(
    ArgumentParser const& args,
    Rank nranks,
    std::uint64_t op,
    Rank src,
    Rank dst,
    std::uint64_t round,
    std::uint64_t direction
) {
    switch (args.tag_mode) {
    case TagMode::Constant:
        return Tag{OpID{0}, StageID{0}};
    case TagMode::PerCopy:
        {
            auto const rank_count = rapidsmpf::safe_cast<std::uint64_t>(nranks);
            auto const tags_per_op = rank_count * rank_count;
            return make_linear_tag(
                std::uint64_t{1} + op * tags_per_op + rank_pair_index(nranks, src, dst)
            );
        }
    case TagMode::PerRound:
        {
            if (args.post_order == PostOrder::RankMajor) {
                auto const rank_count = rapidsmpf::safe_cast<std::uint64_t>(nranks);
                auto const tags_per_op = rank_count * rank_count;
                return make_linear_tag(
                    std::uint64_t{1} + op * tags_per_op
                    + rank_pair_index(nranks, src, dst)
                );
            }
            auto const rank_count = rapidsmpf::safe_cast<std::uint64_t>(nranks);
            auto const participant_count = rank_count + (rank_count % 2);
            auto const rounds_per_op = participant_count > 1 ? participant_count - 1 : 1;
            auto const tags_per_op = rounds_per_op * std::uint64_t{2};
            RAPIDSMPF_EXPECTS(direction < 2, "invalid balanced direction");
            RAPIDSMPF_EXPECTS(round < rounds_per_op, "invalid balanced round");
            return make_linear_tag(
                std::uint64_t{1} + op * tags_per_op + round * std::uint64_t{2} + direction
            );
        }
    }
    RAPIDSMPF_FAIL("unknown tag mode", std::invalid_argument);
}

void progress_communicator(Communicator& comm) {
    auto* ucxx_comm = dynamic_cast<rapidsmpf::ucxx::UCXX*>(&comm);
    if (ucxx_comm != nullptr) {
        ucxx_comm->progress();
    }
}

void progress_after(
    Communicator& comm, ArgumentParser const& args, ProgressDuringPost point
) {
    if (args.progress_during_post == point) {
        progress_communicator(comm);
    }
}

using RankPair = std::pair<Rank, Rank>;
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

std::vector<PairRound> round_robin_pair_rounds(Rank nranks) {
    auto const rank_count = rapidsmpf::safe_cast<std::size_t>(nranks);
    if (rank_count < 2) {
        return {};
    }

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
                pairs.emplace_back(
                    rapidsmpf::safe_cast<Rank>(first), rapidsmpf::safe_cast<Rank>(second)
                );
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

BufferSet allocate_buffers(
    std::shared_ptr<Communicator> const& comm,
    ArgumentParser const& args,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    BufferSet buffers;
    auto const buffer_count = args.num_ops * static_cast<std::uint64_t>(comm->nranks());
    buffers.send_bufs.reserve(buffer_count);
    buffers.recv_bufs.reserve(buffer_count);
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < comm->nranks(); ++rank) {
            auto [res, _] =
                br->reserve(MemoryType::DEVICE, args.msg_size * 2, AllowOverbooking::YES);
            auto buf = br->make_buffer(args.msg_size, stream, res);
            random_fill(*buf, br->device_mr());
            buffers.send_bufs.push_back(std::move(buf));
            buffers.recv_bufs.push_back(br->make_buffer(args.msg_size, stream, res));
        }
    }
    return buffers;
}

void post_recv(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    Tag tag,
    std::uint64_t op,
    Rank peer,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    auto const idx = buffer_index(comm, op, peer);
    auto buf = std::move(buffers.recv_bufs.at(idx));
    RAPIDSMPF_EXPECTS(buf != nullptr, "recv buffer slot is empty");
    statistics.add_bytes_stat("all-to-all-recv", buf->size);
    futures.push_back(comm.recv(peer, tag, std::move(buf)));
    future_slots.push_back({FutureSlot::Kind::Recv, idx});
    progress_after(comm, args, ProgressDuringPost::Request);
}

void post_send(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    Tag tag,
    std::uint64_t op,
    Rank peer,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    auto const idx = buffer_index(comm, op, peer);
    auto buf = std::move(buffers.send_bufs.at(idx));
    RAPIDSMPF_EXPECTS(buf != nullptr, "send buffer slot is empty");
    statistics.add_bytes_stat("all-to-all-send", buf->size);
    futures.push_back(comm.send(std::move(buf), peer, tag));
    future_slots.push_back({FutureSlot::Kind::Send, idx});
    progress_after(comm, args, ProgressDuringPost::Request);
}

void post_rank_major_copies(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (Rank rank = 0; rank < comm.nranks(); ++rank) {
            if (rank != comm.rank()) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    i,
                    rank,
                    comm.rank(),
                    rank_pair_index(comm.nranks(), rank, comm.rank()),
                    0
                );
                post_recv(
                    comm, args, statistics, tag, i, rank, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        for (Rank rank = 0; rank < comm.nranks(); ++rank) {
            if (rank != comm.rank()) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    i,
                    comm.rank(),
                    rank,
                    rank_pair_index(comm.nranks(), comm.rank(), rank),
                    0
                );
                post_send(
                    comm, args, statistics, tag, i, rank, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        progress_after(comm, args, ProgressDuringPost::Operation);
    }
}

void post_balanced_recvs(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    std::uint64_t op,
    std::vector<PairRound> const& rounds,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    for (std::size_t round_idx = 0; round_idx < rounds.size(); ++round_idx) {
        auto const& round = rounds.at(round_idx);
        for (auto const& [src, dst] : round) {
            if (comm.rank() == dst) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    op,
                    src,
                    dst,
                    rapidsmpf::safe_cast<std::uint64_t>(round_idx),
                    0
                );
                post_recv(
                    comm, args, statistics, tag, op, src, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        for (auto const& [dst, src] : round) {
            if (comm.rank() == dst) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    op,
                    src,
                    dst,
                    rapidsmpf::safe_cast<std::uint64_t>(round_idx),
                    1
                );
                post_recv(
                    comm, args, statistics, tag, op, src, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        progress_after(comm, args, ProgressDuringPost::Round);
    }
}

void post_balanced_sends(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    std::uint64_t op,
    std::vector<PairRound> const& rounds,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    for (std::size_t round_idx = 0; round_idx < rounds.size(); ++round_idx) {
        auto const& round = rounds.at(round_idx);
        for (auto const& [src, dst] : round) {
            if (comm.rank() == src) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    op,
                    src,
                    dst,
                    rapidsmpf::safe_cast<std::uint64_t>(round_idx),
                    0
                );
                post_send(
                    comm, args, statistics, tag, op, dst, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        for (auto const& [dst, src] : round) {
            if (comm.rank() == src) {
                auto const tag = make_message_tag(
                    args,
                    comm.nranks(),
                    op,
                    src,
                    dst,
                    rapidsmpf::safe_cast<std::uint64_t>(round_idx),
                    1
                );
                post_send(
                    comm, args, statistics, tag, op, dst, buffers, futures, future_slots
                );
            }
        }
        progress_after(comm, args, ProgressDuringPost::Direction);
        progress_after(comm, args, ProgressDuringPost::Round);
    }
}

void post_balanced_copies(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    auto const rounds = round_robin_pair_rounds(comm.nranks());
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        post_balanced_recvs(
            comm, args, statistics, i, rounds, buffers, futures, future_slots
        );
        post_balanced_sends(
            comm, args, statistics, i, rounds, buffers, futures, future_slots
        );
        progress_after(comm, args, ProgressDuringPost::Operation);
    }
}

void post_balanced_interleaved_direction_recvs(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    std::uint64_t op,
    PairRound const& round,
    std::uint64_t round_idx,
    std::uint64_t direction,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    for (auto const& [first, second] : round) {
        auto const src = direction == 0 ? first : second;
        auto const dst = direction == 0 ? second : first;
        if (comm.rank() == dst) {
            auto const tag =
                make_message_tag(args, comm.nranks(), op, src, dst, round_idx, direction);
            post_recv(
                comm, args, statistics, tag, op, src, buffers, futures, future_slots
            );
        }
    }
}

void post_balanced_interleaved_direction_sends(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    std::uint64_t op,
    PairRound const& round,
    std::uint64_t round_idx,
    std::uint64_t direction,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    for (auto const& [first, second] : round) {
        auto const src = direction == 0 ? first : second;
        auto const dst = direction == 0 ? second : first;
        if (comm.rank() == src) {
            auto const tag =
                make_message_tag(args, comm.nranks(), op, src, dst, round_idx, direction);
            post_send(
                comm, args, statistics, tag, op, dst, buffers, futures, future_slots
            );
        }
    }
}

void post_balanced_interleaved_copies(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    auto const rounds = round_robin_pair_rounds(comm.nranks());
    for (std::uint64_t i = 0; i < args.num_ops; ++i) {
        for (std::size_t round_idx = 0; round_idx < rounds.size(); ++round_idx) {
            auto const& round = rounds.at(round_idx);
            auto const round_id = rapidsmpf::safe_cast<std::uint64_t>(round_idx);
            for (std::uint64_t direction = 0; direction < 2; ++direction) {
                post_balanced_interleaved_direction_recvs(
                    comm,
                    args,
                    statistics,
                    i,
                    round,
                    round_id,
                    direction,
                    buffers,
                    futures,
                    future_slots
                );
                post_balanced_interleaved_direction_sends(
                    comm,
                    args,
                    statistics,
                    i,
                    round,
                    round_id,
                    direction,
                    buffers,
                    futures,
                    future_slots
                );
                progress_after(comm, args, ProgressDuringPost::Direction);
            }
            progress_after(comm, args, ProgressDuringPost::Round);
        }
        progress_after(comm, args, ProgressDuringPost::Operation);
    }
}

void post_all_copies(
    Communicator& comm,
    ArgumentParser const& args,
    rapidsmpf::Statistics& statistics,
    BufferSet& buffers,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots
) {
    switch (args.post_order) {
    case PostOrder::RankMajor:
        post_rank_major_copies(comm, args, statistics, buffers, futures, future_slots);
        break;
    case PostOrder::Balanced:
        post_balanced_copies(comm, args, statistics, buffers, futures, future_slots);
        break;
    case PostOrder::BalancedInterleaved:
        post_balanced_interleaved_copies(
            comm, args, statistics, buffers, futures, future_slots
        );
        break;
    }
}

void release_completed_future(
    Communicator& comm,
    std::unique_ptr<Communicator::Future> completed,
    FutureSlot const& slot,
    BufferSet& buffers
) {
    auto buf = comm.release_data(std::move(completed));
    switch (slot.kind) {
    case FutureSlot::Kind::Send:
        buffers.send_bufs.at(slot.index) = std::move(buf);
        break;
    case FutureSlot::Kind::Recv:
        buffers.recv_bufs.at(slot.index) = std::move(buf);
        break;
    }
}

void release_completed_futures_ordered(
    Communicator& comm,
    std::vector<std::unique_ptr<Communicator::Future>>&& completed,
    std::vector<std::size_t> const& indices,
    std::vector<FutureSlot>& slots,
    BufferSet& buffers
) {
    RAPIDSMPF_EXPECTS(completed.size() == indices.size(), "completed futures mismatch");
    for (std::size_t i = 0; i < completed.size(); ++i) {
        auto const slot = slots.at(indices.at(i));
        release_completed_future(comm, std::move(completed.at(i)), slot, buffers);
    }

    auto sorted_indices = indices;
    std::ranges::sort(sorted_indices, std::greater<>{});
    for (auto index : sorted_indices) {
        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
    }
}

void wait_for_completion_ordered(
    Communicator& comm,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots,
    BufferSet& buffers
) {
    while (!futures.empty()) {
        auto [completed, indices] = comm.test_some(futures);
        release_completed_futures_ordered(
            comm, std::move(completed), indices, future_slots, buffers
        );
    }
    RAPIDSMPF_EXPECTS(future_slots.empty(), "all futures completed but slots remain");
}

void wait_for_completion_unordered(
    Communicator& comm,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots,
    BufferSet& buffers
) {
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> future_map;
    future_map.reserve(futures.size());
    for (std::size_t i = 0; i < futures.size(); ++i) {
        future_map.emplace(i, std::move(futures.at(i)));
    }
    futures.clear();

    while (!future_map.empty()) {
        auto const completed_keys = comm.test_some(future_map);
        for (auto const key : completed_keys) {
            auto iter = future_map.find(key);
            RAPIDSMPF_EXPECTS(iter != future_map.end(), "completed future key missing");
            auto completed = std::move(iter->second);
            future_map.erase(iter);
            release_completed_future(
                comm, std::move(completed), future_slots.at(key), buffers
            );
        }
    }
    future_slots.clear();
}

void wait_for_completion(
    Communicator& comm,
    ArgumentParser const& args,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    std::vector<FutureSlot>& future_slots,
    BufferSet& buffers
) {
    switch (args.completion_mode) {
    case CompletionMode::Ordered:
        wait_for_completion_ordered(comm, futures, future_slots, buffers);
        break;
    case CompletionMode::Unordered:
        wait_for_completion_unordered(comm, futures, future_slots, buffers);
        break;
    }
}

Duration run(
    std::shared_ptr<Communicator> comm,
    ArgumentParser const& args,
    BufferSet& buffers,
    std::shared_ptr<rapidsmpf::Statistics> statistics
) {
    // Sync before we start the timer.
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());
    barrier(comm);

    auto const t0_elapsed = Clock::now();

    std::vector<std::unique_ptr<Communicator::Future>> futures;
    std::vector<FutureSlot> future_slots;
    futures.reserve(args.num_ops * static_cast<std::uint64_t>(comm->nranks() - 1) * 2);
    future_slots.reserve(futures.capacity());
    post_all_copies(*comm, args, *statistics, buffers, futures, future_slots);
    wait_for_completion(*comm, args, futures, future_slots, buffers);

    return Clock::now() - t0_elapsed;
}

int main(int argc, char** argv) {
    bool use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();

    int provided = 0;
    if (!use_bootstrap) {
        // Explicitly initialize MPI with thread support, as this is needed for both mpi
        // and ucxx communicators.
        RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

        RAPIDSMPF_EXPECTS(
            provided == MPI_THREAD_MULTIPLE,
            "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
        );
    }

    ArgumentParser args{argc, argv, !use_bootstrap};

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    // We'll only measure the last run, so start disabled.
    auto stats = rapidsmpf::Statistics::disabled();
    auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>(stats);
    std::shared_ptr<Communicator> comm;
    if (args.comm_type == "mpi") {
        if (use_bootstrap) {
            std::cerr << "Error: MPI communicator requires MPI initialization. "
                      << "Don't use with rrun or unset RRUN_RANK." << std::endl;
            return 1;
        }
        mpi::init(&argc, &argv);
        comm = std::make_shared<MPI>(MPI_COMM_WORLD, options, progress_thread);
    } else if (args.comm_type == "ucxx") {
        if (use_bootstrap) {
            // Launched with rrun - use bootstrap backend
            comm = rapidsmpf::bootstrap::create_ucxx_comm(
                progress_thread, rapidsmpf::bootstrap::BackendType::AUTO, options
            );
        } else {
            // Launched with mpirun - use MPI bootstrap
            comm =
                rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options, progress_thread);
        }
    } else {
        std::cerr << "Error: Unknown communicator type: " << args.comm_type << std::endl;
        return 1;
    }

    auto& log = comm->logger();
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    args.pprint(*comm);
    set_current_rmm_resource(args.rmm_mr);

    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    auto br = BufferResource::create(
        mr,
        PinnedMemoryResource::Disabled,
        {},
        std::chrono::milliseconds{1},
        std::make_shared<rmm::cuda_stream_pool>(
            16, rmm::cuda_stream::flags::non_blocking
        ),
        stats
    );

    // Print benchmark/hardware info.
    {
        std::stringstream ss;
        auto const cur_dev = rmm::get_current_cuda_device().value();
        std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
        RAPIDSMPF_CUDA_TRY(
            cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev)
        );
        cudaDeviceProp properties;
        RAPIDSMPF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
        ss << "Hardware setup: \n";
        ss << "  GPU (" << properties.name << "): \n";
        ss << "    Device number: " << cur_dev << "\n";
        ss << "    PCI Bus ID: " << pci_bus_id.substr(0, pci_bus_id.find('\0')) << "\n";
        ss << "    Total Memory: " << format_nbytes(properties.totalGlobalMem, 0) << "\n";
        ss << "  Comm: " << *comm << "\n";
        log->print(ss.str());
    }

    auto buffers = allocate_buffers(comm, args, stream, br.get());

#ifdef RAPIDSMPF_HAVE_CUPTI
    // Create CUPTI monitor if enabled
    std::unique_ptr<rapidsmpf::CuptiMonitor> cupti_monitor;
    if (args.enable_cupti_monitoring) {
        cupti_monitor = std::make_unique<rapidsmpf::CuptiMonitor>();
        cupti_monitor->start_monitoring();
        log->print("CUPTI memory monitoring enabled");
    }
#endif

    auto const local_messages_send =
        args.msg_size * args.num_ops * (static_cast<std::uint64_t>(comm->nranks()) - 1);
    auto const global_messages =
        local_messages_send * static_cast<std::uint64_t>(comm->nranks());
    std::vector<double> elapsed_vec;
    for (std::uint64_t i = 0; i < args.num_warmups + args.num_runs; ++i) {
        // Enable statistics for the last run.
        if (i == args.num_warmups + args.num_runs - 1) {
            stats->enable();
        }
        auto const elapsed = run(comm, args, buffers, stats).count();
        barrier(comm);
        std::stringstream ss;
        ss << "elapsed: " << format_duration(elapsed)
           << " | local comm: " << format_nbytes(local_messages_send / elapsed)
           << "/s | global throughput: " << format_nbytes(global_messages / elapsed)
           << "/s";
        if (i < args.num_warmups) {
            ss << " (warmup run)";
        }
        log->print(ss.str());
        if (i >= args.num_warmups) {
            elapsed_vec.push_back(elapsed);
        }
    }

    {
        auto const elapsed_mean = harmonic_mean(elapsed_vec);
        std::stringstream ss;
        ss << "means: " << format_duration(elapsed_mean)
           << " | local comm: " << format_nbytes(local_messages_send / elapsed_mean)
           << "/s | global throughput: " << format_nbytes(global_messages / elapsed_mean)
           << "/s | num_ops: " << args.num_ops << " | nranks: " << comm->nranks();
        log->print(ss.str());
    }
    log->print(stats->report({.header = "Statistics (of the last run):"}));

#ifdef RAPIDSMPF_HAVE_CUPTI
    // Save CUPTI monitoring results to CSV file
    if (args.enable_cupti_monitoring && cupti_monitor) {
        cupti_monitor->stop_monitoring();

        std::string csv_filename =
            args.cupti_csv_prefix + std::to_string(comm->rank()) + ".csv";
        try {
            cupti_monitor->write_csv(csv_filename);
            log->print(
                "CUPTI memory data written to " + csv_filename + " ("
                + std::to_string(cupti_monitor->get_sample_count()) + " samples, "
                + std::to_string(cupti_monitor->get_total_callback_count())
                + " callbacks)"
            );

            // Print callback summary for rank 0
            if (comm->rank() == 0) {
                log->print(
                    "CUPTI Callback Summary:\n" + cupti_monitor->get_callback_summary()
                );
            }
        } catch (std::exception const& e) {
            log->print("Failed to write CUPTI CSV file: " + std::string(e.what()));
        }
    }
#endif

    if (!use_bootstrap) {
        RAPIDSMPF_MPI(MPI_Finalize());
    }
    return 0;
}
