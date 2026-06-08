/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>
#include <ucxx/listener.h>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include "../environment.hpp"
#include "../utils.hpp"

namespace {

std::string memory_label(rapidsmpf::MemoryType memory_type) {
    switch (memory_type) {
    case rapidsmpf::MemoryType::HOST:
        return "host";
    case rapidsmpf::MemoryType::DEVICE:
        return "device";
    case rapidsmpf::MemoryType::PINNED_HOST:
        return "host";
    }
    return "unknown";
}

bool has_stat_with_prefix(rapidsmpf::Statistics const& stats, std::string const& prefix) {
    for (auto const& name : stats.list_stat_names()) {
        if (name.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

bool has_stat(rapidsmpf::Statistics const& stats, std::string const& stat_name) {
    for (auto const& name : stats.list_stat_names()) {
        if (name == stat_name) {
            return true;
        }
    }
    return false;
}

void self_transfer(
    rapidsmpf::Communicator& comm,
    rapidsmpf::BufferResource& br,
    rapidsmpf::MemoryType memory_type,
    std::size_t nbytes,
    rapidsmpf::Tag tag
) {
    auto send_data_h = iota_vector<std::uint8_t>(static_cast<int>(nbytes));
    auto stream = rmm::cuda_stream_default;
    auto send_buf = br.make_buffer(stream, br.reserve_or_fail(nbytes, memory_type));
    send_buf->write_access([&](std::byte* send_buf_data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(
            rapidsmpf::cuda_memcpy_async(
                send_buf_data, send_data_h.data(), nbytes, stream
            )
        );
    });
    send_buf->stream().synchronize();

    auto recv_buf = br.make_buffer(stream, br.reserve_or_fail(nbytes, memory_type));
    recv_buf->stream().synchronize();

    auto send_fut = comm.send(std::move(send_buf), comm.rank(), tag);
    auto recv_fut = comm.recv(comm.rank(), tag, std::move(recv_buf));
    std::ignore = comm.wait(std::move(send_fut));
    recv_buf = comm.wait(std::move(recv_fut));
}

}  // namespace

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

TestEnvironmentType Environment::type() const {
    return TestEnvironmentType::UCXX;
}

void Environment::SetUp() {
    // Ensure CUDA context is created before UCX is initialized.
    cudaFree(nullptr);

    // Explicitly initialize MPI. We can not use rapidsmpf::mpi::init as it checks some
    // rapidsmpf::MPI communicator specific conditions
    int provided;
    RAPIDSMPF_MPI(MPI_Init_thread(&argc_, &argv_, MPI_THREAD_MULTIPLE, &provided));
    RAPIDSMPF_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );

    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
    options_.insert_if_absent({{"ucxx_request_attributes", "true"}});
    comm_ = rapidsmpf::ucxx::init_using_mpi(
        MPI_COMM_WORLD, options_, std::make_shared<rapidsmpf::ProgressThread>()
    );
}

void Environment::TearDown() {
    // Ensure UCXX cleanup before MPI. If this is not done failures related to
    // accessing the CUDA context may be thrown during shutdown.
    split_comm_ = nullptr;  // Clean up the split communicator.
    comm_ = nullptr;  // Clean up the communicator.
    RAPIDSMPF_MPI(MPI_Finalize());
}

void Environment::barrier() {
    std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->barrier();
}

std::shared_ptr<rapidsmpf::Communicator> Environment::split_comm() {
    // Return cached split communicator if it exists
    if (split_comm_ != nullptr) {
        return split_comm_;
    }

    // Create and cache the new split communicator
    split_comm_ = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->split();
    return split_comm_;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}

TEST(UCXXDetailedMetricsTest, RequestAttributesRecordSourceAndDestinationMemory) {
    auto comm = GlobalEnvironment->comm_;
    auto stats = comm->progress_thread()->statistics();
    ASSERT_NE(stats, nullptr);
    stats->clear();
    stats->enable();

    rmm::mr::cuda_memory_resource mr;
    auto br =
        rapidsmpf::BufferResource::from_options(mr, rapidsmpf::config::Options{}, stats);

    constexpr std::size_t nbytes = 4_MiB;
    self_transfer(
        *comm, *br, rapidsmpf::MemoryType::HOST, nbytes, rapidsmpf::Tag{997, 0}
    );
    self_transfer(
        *comm, *br, rapidsmpf::MemoryType::DEVICE, nbytes, rapidsmpf::Tag{997, 1}
    );

    for (auto memory_type : {rapidsmpf::MemoryType::HOST, rapidsmpf::MemoryType::DEVICE})
    {
        auto const label = memory_label(memory_type);
        auto const send_name = "ucxx-tag-send-source-" + label;
        auto const recv_name = "ucxx-tag-recv-destination-" + label;

        SCOPED_TRACE(stats->report());
        ASSERT_TRUE(has_stat(*stats, send_name));
        ASSERT_TRUE(has_stat(*stats, recv_name));
        EXPECT_EQ(stats->get_stat(send_name).value(), nbytes);
        EXPECT_EQ(stats->get_stat(recv_name).value(), nbytes);
        EXPECT_TRUE(has_stat_with_prefix(*stats, send_name + "-debug-"));
        EXPECT_TRUE(has_stat_with_prefix(*stats, recv_name + "-debug-"));
    }
}
