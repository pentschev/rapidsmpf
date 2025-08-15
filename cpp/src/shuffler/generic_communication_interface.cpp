/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

#include <cuda_runtime.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/generic_communication_interface.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler {

TagGenericCommunicationInterface::TagGenericCommunicationInterface(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    Rank rank,
    std::shared_ptr<Statistics> statistics
)
    : comm_(std::move(comm)),
      rank_(rank),
      ready_for_data_tag_{op_id, 1},
      metadata_tag_{op_id, 2},
      gpu_data_tag_{op_id, 3},
      statistics_{std::move(statistics)} {}

void TagGenericCommunicationInterface::submit_outgoing_messages(
    std::vector<std::unique_ptr<MessageInterface>>&& messages,
    std::function<Rank(MessageInterface const&)> peer_rank_fn,
    BufferResource* br
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    // Store messages for sending and initiate metadata transmission
    for (auto&& message : messages) {
        auto dst = peer_rank_fn(*message);
        message->set_peer_rank(dst);
        log.trace("send metadata to ", dst, ": ", message->to_string());
        RAPIDSMPF_EXPECTS(dst != rank_, "sending message to ourselves");

        auto metadata = message->serialize_metadata();
        fire_and_forget_.push_back(comm_->send(
            std::make_unique<std::vector<std::uint8_t>>(std::move(metadata)),
            dst,
            metadata_tag_,
            br
        ));

        if (message->total_data_size() > 0) {
            auto message_id = message->message_id();
            RAPIDSMPF_EXPECTS(
                outgoing_messages_.insert({message_id, std::move(message)}).second,
                "outgoing message already exists"
            );
            ready_ack_receives_[dst].push_back(comm_->recv(
                dst,
                ready_for_data_tag_,
                br->move(
                    std::make_unique<std::vector<std::uint8_t>>(
                        ReadyForDataMessage::byte_size
                    )
                )
            ));
        }
    }

    statistics_->add_duration_stat(
        "generic-comms-interface-submit-outgoing-messages", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MessageInterface>>
TagGenericCommunicationInterface::process_communication(
    MessageFactory const& message_factory,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto const t0 = Clock::now();

    // Process all phases of the communication protocol
    receive_metadata_phase(message_factory);
    setup_data_receives_phase(message_factory, stream, br);
    process_ready_acks_phase();
    auto completed_messages = complete_data_transfers_phase();
    cleanup_completed_operations();

    statistics_->add_duration_stat(
        "generic-comms-interface-process-communication-total", Clock::now() - t0
    );

    return completed_messages;
}

bool TagGenericCommunicationInterface::is_idle() const {
    return fire_and_forget_.empty() && incoming_messages_.empty()
           && outgoing_messages_.empty() && in_transit_messages_.empty()
           && in_transit_futures_.empty()
           && std::ranges::all_of(ready_ack_receives_, [](auto const& kv) {
                  return kv.second.empty();
              });
}

void TagGenericCommunicationInterface::receive_metadata_phase(
    MessageFactory const& message_factory
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag_);
        if (!msg)
            break;

        // The msg is already a vector<uint8_t>, so we can use it directly
        auto message = message_factory.create_from_metadata(*msg);
        message->set_peer_rank(src);
        log.trace("recv_any from ", src, ": ", message->to_string());
        incoming_messages_.insert({src, std::move(message)});
    }

    statistics_->add_duration_stat(
        "generic-comms-interface-receive-metadata", Clock::now() - t0
    );
}

void TagGenericCommunicationInterface::setup_data_receives_phase(
    MessageFactory const& message_factory,
    rmm::cuda_stream_view /* stream */,
    BufferResource* br
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        log.trace(
            "checking incoming message data from ", src, ": ", message->to_string()
        );

        if (message->total_data_size() > 0) {
            if (!message->is_data_ready()) {
                auto buffers = message_factory.allocate_receive_buffers(
                    message->total_data_size(), *message
                );
                message->set_data_buffers(std::move(buffers));
            }

            if (!message->is_ready()) {
                ++it;
                continue;
            }

            // Extract the message and set up for data transfer
            auto message_ptr = std::move(it->second);
            auto src = it->first;
            it = incoming_messages_.erase(it);

            auto data_buffers = message_ptr->release_data_buffers();
            // For simplicity, assume single buffer for now - can be extended for multiple
            // buffers
            RAPIDSMPF_EXPECTS(!data_buffers.empty(), "No data buffers available");
            auto future = comm_->recv(src, gpu_data_tag_, std::move(data_buffers[0]));

            auto message_id = message_ptr->message_id();
            RAPIDSMPF_EXPECTS(
                in_transit_futures_.insert({message_id, std::move(future)}).second,
                "in transit future already exists"
            );
            RAPIDSMPF_EXPECTS(
                in_transit_messages_.insert({message_id, std::move(message_ptr)}).second,
                "in transit message already exists"
            );

            auto ready_msg = ReadyForDataMessage{message_id};
            fire_and_forget_.push_back(comm_->send(
                std::make_unique<std::vector<std::uint8_t>>(ready_msg.pack()),
                src,
                ready_for_data_tag_,
                br
            ));
        } else {
            // Control/metadata-only message - will be handled in
            // complete_data_transfers_phase
            ++it;
        }
    }

    statistics_->add_duration_stat(
        "generic-comms-interface-setup-data-receives", Clock::now() - t0
    );
}

void TagGenericCommunicationInterface::process_ready_acks_phase() {
    auto const t0 = Clock::now();

    for (auto& [dst, futures] : ready_ack_receives_) {
        auto finished = comm_->test_some(futures);
        for (auto&& future : finished) {
            auto const msg_data = comm_->get_gpu_data(std::move(future));
            // The msg_data should be a Buffer containing the message data
            // We need to convert it to vector for processing
            std::vector<std::uint8_t> data(msg_data->size);
            RAPIDSMPF_CUDA_TRY(cudaMemcpy(
                data.data(), msg_data->data(), msg_data->size, cudaMemcpyDeviceToHost
            ));
            auto msg = ReadyForDataMessage::unpack(data);

            auto message_it = outgoing_messages_.find(msg.message_id);
            RAPIDSMPF_EXPECTS(
                message_it != outgoing_messages_.end(), "outgoing message not found"
            );

            auto data_buffers = message_it->second->release_data_buffers();
            // For simplicity, assume single buffer for now - can be extended for multiple
            // buffers
            RAPIDSMPF_EXPECTS(!data_buffers.empty(), "No data buffers available");

            fire_and_forget_.push_back(
                comm_->send(std::move(data_buffers[0]), dst, gpu_data_tag_)
            );

            outgoing_messages_.erase(message_it);
        }
    }

    statistics_->add_duration_stat(
        "generic-comms-interface-process-ready-acks", Clock::now() - t0
    );
}

std::vector<std::unique_ptr<MessageInterface>>
TagGenericCommunicationInterface::complete_data_transfers_phase() {
    auto const t0 = Clock::now();

    std::vector<std::unique_ptr<MessageInterface>> completed_messages;

    // Handle completed data transfers
    if (!in_transit_futures_.empty()) {
        std::vector<std::uint64_t> finished = comm_->test_some(in_transit_futures_);
        for (auto message_id : finished) {
            auto message_it = in_transit_messages_.find(message_id);
            auto future_it = in_transit_futures_.find(message_id);

            RAPIDSMPF_EXPECTS(
                message_it != in_transit_messages_.end(), "in transit message not found"
            );
            RAPIDSMPF_EXPECTS(
                future_it != in_transit_futures_.end(), "in transit future not found"
            );

            auto message = std::move(message_it->second);
            auto future = std::move(future_it->second);
            auto received_buffer = comm_->get_gpu_data(std::move(future));

            std::vector<std::unique_ptr<Buffer>> buffers;
            buffers.push_back(std::move(received_buffer));
            message->set_data_buffers(std::move(buffers));

            completed_messages.push_back(std::move(message));

            in_transit_messages_.erase(message_it);
            in_transit_futures_.erase(future_it);
        }
    }

    // Handle control/metadata-only messages from incoming_messages_
    for (auto it = incoming_messages_.begin(); it != incoming_messages_.end();) {
        auto& [src, message] = *it;
        if (message->total_data_size() == 0) {
            completed_messages.push_back(std::move(it->second));
            it = incoming_messages_.erase(it);
        } else {
            ++it;
        }
    }

    statistics_->add_duration_stat(
        "generic-comms-interface-complete-data-transfers", Clock::now() - t0
    );

    return completed_messages;
}

void TagGenericCommunicationInterface::cleanup_completed_operations() {
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }
}

// ReadyForDataMessage implementation
std::vector<std::uint8_t> ReadyForDataMessage::pack() const {
    std::vector<std::uint8_t> buffer(byte_size);
    std::memcpy(buffer.data(), &message_id, sizeof(message_id));
    return buffer;
}

ReadyForDataMessage ReadyForDataMessage::unpack(std::vector<std::uint8_t> const& data) {
    RAPIDSMPF_EXPECTS(data.size() == byte_size, "Invalid message size");
    ReadyForDataMessage msg;
    std::memcpy(&msg.message_id, data.data(), sizeof(msg.message_id));
    return msg;
}

std::string ReadyForDataMessage::to_string() const {
    return "ReadyForDataMessage{message_id=" + std::to_string(message_id) + "}";
}

}  // namespace rapidsmpf::shuffler
