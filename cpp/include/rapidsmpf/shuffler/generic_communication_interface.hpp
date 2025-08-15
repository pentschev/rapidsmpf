/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/message_interface.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Generic, type-agnostic communication interface.
 *
 * This interface provides a high-level, stateful communication management layer that
 * works with any message type implementing MessageInterface. It encapsulates the entire
 * communication protocol and state machine, providing coarse-grained operations that
 * can drive communication forward while maintaining full control over the underlying
 * communication patterns and optimizations.
 */
class GenericCommunicationInterface {
  public:
    virtual ~GenericCommunicationInterface() = default;

    /**
     * @brief Submit outgoing messages for communication.
     *
     * Takes ownership of ready messages and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * @param messages Vector of messages ready to be sent to remote ranks.
     * @param peer_rank_fn Function to determine destination rank for each message.
     * @param br Buffer resource for communication operations.
     */
    virtual void submit_outgoing_messages(
        std::vector<std::unique_ptr<MessageInterface>>&& messages,
        std::function<Rank(MessageInterface const&)> peer_rank_fn,
        BufferResource* br
    ) = 0;

    /**
     * @brief Process all pending communication operations.
     *
     * Advances the communication state machine by:
     * - Receiving incoming message metadata and setting up data transfers
     * - Processing ready-for-data acknowledgments and sending data
     * - Completing data transfers and making messages available
     * - Cleaning up completed operations
     *
     * @param message_factory Factory for creating message instances from metadata.
     * @param stream CUDA stream for memory operations.
     * @param br Buffer resource for communication operations.
     * @return Vector of completed messages ready for local processing.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<MessageInterface>>
    process_communication(
        MessageFactory const& message_factory,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) = 0;

    /**
     * @brief Check if all communication operations are complete.
     *
     * @return True if no pending operations remain, false otherwise.
     */
    [[nodiscard]] virtual bool is_idle() const = 0;
};

/**
 * @brief Tag-based implementation of GenericCommunicationInterface.
 *
 * This implementation provides the same communication protocol as
 * TagCommunicationInterface but works with the abstract MessageInterface instead of
 * concrete Chunk types.
 */
class TagGenericCommunicationInterface : public GenericCommunicationInterface {
  public:
    /**
     * @brief Constructor for TagGenericCommunicationInterface.
     *
     * @param comm The communicator to use for operations.
     * @param op_id The operation ID for tagging messages.
     * @param rank The current rank (for logging and validation).
     * @param statistics The statistics to use for tracking communication operations.
     */
    TagGenericCommunicationInterface(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        Rank rank,
        std::shared_ptr<Statistics> statistics
    );

    void submit_outgoing_messages(
        std::vector<std::unique_ptr<MessageInterface>>&& messages,
        std::function<Rank(MessageInterface const&)> peer_rank_fn,
        BufferResource* br
    ) override;

    std::vector<std::unique_ptr<MessageInterface>> process_communication(
        MessageFactory const& message_factory,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override;

    bool is_idle() const override;

  private:
    // Core communication infrastructure
    std::shared_ptr<Communicator> comm_;
    Rank rank_;
    Tag ready_for_data_tag_;
    Tag metadata_tag_;
    Tag gpu_data_tag_;

    // Communication state containers
    std::vector<std::unique_ptr<Communicator::Future>>
        fire_and_forget_;  ///< Ongoing "fire-and-forget" operations (non-blocking sends).
    std::multimap<Rank, std::unique_ptr<MessageInterface>>
        incoming_messages_;  ///< Messages ready to be received.
    std::unordered_map<std::uint64_t, std::unique_ptr<MessageInterface>>
        outgoing_messages_;  ///< Messages ready to be sent.
    std::unordered_map<std::uint64_t, std::unique_ptr<MessageInterface>>
        in_transit_messages_;  ///< Messages currently in transit.
    std::unordered_map<std::uint64_t, std::unique_ptr<Communicator::Future>>
        in_transit_futures_;  ///< Futures corresponding to in-transit messages.
    std::unordered_map<Rank, std::vector<std::unique_ptr<Communicator::Future>>>
        ready_ack_receives_;  ///< Receives matching ready for data messages.

    // Statistics tracking
    std::shared_ptr<Statistics> statistics_;

    // Helper methods for the communication protocol phases
    void send_metadata_phase(
        std::function<Rank(MessageInterface const&)> peer_rank_fn, BufferResource* br
    );
    void receive_metadata_phase(MessageFactory const& message_factory);
    void setup_data_receives_phase(
        MessageFactory const& message_factory,
        rmm::cuda_stream_view stream,
        BufferResource* br
    );
    void process_ready_acks_phase();
    std::vector<std::unique_ptr<MessageInterface>> complete_data_transfers_phase();
    void cleanup_completed_operations();
};

/**
 * @brief Message representing a ready-for-data acknowledgment.
 *
 * This is used internally by the communication protocol to coordinate data transfers.
 * When a receiver is ready to receive data for a specific message, it sends this
 * acknowledgment back to the sender containing the message ID.
 */
struct ReadyForDataMessage {
    /// @brief The unique ID of the message that the receiver is ready to receive data
    /// for.
    std::uint64_t message_id;

    /// @brief The size in bytes of the serialized message.
    static constexpr std::size_t byte_size = sizeof(std::uint64_t);

    /**
     * @brief Serialize the message into a byte vector.
     *
     * @return A vector containing the serialized message data.
     */
    [[nodiscard]] std::vector<std::uint8_t> pack() const;

    /**
     * @brief Deserialize a byte vector into a ReadyForDataMessage.
     *
     * @param data The serialized message data.
     * @return The deserialized ReadyForDataMessage.
     * @throws std::runtime_error if the data size is incorrect.
     */
    static ReadyForDataMessage unpack(std::vector<std::uint8_t> const& data);

    /**
     * @brief Create a string representation of the message for debugging.
     *
     * @return A string describing the message contents.
     */
    [[nodiscard]] std::string to_string() const;
};

}  // namespace rapidsmpf::shuffler
