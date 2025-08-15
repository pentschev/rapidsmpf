/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <vector>

#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/message_interface.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Adapter that makes Chunk compatible with MessageInterface.
 *
 * This adapter allows existing Chunk-based code to work with the new generic
 * communication interface without requiring immediate changes to all Chunk usage.
 */
class ChunkMessageAdapter : public MessageInterface {
  public:
    /**
     * @brief Construct adapter from an existing Chunk.
     *
     * @param chunk The chunk to wrap.
     */
    explicit ChunkMessageAdapter(detail::Chunk chunk);

    /**
     * @brief Construct adapter with just message ID (for deserialization).
     *
     * @param message_id The message ID.
     */
    explicit ChunkMessageAdapter(std::uint64_t message_id);

    // MessageInterface implementation
    [[nodiscard]] std::uint64_t message_id() const override;
    [[nodiscard]] Rank peer_rank() const override;
    void set_peer_rank(Rank rank) override;
    [[nodiscard]] std::vector<std::uint8_t> serialize_metadata() const override;
    [[nodiscard]] std::size_t total_data_size() const override;
    [[nodiscard]] bool is_data_ready() const override;
    void set_data_buffers(std::vector<std::unique_ptr<Buffer>> buffers) override;
    [[nodiscard]] std::vector<std::unique_ptr<Buffer>> release_data_buffers() override;
    [[nodiscard]] MemoryType data_memory_type() const override;
    [[nodiscard]] bool is_ready() const override;
    [[nodiscard]] std::string to_string() const override;

    /**
     * @brief Get the underlying Chunk.
     *
     * @return Reference to the wrapped chunk.
     */
    [[nodiscard]] detail::Chunk& chunk() {
        return chunk_;
    }

    /**
     * @brief Get the underlying Chunk (const version).
     *
     * @return Const reference to the wrapped chunk.
     */
    [[nodiscard]] detail::Chunk const& chunk() const {
        return chunk_;
    }

    /**
     * @brief Release the underlying Chunk.
     *
     * @return The wrapped chunk with ownership transferred.
     */
    [[nodiscard]] detail::Chunk release_chunk() {
        return std::move(chunk_);
    }

  private:
    detail::Chunk chunk_;
    Rank peer_rank_{0};  // Will be set by communication interface
};

/**
 * @brief Factory for creating ChunkMessageAdapter instances.
 *
 * This factory allows the generic communication interface to create Chunk-based
 * messages without knowing about the Chunk type directly.
 */
class ChunkMessageFactory : public MessageFactory {
  public:
    /**
     * @brief Constructor.
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     */
    explicit ChunkMessageFactory(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    );

    [[nodiscard]] std::unique_ptr<MessageInterface> create_from_metadata(
        std::vector<std::uint8_t> const& metadata
    ) const override;

    [[nodiscard]] std::vector<std::unique_ptr<Buffer>> allocate_receive_buffers(
        std::size_t total_size, MessageInterface const& message
    ) const override;

  private:
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn_;
};

/**
 * @brief Helper function to convert Chunk vector to MessageInterface vector.
 *
 * @param chunks Vector of chunks to convert.
 * @return Vector of MessageInterface instances wrapping the chunks.
 */
[[nodiscard]] std::vector<std::unique_ptr<MessageInterface>> chunks_to_messages(
    std::vector<detail::Chunk>&& chunks
);

/**
 * @brief Helper function to convert MessageInterface vector back to Chunk vector.
 *
 * @param messages Vector of MessageInterface instances (must be ChunkMessageAdapter).
 * @return Vector of chunks extracted from the adapters.
 */
[[nodiscard]] std::vector<detail::Chunk> messages_to_chunks(
    std::vector<std::unique_ptr<MessageInterface>>&& messages
);

}  // namespace rapidsmpf::shuffler
