/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk_message_adapter.hpp>

namespace rapidsmpf::shuffler {

ChunkMessageAdapter::ChunkMessageAdapter(detail::Chunk chunk)
    : chunk_(std::move(chunk)) {}

ChunkMessageAdapter::ChunkMessageAdapter(std::uint64_t message_id)
    : chunk_(detail::Chunk{}) {
    // Note: This constructor is used for deserialization - the actual chunk
    // will be reconstructed from metadata in a separate step
    (void)message_id;  // Suppress unused parameter warning
}

std::uint64_t ChunkMessageAdapter::message_id() const {
    return static_cast<std::uint64_t>(chunk_.chunk_id());
}

Rank ChunkMessageAdapter::peer_rank() const {
    return peer_rank_;
}

void ChunkMessageAdapter::set_peer_rank(Rank rank) {
    peer_rank_ = rank;
}

std::vector<std::uint8_t> ChunkMessageAdapter::serialize_metadata() const {
    return chunk_.serialize();
}

std::size_t ChunkMessageAdapter::total_data_size() const {
    return chunk_.concat_data_size();
}

bool ChunkMessageAdapter::is_data_ready() const {
    return chunk_.is_data_buffer_set();
}

void ChunkMessageAdapter::set_data_buffers(std::vector<std::unique_ptr<Buffer>> buffers) {
    // Chunk expects a single concatenated buffer
    RAPIDSMPF_EXPECTS(buffers.size() == 1, "Chunk expects single data buffer");
    chunk_.set_data_buffer(std::move(buffers[0]));
}

std::vector<std::unique_ptr<Buffer>> ChunkMessageAdapter::release_data_buffers() {
    std::vector<std::unique_ptr<Buffer>> buffers;
    if (chunk_.is_data_buffer_set()) {
        buffers.push_back(chunk_.release_data_buffer());
    }
    return buffers;
}

MemoryType ChunkMessageAdapter::data_memory_type() const {
    if (!chunk_.is_data_buffer_set()) {
        // Return a reasonable default - actual type will be known once buffer is set
        return MemoryType::DEVICE;
    }
    return chunk_.data_memory_type();
}

bool ChunkMessageAdapter::is_ready() const {
    return chunk_.is_ready();
}

std::string ChunkMessageAdapter::to_string() const {
    return chunk_.str();
}

ChunkMessageFactory::ChunkMessageFactory(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
)
    : allocate_buffer_fn_(std::move(allocate_buffer_fn)) {}

std::unique_ptr<MessageInterface> ChunkMessageFactory::create_from_metadata(
    std::vector<std::uint8_t> const& metadata
) const {
    auto chunk = detail::Chunk::deserialize(metadata, false);
    return std::make_unique<ChunkMessageAdapter>(std::move(chunk));
}

std::vector<std::unique_ptr<Buffer>> ChunkMessageFactory::allocate_receive_buffers(
    std::size_t total_size, MessageInterface const& /* message */
) const {
    std::vector<std::unique_ptr<Buffer>> buffers;
    if (total_size > 0) {
        buffers.push_back(allocate_buffer_fn_(total_size));
    }
    return buffers;
}

std::vector<std::unique_ptr<MessageInterface>> chunks_to_messages(
    std::vector<detail::Chunk>&& chunks
) {
    std::vector<std::unique_ptr<MessageInterface>> messages;
    messages.reserve(chunks.size());

    for (auto&& chunk : chunks) {
        messages.push_back(std::make_unique<ChunkMessageAdapter>(std::move(chunk)));
    }

    return messages;
}

std::vector<detail::Chunk> messages_to_chunks(
    std::vector<std::unique_ptr<MessageInterface>>&& messages
) {
    std::vector<detail::Chunk> chunks;
    chunks.reserve(messages.size());

    for (auto&& message : messages) {
        auto* adapter = dynamic_cast<ChunkMessageAdapter*>(message.get());
        RAPIDSMPF_EXPECTS(adapter != nullptr, "Message is not a ChunkMessageAdapter");
        chunks.push_back(adapter->release_chunk());
    }

    return chunks;
}

}  // namespace rapidsmpf::shuffler
