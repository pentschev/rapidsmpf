/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

/**
 * @brief A device memory resource wrapper for rapidsmpf::PinnedMemoryResource.
 *
 * This wrapper allows the rapidsmpf pinned memory resource to be used as a device
 * memory resource in the RMM ecosystem. The underlying resource allocates pinned
 * host memory that is accessible from device code.
 */
class RapidsMPFPinnedDeviceMemoryResource : public rmm::mr::device_memory_resource {
  public:
    RapidsMPFPinnedDeviceMemoryResource() {
        if (!rapidsmpf::is_pinned_memory_resources_supported()) {
            throw std::runtime_error(
                "RapidsMPF pinned memory resource is not supported on this CUDA version"
            );
        }
        pool_ = std::make_unique<rapidsmpf::PinnedMemoryPool>();
        resource_ = std::make_unique<rapidsmpf::PinnedMemoryResource>(*pool_);
    }

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        return resource_->allocate(stream, bytes);
    }

    void do_deallocate(
        void* ptr, std::size_t bytes, rmm::cuda_stream_view stream
    ) noexcept override {
        resource_->deallocate(stream, ptr, bytes);
    }

    bool do_is_equal(
        const rmm::mr::device_memory_resource& other
    ) const noexcept override {
        auto const* other_wrapper =
            dynamic_cast<const RapidsMPFPinnedDeviceMemoryResource*>(&other);
        return other_wrapper != nullptr
               && resource_.get() == other_wrapper->resource_.get();
    }

  private:
    std::unique_ptr<rapidsmpf::PinnedMemoryPool> pool_;
    std::unique_ptr<rapidsmpf::PinnedMemoryResource> resource_;
};

/**
 * @brief Create and set a RMM stack as the current device memory resource.
 *
 * @param name The name of the stack:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a memory pool backed by a CUDA managed memory resource.
 *  - `pinned`: use the rapidsmpf pinned memory resource.
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rmm::mr::device_memory_resource>
set_current_rmm_stack(std::string const& name) {
    std::shared_ptr<rmm::mr::device_memory_resource> ret;
    if (name == "cuda") {
        ret = std::make_shared<rmm::mr::cuda_memory_resource>();
    } else if (name == "async") {
        ret = std::make_shared<rmm::mr::cuda_async_memory_resource>();
    } else if (name == "managed") {
        ret = std::make_shared<rmm::mr::managed_memory_resource>();
    } else if (name == "pool") {
        ret = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
            std::make_shared<rmm::mr::cuda_memory_resource>(),
            rmm::percent_of_free_device_memory(80),
            rmm::percent_of_free_device_memory(80)
        );
    } else if (name == "pinned") {
        ret = std::make_shared<RapidsMPFPinnedDeviceMemoryResource>();
    } else {
        RAPIDSMPF_FAIL("unknown RMM stack name: " + name);
    }
    // Note, RMM maintains two default resources, we set both here.
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}

/**
 * @brief Create a statistics-enabled device memory resource with on the current RMM
 * stack.
 *
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rapidsmpf::RmmResourceAdaptor>
set_device_mem_resource_with_stats() {
    auto ret = std::make_shared<rapidsmpf::RmmResourceAdaptor>(
        cudf::get_current_device_resource_ref()
    );
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}
