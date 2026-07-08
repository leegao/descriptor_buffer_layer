#ifndef PIPELINE_STATE_HPP
#define PIPELINE_STATE_HPP

#include "logger.hpp"
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

constexpr uint32_t kMaxTrackedDescriptorSets = 32;
constexpr uint32_t kMaxTrackedPushConstantBytes = 256;
constexpr uint32_t kMaxTrackedDynamicOffsets = 8;

constexpr uint32_t kMaxBoundSets = 64;

struct BoundDescriptorSet {
    bool valid = false;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t dynamicOffsetCount = 0;
    std::array<uint32_t, kMaxTrackedDynamicOffsets> dynamicOffsets{};
};

struct PushConstantRange {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderStageFlags stageFlags = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
};

class ComputePipelineBindingsState {
  public:
    void reset() {
#ifdef ENABLE_COMPUTE_TRACKING
        pipelineBound = false;
        pipeline = VK_NULL_HANDLE;
        for (auto &s : sets)
            s = BoundDescriptorSet{};
        anyPushConstantsPushed = false;
        pushConstantRanges.clear();
#endif
    }

    void TrackPipeline(VkPipeline pipeline);

    void TrackPushConstants(VkPipelineLayout layout,
                            VkShaderStageFlags stageFlags, uint32_t offset,
                            uint32_t size, const void *pValues);

    void TrackDescriptorSets(VkPipelineLayout layout, uint32_t firstSet,
                             uint32_t descriptorSetCount,
                             const VkDescriptorSet *pDescriptorSets,
                             uint32_t dynamicOffsetCount,
                             const uint32_t *pDynamicOffsets);

  private:
    bool pipelineBound = false;
    VkPipeline pipeline = VK_NULL_HANDLE;
#ifdef ENABLE_COMPUTE_TRACKING
    std::array<BoundDescriptorSet, kMaxTrackedDescriptorSets> sets{};
    std::array<uint8_t, kMaxTrackedPushConstantBytes> pushConstantBytes{};
    bool anyPushConstantsPushed = false;
    std::vector<PushConstantRange> pushConstantRanges;
#endif
};

// Any injected compute pipeline must save and restore a snapshot of the
// pipeline state for the command buffer, this is done automatically by
// ScopedPipelineStateSnapshot
class ScopedPipelineStateSnapshot {
  public:
    explicit ScopedPipelineStateSnapshot(struct command_buffer *cb);
    ~ScopedPipelineStateSnapshot();

    // No copies or moves, this is purely a scope-guard.
    ScopedPipelineStateSnapshot(const ScopedPipelineStateSnapshot &) = delete;
    ScopedPipelineStateSnapshot &
    operator=(const ScopedPipelineStateSnapshot &) = delete;
    ScopedPipelineStateSnapshot(ScopedPipelineStateSnapshot &&other) = delete;
    ScopedPipelineStateSnapshot &
    operator=(ScopedPipelineStateSnapshot &&other) = delete;

  private:
    struct command_buffer *m_cb;
    ComputePipelineBindingsState m_snapshot;
};

struct DescriptorBufferBindingsState {
    VkPipelineLayout currentPipelineLayout = VK_NULL_HANDLE;
    VkPipeline currentGraphicsPipeline = VK_NULL_HANDLE;
    VkPipeline currentComputePipeline = VK_NULL_HANDLE;

    std::array<bool, kMaxBoundSets> setIsBound{};
    std::array<uint32_t, kMaxBoundSets> bufferIndexForSet{};
    std::array<VkDeviceSize, kMaxBoundSets> offsetForSet{};

    std::array<VkBuffer, kMaxBoundSets> boundBuffersByBindingIndex{};
    std::array<VkDeviceAddress, kMaxBoundSets>
        boundBufferAddressesByBindingIndex{};

    void Reset() {
        currentPipelineLayout = VK_NULL_HANDLE;
        currentGraphicsPipeline = VK_NULL_HANDLE;
        currentComputePipeline = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < kMaxBoundSets; ++i) {
            setIsBound[i] = false;
            bufferIndexForSet[i] = 0;
            offsetForSet[i] = 0;
            boundBuffersByBindingIndex[i] = VK_NULL_HANDLE;
            boundBufferAddressesByBindingIndex[i] = 0;
        }
    }

    void TrackPipelineBindings(VkPipelineBindPoint bindPoint,
                               VkPipeline pipeline) {
        if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
            currentGraphicsPipeline = pipeline;
        } else if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
            currentComputePipeline = pipeline;
        }
    }

    void TrackDescriptorSetBindings(uint32_t setIndex, uint32_t bufferIndex,
                                    VkDeviceSize offset) {
        if (setIndex < kMaxBoundSets) {
            setIsBound[setIndex] = true;
            bufferIndexForSet[setIndex] = bufferIndex;
            offsetForSet[setIndex] = offset;
        } else {
            Logger::log("error",
                        "TrackDescriptorSetBindings: setIndex=%u exceeds "
                        "kMaxBoundSets=%u",
                        setIndex, kMaxBoundSets);
        }
    }

    void UntrackDescriptorSetBindings(uint32_t setIndex) {
        if (setIndex < kMaxBoundSets) {
            setIsBound[setIndex] = false;
            bufferIndexForSet[setIndex] = 0;
            offsetForSet[setIndex] = 0;
            boundBuffersByBindingIndex[setIndex] = VK_NULL_HANDLE;
        } else {
            Logger::log("error",
                        "UntrackDescriptorSetBindings: setIndex=%u exceeds "
                        "kMaxBoundSets=%u",
                        setIndex, kMaxBoundSets);
        }
    }

    void TrackDescriptorBufferBindings(uint32_t setIndex, VkBuffer buffer,
                                       VkDeviceAddress address) {
        if (buffer == VK_NULL_HANDLE || address == 0) {
            Logger::log("error",
                        "TrackDescriptorBufferBindings: invalid buffer %u, "
                        "address 0x%lx",
                        buffer, address);
            return;
        }

        if (setIndex < kMaxBoundSets) {
            boundBuffersByBindingIndex[setIndex] = buffer;
            boundBufferAddressesByBindingIndex[setIndex] = address;
        } else {
            Logger::log("error",
                        "TrackDescriptorBufferAddress: setIndex=%u exceeds "
                        "kMaxBoundSets=%u",
                        setIndex, kMaxBoundSets);
        }
    }
};

#endif // PIPELINE_STATE_HPP
