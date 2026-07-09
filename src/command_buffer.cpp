#include "command_buffer.hpp"

#include "descriptor_buffer.hpp"
#include "layer.hpp"
#include "logger.hpp"
#include "mali_gpu_profiler.hpp"
#include "pipeline_state.hpp"
#include "staging_resources.hpp"

std::unordered_map<VkCommandBuffer, std::shared_ptr<struct command_buffer>>
    commandBuffersMap;

struct command_buffer *get_command_buffer(VkCommandBuffer commandbuffer) {
    auto it = commandBuffersMap.find(commandbuffer);

    if (it == commandBuffersMap.end())
        return nullptr;

    return it->second.get();
}

VkResult DispatchOneShotAndSample(
    struct device *dev,
    std::function<void(struct command_buffer *)> record_func,
    const std::string_view shader_label) {
    VkResult result;
    const auto &table = dev->table;

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = dev->queueFamilyIndex};
    VkCommandPool command_pool;
    result = table.CreateCommandPool(dev->handle, &pool_info, nullptr,
                                     &command_pool);
    if (result != VK_SUCCESS) {
        Logger::log("error", "one_shot: vkCreateCommandPool failed, result: %d",
                    result);
        return result;
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    VkCommandBuffer commandBuffer;
    result =
        table.AllocateCommandBuffers(dev->handle, &alloc_info, &commandBuffer);
    if (result != VK_SUCCESS) {
        Logger::log("error",
                    "one_shot: vkAllocateCommandBuffers failed, result: %d",
                    result);
        table.DestroyCommandPool(dev->handle, command_pool, nullptr);
        return result;
    }

    // See
    // https://vulkan.lunarg.com/doc/view/latest/linux/LoaderLayerInterface.html#creating-new-dispatchable-objects
    // To fill in the dispatch table pointer in newly created dispatchable
    // object, the layer should copy the dispatch pointer, which is always the
    // first entry in the structure, from an existing parent object of the same
    // level (instance versus device).
    if (dev->has_more_layers) {
        *reinterpret_cast<void **>(commandBuffer) =
            *reinterpret_cast<void **>(dev->handle);
    }

    auto cb = std::make_shared<struct command_buffer>();
    cb->handle = commandBuffer;
    cb->device = dev;
    cb->pool = command_pool;
    cb->currentStagingResources =
        std::make_unique<StagingResources>(dev->handle);
    cb->reset_compute_state();

    {
        scoped_lock l(global_lock);
        commandBuffersMap[commandBuffer] = cb;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    result = table.BeginCommandBuffer(commandBuffer, &begin_info);
    if (result != VK_SUCCESS) {
        Logger::log("error",
                    "one_shot: vkBeginCommandBuffer failed, result: %d",
                    result);
        goto cleanup_registry;
    }

    record_func(cb.get());

    result = table.EndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        Logger::log("error", "one_shot: vkEndCommandBuffer failed, result: %d",
                    result);
        goto cleanup_registry;
    }

    {
        VkSubmitInfo submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers = &commandBuffer};

        VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        result = table.CreateFence(dev->handle, &fence_info, nullptr, &fence);
        if (result != VK_SUCCESS)
            goto cleanup_registry;

        auto now = std::chrono::system_clock::now();
        cb->currentStagingResources->timestamp =
            std::chrono::time_point_cast<std::chrono::milliseconds>(now)
                .time_since_epoch()
                .count();

        if (dev->sample_gpu_counters) {
            get_mali_gpu_profiler().Start();
        }

        result = table.QueueSubmit(dev->queue, 1, &submit_info, fence);
        if (result != VK_SUCCESS) {
            Logger::log("error", "one_shot: vkQueueSubmit failed, result: %d",
                        result);
        }
        table.WaitForFences(dev->handle, 1, &fence, VK_TRUE, UINT64_MAX);

        if (dev->sample_gpu_counters) {
            get_mali_gpu_profiler().StopAndProcess(shader_label);
        }

        table.DestroyFence(dev->handle, fence, nullptr);
    }

    if (cb->currentStagingResources) {
        cb->currentStagingResources->Cleanup();
    }

cleanup_registry: {
    scoped_lock l(global_lock);
    commandBuffersMap.erase(commandBuffer);
}
    table.FreeCommandBuffers(dev->handle, command_pool, 1, &commandBuffer);
    table.DestroyCommandPool(dev->handle, command_pool, nullptr);

    return result;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL
DescriptorBufferLayer_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers) {
    VkResult result;
    VkLayerDispatchTable table;

    struct device *dev = get_device(device);
    if (!dev)
        return VK_ERROR_INITIALIZATION_FAILED;

    table = dev->table;

    result =
        table.AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to allocate command buffers, res %d",
                    result);
        return result;
    }

    for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
        auto cmd = std::make_shared<struct command_buffer>();
        cmd->handle = pCommandBuffers[i];
        cmd->device = dev;
        cmd->pool = pAllocateInfo->commandPool;
        cmd->currentStagingResources =
            std::make_unique<StagingResources>(device);
        cmd->reset_compute_state();
        {
            scoped_lock l(global_lock);
            commandBuffersMap[pCommandBuffers[i]] = cmd;
        }

        if (dev->has_more_layers) {
            *reinterpret_cast<void **>(pCommandBuffers[i]) =
                *reinterpret_cast<void **>(device);
        }
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers) {
    scoped_lock l(global_lock);

    struct device *dev = get_device(device);
    if (!dev)
        return;

    for (uint32_t i = 0; i < commandBufferCount; i++) {
        struct command_buffer *cb = get_command_buffer(pCommandBuffers[i]);
        if (!cb)
            continue;

        dev->table.FreeCommandBuffers(dev->handle, commandPool, 1, &cb->handle);
        commandBuffersMap.erase(pCommandBuffers[i]);
    }
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    cb->reset_compute_state(); // begin/reset should clear inherited compute
                               // pipeline states

    return dev->table.BeginCommandBuffer(commandBuffer, pBeginInfo);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    cb->reset_compute_state(); // begin/reset should clear inherited compute
                               // pipeline states

    return dev->table.ResetCommandBuffer(commandBuffer, flags);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdBindPipeline(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipeline pipeline) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        cb->computePipelineState.TrackPipeline(pipeline);
    }

    {
        scoped_lock l(global_lock);
        cb->descriptorBufferState.TrackPipelineBindings(pipelineBindPoint,
                                                        pipeline);
    }

    dev->table.CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
    const uint32_t *pDynamicOffsets) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        cb->computePipelineState.TrackDescriptorSets(
            layout, firstSet, descriptorSetCount, pDescriptorSets,
            dynamicOffsetCount, pDynamicOffsets);
    }

    {
        // Untrack this if bound manually
        scoped_lock l(global_lock);
        for (uint32_t i = 0; i < descriptorSetCount; ++i) {
            uint32_t setIndex = firstSet + i;
            cb->descriptorBufferState.UntrackDescriptorSetBindings(setIndex);
        }
    }

    dev->table.CmdBindDescriptorSets(
        commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount,
        pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdBindDescriptorSets2(
    VkCommandBuffer commandBuffer,
    const VkBindDescriptorSetsInfo *pBindDescriptorSetsInfo) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
        cb->computePipelineState.TrackDescriptorSets(
            pBindDescriptorSetsInfo->layout, pBindDescriptorSetsInfo->firstSet,
            pBindDescriptorSetsInfo->descriptorSetCount,
            pBindDescriptorSetsInfo->pDescriptorSets,
            pBindDescriptorSetsInfo->dynamicOffsetCount,
            pBindDescriptorSetsInfo->pDynamicOffsets);
    }

    {
        // Untrack this if bound manually
        scoped_lock l(global_lock);
        for (int i = 0; i < pBindDescriptorSetsInfo->descriptorSetCount; ++i) {
            uint32_t setIndex = pBindDescriptorSetsInfo->firstSet + i;
            cb->descriptorBufferState.UntrackDescriptorSetBindings(setIndex);
        }
    }

    dev->table.CmdBindDescriptorSets2(commandBuffer, pBindDescriptorSetsInfo);
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_CmdBindDescriptorBuffersEXT(
    VkCommandBuffer commandBuffer, uint32_t bufferCount,
    const VkDescriptorBufferBindingInfoEXT *pBindingInfos) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log(
            "error",
            "Failed to get command buffer for CmdBindDescriptorBuffersEXT");
        return;
    }

    struct device *dev = cb->device;

    scoped_lock lock(global_lock);
    // reader (for FindBufferForAddress)
    std::shared_lock<std::shared_mutex> deviceLock(dev->db.mutex);
    auto &state = cb->descriptorBufferState;

    for (uint32_t i = 0; i < bufferCount; ++i) {
        if (i >= kMaxBoundSets) {
            Logger::log("error",
                        "vkCmdBindDescriptorBuffersEXT: bufferCount=%u exceeds "
                        "kMaxBoundSets=%u, ignoring remaining %u buffers",
                        bufferCount, kMaxBoundSets, bufferCount - i);
            break;
        }

        auto address = pBindingInfos[i].address;
        auto descriptorBuffer = FindBufferForAddress(dev, address);
        state.TrackDescriptorBufferBindings(
            i, descriptorBuffer.value_or(VK_NULL_HANDLE), address);
    }
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_CmdSetDescriptorBufferOffsetsEXT(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
    const uint32_t *pBufferIndices, const VkDeviceSize *pOffsets) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error", "Failed to get command buffer for "
                             "CmdSetDescriptorBufferOffsetsEXT");
        return;
    }

    scoped_lock lock(global_lock);
    auto &state = cb->descriptorBufferState;
    state.currentPipelineLayout = layout;

    for (uint32_t i = 0; i < setCount; ++i) {
        uint32_t setIndex = firstSet + i;
        if (setIndex >= kMaxBoundSets) {
            Logger::log(
                "error",
                "vkCmdSetDescriptorBufferOffsetsEXT: setIndex=%u exceeds "
                "kMaxBoundSets=%u, ignoring remaining %u sets",
                setIndex, kMaxBoundSets, setCount - i);
            break;
        }
        state.TrackDescriptorSetBindings(setIndex, pBufferIndices[i],
                                         pOffsets[i]);
    }
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdPushConstants(
    VkCommandBuffer commandBuffer, VkPipelineLayout layout,
    VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size,
    const void *pValues) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    cb->computePipelineState.TrackPushConstants(layout, stageFlags, offset,
                                                size, pValues);

    dev->table.CmdPushConstants(commandBuffer, layout, stageFlags, offset, size,
                                pValues);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdPushConstants2(
    VkCommandBuffer commandBuffer,
    const VkPushConstantsInfo *pPushConstantsInfo) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    struct device *dev = cb->device;

    cb->computePipelineState.TrackPushConstants(
        pPushConstantsInfo->layout, pPushConstantsInfo->stageFlags,
        pPushConstantsInfo->offset, pPushConstantsInfo->size,
        pPushConstantsInfo->pValues);

    dev->table.CmdPushConstants2(commandBuffer, pPushConstantsInfo);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdCopyBuffer(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
    uint32_t regionCount, const VkBufferCopy *pRegions) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "vkCmdCopyBuffer called on invalid command buffer");
        return;
    }

    struct buffer *buf = find_buffer(dstBuffer);
    if (buf && buf->isDescriptorBuffer) {
        Logger::log("error",
                    "vkCmdCopyBuffer on descriptor buffers not supported");
    }
    cb->device->table.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer,
                                    regionCount, pRegions);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdCopyBuffer2(
    VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "vkCmdCopyBuffer2 called on invalid command buffer");
        return;
    }

    struct buffer *buf = find_buffer(pCopyBufferInfo->dstBuffer);
    if (buf && buf->isDescriptorBuffer) {
        Logger::log("error",
                    "vkCmdCopyBuffer2 on descriptor buffers not supported");
    }
    cb->device->table.CmdCopyBuffer2(commandBuffer, pCopyBufferInfo);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdUpdateBuffer(
    VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
    VkDeviceSize dataSize, const void *pData) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "vkCmdUpdateBuffer called on invalid command buffer");
        return;
    }

    struct buffer *buf = find_buffer(dstBuffer);
    if (buf && buf->isDescriptorBuffer) {
        Logger::log("error",
                    "vkCmdUpdateBuffer on descriptor buffers not supported");
    }
    cb->device->table.CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset,
                                      dataSize, pData);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDraw(
    VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
    uint32_t firstVertex, uint32_t firstInstance) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error", "Failed to get command buffer for CmdDraw");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS);
    cb->device->table.CmdDraw(commandBuffer, vertexCount, instanceCount,
                              firstVertex, firstInstance);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDrawIndexed(
    VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error", "Failed to get command buffer for CmdDrawIndexed");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS);
    cb->device->table.CmdDrawIndexed(commandBuffer, indexCount, instanceCount,
                                     firstIndex, vertexOffset, firstInstance);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDrawIndirect(
    VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    uint32_t drawCount, uint32_t stride) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "Failed to get command buffer for CmdDrawIndirect");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS);
    cb->device->table.CmdDrawIndirect(commandBuffer, buffer, offset, drawCount,
                                      stride);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDrawIndexedIndirect(
    VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    uint32_t drawCount, uint32_t stride) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "Failed to get command buffer for CmdDrawIndexedIndirect");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS);
    cb->device->table.CmdDrawIndexedIndirect(commandBuffer, buffer, offset,
                                             drawCount, stride);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDispatch(
    VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY,
    uint32_t groupCountZ) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error", "Failed to get command buffer for CmdDispatch");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_COMPUTE);
    cb->device->table.CmdDispatch(commandBuffer, groupCountX, groupCountY,
                                  groupCountZ);
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_CmdDispatchIndirect(
    VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    struct command_buffer *cb = get_command_buffer(commandBuffer);
    if (!cb) {
        Logger::log("error",
                    "Failed to get command buffer for CmdDispatchIndirect");
        return;
    }
    ResolveAndBindDescriptorSets(cb->device, cb,
                                 VK_PIPELINE_BIND_POINT_COMPUTE);
    cb->device->table.CmdDispatchIndirect(commandBuffer, buffer, offset);
}
