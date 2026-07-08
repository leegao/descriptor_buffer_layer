#include "layer.hpp"
#include "pipelines.hpp"

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_CreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkPipelineLayout *pPipelineLayout) {
    struct device *dev = get_device(device);

    VkResult result = dev->table.CreatePipelineLayout(
        device, pCreateInfo, pAllocator, pPipelineLayout);
    if (result != VK_SUCCESS) {
        Logger::log("error", "vkCreatePipelineLayout failed: %d", result);
        return result;
    }

    TrackedPipelineLayout pipelineLayout{};
    for (uint32_t i = 0; i < pCreateInfo->setLayoutCount; ++i) {
        pipelineLayout.setLayouts.push_back(pCreateInfo->pSetLayouts[i]);
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.pipelineLayouts[*pPipelineLayout] = std::move(pipelineLayout);

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_DestroyPipelineLayout(
    VkDevice device, VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks *pAllocator) {
    struct device *dev = get_device(device);

    dev->table.DestroyPipelineLayout(device, pipelineLayout, pAllocator);

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.pipelineLayouts.erase(pipelineLayout);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL
DescriptorBufferLayer_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    struct device *dev = get_device(device);

    // Strip away VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
    std::vector<VkGraphicsPipelineCreateInfo> createInfos(
        pCreateInfos, pCreateInfos + createInfoCount);
    std::vector<bool> usesDescriptorBuffers(createInfoCount);
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        usesDescriptorBuffers[i] =
            (createInfos[i].flags &
             VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) != 0;
        createInfos[i].flags &= ~VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    VkResult result = dev->table.CreateGraphicsPipelines(
        device, pipelineCache, createInfoCount, createInfos.data(), pAllocator,
        pPipelines);

    if (result != VK_SUCCESS && result != VK_PIPELINE_COMPILE_REQUIRED) {
        Logger::log("error", "vkCreateGraphicsPipelines failed, result: %d",
                    result);
        return result;
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            dev->db.pipelines[pPipelines[i]] = {createInfos[i].layout,
                                                usesDescriptorBuffers[i]};
        }
    }

    return result;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL
DescriptorBufferLayer_CreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    struct device *dev = get_device(device);

    // Strip away VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
    std::vector<VkComputePipelineCreateInfo> createInfos(
        pCreateInfos, pCreateInfos + createInfoCount);
    std::vector<bool> usesDescriptorBuffers(createInfoCount);
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        usesDescriptorBuffers[i] =
            (createInfos[i].flags &
             VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) != 0;
        createInfos[i].flags &= ~VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    VkResult result = dev->table.CreateComputePipelines(
        device, pipelineCache, createInfoCount, createInfos.data(), pAllocator,
        pPipelines);

    if (result != VK_SUCCESS && result != VK_PIPELINE_COMPILE_REQUIRED) {
        Logger::log("error", "vkCreateComputePipelines failed, result: %d",
                    result);
        return result;
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    for (uint32_t i = 0; i < createInfoCount; ++i) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            dev->db.pipelines[pPipelines[i]] = {createInfos[i].layout,
                                                usesDescriptorBuffers[i]};
        }
    }

    return result;
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                      const VkAllocationCallbacks *pAllocator) {
    struct device *dev = get_device(device);

    dev->table.DestroyPipeline(device, pipeline, pAllocator);

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.pipelines.erase(pipeline);
}
