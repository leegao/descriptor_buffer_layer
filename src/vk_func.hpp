#ifndef __VK_FUNC_HPP
#define __VK_FUNC_HPP

#include "vulkan/vk_layer.h"
#include <vulkan/vulkan.h>

void init_dispatch_table(PFN_vkGetDeviceProcAddr, VkDevice,
                         VkLayerDispatchTable &);

extern "C" {

VkResult VKAPI_CALL DescriptorBufferLayer_CreateImage(
    VkDevice device, const VkImageCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImage *pImage);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pImageView);

void VKAPI_CALL DescriptorBufferLayer_DestroyImage(
    VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);

VkResult VKAPI_CALL DescriptorBufferLayer_BindBufferMemory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
    VkDeviceSize memoryOffset);

VkResult VKAPI_CALL DescriptorBufferLayer_BindBufferMemory2(
    VkDevice device, uint32_t bindInfoCount,
    const VkBindBufferMemoryInfo *pBindInfos);

void VKAPI_CALL DescriptorBufferLayer_DestroyBuffer(
    VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *pAllocator);

VkResult VKAPI_CALL DescriptorBufferLayer_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers);

void VKAPI_CALL DescriptorBufferLayer_FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers);

VkResult VKAPI_CALL DescriptorBufferLayer_BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo);

VkResult VKAPI_CALL DescriptorBufferLayer_ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);

void VKAPI_CALL DescriptorBufferLayer_CmdBindPipeline(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipeline pipeline);

void VKAPI_CALL DescriptorBufferLayer_CmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
    const uint32_t *pDynamicOffsets);

void VKAPI_CALL DescriptorBufferLayer_CmdBindDescriptorSets2(
    VkCommandBuffer commandBuffer,
    const VkBindDescriptorSetsInfo *pBindDescriptorSetsInfo);

void VKAPI_CALL DescriptorBufferLayer_CmdPushConstants(
    VkCommandBuffer commandBuffer, VkPipelineLayout layout,
    VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size,
    const void *pValues);

void VKAPI_CALL DescriptorBufferLayer_CmdPushConstants2(
    VkCommandBuffer commandBuffer,
    const VkPushConstantsInfo *pPushConstantsInfo);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyBufferToImage(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount,
    const VkBufferImageCopy *pRegions);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyBufferToImage2(
    VkCommandBuffer commandBuffer,
    const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyImage2(
    VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyImage(
    VkCommandBuffer commandBuffer, VkImage srcImage,
    VkImageLayout srcImageLayout, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount,
    const VkImageCopy *pRegions);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyBuffer(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
    uint32_t regionCount, const VkBufferCopy *pRegions);

void VKAPI_CALL DescriptorBufferLayer_CmdCopyBuffer2(
    VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo);

void VKAPI_CALL DescriptorBufferLayer_CmdUpdateBuffer(
    VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
    VkDeviceSize dataSize, const void *pData);

void VKAPI_CALL DescriptorBufferLayer_GetDeviceQueue(VkDevice device,
                                                     uint32_t queueFamilyIndex,
                                                     uint32_t queueIndex,
                                                     VkQueue *pQueue);

VkResult VKAPI_CALL DescriptorBufferLayer_QueueSubmit(
    VkQueue queue, uint32_t submitInfoCount, const VkSubmitInfo *pSubmitInfos,
    VkFence fence);

VkResult VKAPI_CALL DescriptorBufferLayer_QueueSubmit2(
    VkQueue queue, uint32_t submitInfoCount, const VkSubmitInfo2 *pSubmitInfos,
    VkFence fence);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout);

void VKAPI_CALL DescriptorBufferLayer_DestroyDescriptorSetLayout(
    VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
    const VkAllocationCallbacks *pAllocator);

void VKAPI_CALL DescriptorBufferLayer_GetDescriptorSetLayoutSizeEXT(
    VkDevice device, VkDescriptorSetLayout layout, VkDeviceSize *pSize);

void VKAPI_CALL DescriptorBufferLayer_GetDescriptorSetLayoutBindingOffsetEXT(
    VkDevice device, VkDescriptorSetLayout layout, uint32_t binding,
    VkDeviceSize *pOffset);

VkResult VKAPI_CALL DescriptorBufferLayer_CreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout);

void VKAPI_CALL DescriptorBufferLayer_DestroyPipelineLayout(
    VkDevice device, VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks *pAllocator);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines);

void VKAPI_CALL
DescriptorBufferLayer_DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                      const VkAllocationCallbacks *pAllocator);

VkResult VKAPI_CALL DescriptorBufferLayer_MapMemory(
    VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
    VkDeviceSize size, VkMemoryMapFlags flags, void **ppData);

void VKAPI_CALL DescriptorBufferLayer_UnmapMemory(VkDevice device,
                                                  VkDeviceMemory memory);

VkDeviceAddress VKAPI_CALL DescriptorBufferLayer_GetBufferDeviceAddress(
    VkDevice device, const VkBufferDeviceAddressInfo *pInfo);

VkResult VKAPI_CALL DescriptorBufferLayer_CreateFence(
    VkDevice device, const VkFenceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkFence *pFence);

VkResult VKAPI_CALL DescriptorBufferLayer_WaitForFences(VkDevice device,
                                                        uint32_t fenceCount,
                                                        const VkFence *pFences,
                                                        VkBool32 waitAll,
                                                        uint64_t timeout);

void VKAPI_CALL DescriptorBufferLayer_DestroyFence(
    VkDevice device, VkFence fence, const VkAllocationCallbacks *pAllocator);
}

#endif
