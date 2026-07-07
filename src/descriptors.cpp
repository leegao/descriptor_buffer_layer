#include "descriptors.hpp"
#include "layer.hpp"

VK_LAYER_EXPORT VkResult VKAPI_CALL
DescriptorBufferLayer_CreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDescriptorSetLayout *pSetLayout) {
    struct device *dev = get_device(device);

    VkDescriptorSetLayoutCreateInfo createInfo = *pCreateInfo;
    createInfo.flags &=
        ~VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;

    VkResult result = dev->table.CreateDescriptorSetLayout(
        device, &createInfo, pAllocator, pSetLayout);
    if (result != VK_SUCCESS) {
        Logger::log("error", "vkCreateDescriptorSetLayout failed, result: %d",
                    result);
        return result;
    }

    TrackedDescriptorSetLayout descriptorSetLayout{
        .realLayout = *pSetLayout,
    };

    for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
        descriptorSetLayout.bindings.push_back({
            .binding = pCreateInfo->pBindings[i].binding,
            .descriptorType = pCreateInfo->pBindings[i].descriptorType,
            .descriptorCount = pCreateInfo->pBindings[i].descriptorCount,
        });

        descriptorSetLayout.totalSize +=
            pCreateInfo->pBindings[i].descriptorCount * kDescriptorSize;
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.descriptorSetLayouts[*pSetLayout] = std::move(descriptorSetLayout);

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_DestroyDescriptorSetLayout(
    VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
    const VkAllocationCallbacks *pAllocator) {
    struct device *dev = get_device(device);

    dev->table.DestroyDescriptorSetLayout(device, descriptorSetLayout,
                                          pAllocator);

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.descriptorSetLayouts.erase(descriptorSetLayout);
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_GetDescriptorSetLayoutSizeEXT(
    VkDevice device, VkDescriptorSetLayout layout, VkDeviceSize *pSize) {
    struct device *dev = get_device(device);

    std::shared_lock<std::shared_mutex> lock(dev->db.mutex); // reader
    auto it = dev->db.descriptorSetLayouts.find(layout);
    if (it != dev->db.descriptorSetLayouts.end()) {
        const auto &descriptorSetLayout = it->second;
        *pSize = descriptorSetLayout.totalSize;
    } else {
        *pSize = 0;
    }
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_GetDescriptorSetLayoutBindingOffsetEXT(
    VkDevice device, VkDescriptorSetLayout layout, uint32_t binding,
    VkDeviceSize *pOffset) {
    struct device *dev = get_device(device);

    std::shared_lock<std::shared_mutex> lock(dev->db.mutex); // reader
    auto it = dev->db.descriptorSetLayouts.find(layout);
    if (it != dev->db.descriptorSetLayouts.end()) {
        const auto &descriptorSetLayout = it->second;
        VkDeviceSize offset = 0;
        for (const auto &b : descriptorSetLayout.bindings) {
            if (b.binding == binding) {
                *pOffset = offset;
                return;
            }
            offset += b.descriptorCount * kDescriptorSize;
        }
        *pOffset = 0;
    } else {
        *pOffset = 0;
    }
}
