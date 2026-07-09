#include "descriptor_buffer.hpp"
#include "descriptors.hpp"
#include "layer.hpp"
#include "logger.hpp"

#include <cstring>

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

void VKAPI_CALL DescriptorBufferLayer_GetDescriptorEXT(
    VkDevice device, const VkDescriptorGetInfoEXT *pDescriptorInfo,
    size_t dataSize, void *pDescriptor) {
    struct device *dev = get_device(device);
    if (!pDescriptor || dataSize < sizeof(EmulatedDescriptor)) {
        Logger::log("error",
                    "vkGetDescriptorEXT failed: dataSize=%zu is too small",
                    dataSize);
        return;
    }

    Logger::log("info",
                "vkGetDescriptorEXT: device=%p, pDescriptorInfo=%p, "
                "dataSize=%zu, pDescriptor=%p",
                device, pDescriptorInfo, dataSize, pDescriptor);

    auto checkBufferHandle = [&](VkDeviceAddress address) -> void {
        if (!dev || address == 0) {
            Logger::log("error", "Failed to resolve buffer device address: %p",
                        address);
            return;
        }
        std::shared_lock<std::shared_mutex> lock(dev->db.mutex); // reader
        auto bufferOpt = FindBufferForAddress(dev, address);
        if (!bufferOpt) {
            Logger::log("error", "Failed to resolve buffer device address: %p",
                        address);
        }
    };

    EmulatedDescriptor emulatedDescriptor{
        .magic = kLayerMagic,
    };

    const auto &data = pDescriptorInfo->data;
    switch (pDescriptorInfo->type) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
        emulatedDescriptor.type = static_cast<uint32_t>(EmulatedType::kSampler);
        if (data.pSampler) {
            emulatedDescriptor.handle =
                reinterpret_cast<uint64_t>(*data.pSampler);
        }
        break;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        emulatedDescriptor.type =
            static_cast<uint32_t>(EmulatedType::kCombinedImageSampler);
        if (data.pCombinedImageSampler) {
            emulatedDescriptor.handle = reinterpret_cast<uint64_t>(
                data.pCombinedImageSampler->imageView);
            emulatedDescriptor.handle2 =
                reinterpret_cast<uint64_t>(data.pCombinedImageSampler->sampler);
            emulatedDescriptor.imageLayoutOrFormat =
                static_cast<uint32_t>(data.pCombinedImageSampler->imageLayout);
        }
        break;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        emulatedDescriptor.type = static_cast<uint32_t>(EmulatedType::kImage);
        {
            const VkDescriptorImageInfo *imgInfo = nullptr;
            if (pDescriptorInfo->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
                imgInfo = data.pSampledImage;
            } else if (pDescriptorInfo->type ==
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                imgInfo = data.pStorageImage;
            } else {
                imgInfo = data.pInputAttachmentImage;
            }
            if (imgInfo) {
                emulatedDescriptor.handle =
                    reinterpret_cast<uint64_t>(imgInfo->imageView);
                emulatedDescriptor.imageLayoutOrFormat =
                    static_cast<uint32_t>(imgInfo->imageLayout);
            }
        }
        break;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        emulatedDescriptor.type = static_cast<uint32_t>(EmulatedType::kBuffer);
        {
            const VkDescriptorAddressInfoEXT *bufInfo =
                (pDescriptorInfo->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    ? data.pUniformBuffer
                    : data.pStorageBuffer;
            if (bufInfo) {
                checkBufferHandle(bufInfo->address);
                emulatedDescriptor.handle = bufInfo->address;
                emulatedDescriptor.range = bufInfo->range;
            }
        }
        break;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        emulatedDescriptor.type =
            static_cast<uint32_t>(EmulatedType::kTexelBuffer);
        {
            const VkDescriptorAddressInfoEXT *bufInfo =
                (pDescriptorInfo->type ==
                 VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                    ? data.pUniformTexelBuffer
                    : data.pStorageTexelBuffer;
            if (bufInfo) {
                checkBufferHandle(bufInfo->address);
                emulatedDescriptor.handle = bufInfo->address;
                emulatedDescriptor.range = bufInfo->range;
                emulatedDescriptor.imageLayoutOrFormat =
                    static_cast<uint32_t>(bufInfo->format);
            }
        }
        break;
    default:
        emulatedDescriptor.type = static_cast<uint32_t>(EmulatedType::kNone);
        Logger::log("error", "Unknown descriptor type %d",
                    pDescriptorInfo->type);
        break;
    }

    std::memcpy(pDescriptor, &emulatedDescriptor, sizeof(EmulatedDescriptor));
}
