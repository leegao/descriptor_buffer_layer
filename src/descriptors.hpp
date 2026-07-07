#ifndef DESCRIPTORS_HPP
#define DESCRIPTORS_HPP

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

constexpr size_t kDescriptorSize = 64;

struct TrackedDescriptorSetLayoutBinding {
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
};

struct TrackedDescriptorSetLayout {
    VkDescriptorSetLayout realLayout = VK_NULL_HANDLE;
    std::vector<TrackedDescriptorSetLayoutBinding> bindings;
    VkDeviceSize totalSize = 0;
};

struct TrackedPipelineLayout {
    std::vector<VkDescriptorSetLayout> setLayouts;
};

#endif // DESCRIPTORS_HPP
