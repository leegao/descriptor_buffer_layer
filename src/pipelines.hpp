#ifndef PIPELINES_HPP
#define PIPELINES_HPP

#include <vector>
#include <vulkan/vulkan.h>

struct TrackedPipelineLayout {
    std::vector<VkDescriptorSetLayout> setLayouts;
};

#endif // PIPELINES_HPP
