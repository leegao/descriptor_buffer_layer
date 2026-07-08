#ifndef PIPELINES_HPP
#define PIPELINES_HPP

#include <vector>
#include <vulkan/vulkan.h>

struct TrackedPipelineLayout {
    std::vector<VkDescriptorSetLayout> setLayouts;
};

struct TrackedPipeline {
    VkPipelineLayout layout;
    bool usesDescriptorBuffers;
};

#endif // PIPELINES_HPP
