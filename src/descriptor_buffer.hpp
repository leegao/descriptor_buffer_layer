#ifndef DESCRIPTOR_BUFFER_HPP
#define DESCRIPTOR_BUFFER_HPP

#include "descriptors.hpp"

#include <optional>
#include <vulkan/vulkan.h>

constexpr uint64_t kLayerMagic = 0xDEB0FEE5ULL;
constexpr size_t kDescriptorAlignment = 64;
constexpr uint32_t kInternalPoolMaxSets = 4096;
constexpr size_t kDescriptorSetCacheCapacity = 8192;

struct device;
struct command_buffer;

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint);
std::optional<VkBuffer> FindBufferForAddress(struct device *dev,
                                             VkDeviceAddress address);

#endif // DESCRIPTOR_BUFFER_HPP
