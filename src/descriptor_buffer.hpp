#ifndef DESCRIPTOR_BUFFER_HPP
#define DESCRIPTOR_BUFFER_HPP

#include "descriptors.hpp"

#include <optional>
#include <vulkan/vulkan.h>

constexpr uint64_t kLayerMagic = 0xDEB0FEE5ULL;
constexpr size_t kDescriptorAlignment = 64;
constexpr uint32_t kInternalPoolMaxSets = 4096;
constexpr size_t kDescriptorSetCacheCapacity = 8192;

enum class EmulatedType : uint32_t {
    kNone = 0,
    kBuffer = 1,
    kImage = 2,
    kSampler = 3,
    kCombinedImageSampler = 4,
    kTexelBuffer = 5,
};

struct alignas(8) EmulatedDescriptor {
    uint32_t magic = 0;
    uint32_t type = 0;
    uint32_t imageLayoutOrFormat = 0;
    uint32_t padding = 0;
    uint64_t handle = 0;
    uint64_t handle2 = 0;
    uint64_t range = 0;
};
static_assert(sizeof(EmulatedDescriptor) <= kDescriptorSize,
              "EmulatedDescriptor must fit inside 64-byte layout size.");

struct device;
struct command_buffer;

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint);
std::optional<VkBuffer> FindBufferForAddress(struct device *dev,
                                             VkDeviceAddress address);

#endif // DESCRIPTOR_BUFFER_HPP
