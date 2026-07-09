#ifndef DESCRIPTOR_BUFFER_HPP
#define DESCRIPTOR_BUFFER_HPP

#include "descriptors.hpp"

#include <list>
#include <optional>
#include <unordered_map>
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

class DescriptorSetCache {
  public:
    using Key = std::pair<VkDescriptorSetLayout, uint64_t>;

    std::optional<VkDescriptorSet> Find(const Key &key);
    void Insert(const Key &key, VkDescriptorSet set);
    void Clear();

  private:
    struct KeyHash {
        std::size_t operator()(const Key &k) const {
            return std::hash<void *>{}(k.first) ^
                   (std::hash<uint64_t>{}(k.second) << 1);
        }
    };

    std::list<Key> lru;
    std::unordered_map<
        Key, std::pair<VkDescriptorSet, std::list<Key>::iterator>, KeyHash>
        map;
};

struct device;
struct command_buffer;

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint);
std::optional<VkBuffer> FindBufferForAddress(struct device *dev,
                                             VkDeviceAddress address);

#endif // DESCRIPTOR_BUFFER_HPP
