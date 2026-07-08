#ifndef __BUFFER_HPP
#define __BUFFER_HPP
#include <vulkan/vulkan.h>

#include <string_view>
#include <unordered_map>

struct device;

struct buffer {
    VkBuffer handle;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkDeviceSize offset;
    VkDeviceAddress deviceAddress = 0;
    bool isDescriptorBuffer = false;

    struct device *device;
    const VkAllocationCallbacks *alloc;
    std::string_view label;
    int id;
};

struct BufferViewKey {
    VkBuffer buffer;
    VkFormat format;
    VkDeviceSize offset;
    VkDeviceSize range;

    bool operator==(const BufferViewKey &o) const {
        return buffer == o.buffer && format == o.format && offset == o.offset &&
               range == o.range;
    }
};

struct BufferViewKeyHash {
    std::size_t operator()(const BufferViewKey &k) const {
        return std::hash<void *>{}(k.buffer) ^
               (std::hash<uint32_t>{}(static_cast<uint32_t>(k.format)) << 1) ^
               (std::hash<uint64_t>{}(k.offset) << 2) ^
               (std::hash<uint64_t>{}(k.range) << 3);
    }
};

using BufferViewCache =
    std::unordered_map<BufferViewKey, VkBufferView, BufferViewKeyHash>;

struct buffer *find_buffer(VkBuffer);

void *get_host_pointer(struct device *dev, VkBuffer buffer);

VkBufferView get_staging_buffer_view(struct device *dev, VkBuffer buffer,
                                     VkFormat format, VkDeviceSize offset,
                                     VkDeviceSize range);

#endif
