#ifndef __BUFFER_HPP
#define __BUFFER_HPP
#include <vulkan/vulkan.h>

#include <memory>
#include <string_view>

struct device;

struct buffer {
    VkBuffer handle;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkDeviceSize offset;
    struct device *device;
    const VkAllocationCallbacks *alloc;
    std::string_view label;
    int id;
};

struct buffer *find_buffer(VkBuffer);

#endif
