#include "buffer.hpp"
#include "descriptor_buffer.hpp"
#include "layer.hpp"

#include <atomic>
#include <unordered_map>

std::unordered_map<VkBuffer, std::unique_ptr<struct buffer>> buffersMap;
std::unordered_map<VkDeviceMemory, uint32_t> descriptorBufferMemoryCounts;

std::atomic<int> bufferIdCounter;

struct buffer *find_buffer(VkBuffer buffer) {
    auto it = buffersMap.find(buffer);

    if (it == buffersMap.end())
        return nullptr;

    return it->second.get();
}

void *get_host_pointer(struct device *dev, VkBuffer buffer) {
    struct buffer *buf = find_buffer(buffer);
    if (!buf)
        return nullptr;
    auto memIt = dev->db.mappedMemory.find(buf->memory);
    if (memIt == dev->db.mappedMemory.end())
        return nullptr;
    return static_cast<char *>(memIt->second) + buf->offset;
}

VkBufferView get_staging_buffer_view(struct device *dev, VkBuffer buffer,
                                     VkFormat format, VkDeviceSize offset,
                                     VkDeviceSize range) {
    BufferViewKey key{buffer, format, offset, range};
    {
        std::shared_lock<std::shared_mutex> lock(dev->db.mutex); // reader
        auto it = dev->db.bufferViews.find(key);
        if (it != dev->db.bufferViews.end()) {
            return it->second;
        }
    }

    VkBufferViewCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .buffer = buffer,
        .format = format,
        .offset = offset,
        .range = range};

    VkBufferView view = VK_NULL_HANDLE;
    VkResult result =
        dev->table.CreateBufferView(dev->handle, &createInfo, nullptr, &view);
    if (result != VK_SUCCESS) {
        Logger::log("error", "vkCreateBufferView failed, result: %d", result);
        return VK_NULL_HANDLE;
    }

    {
        std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
        auto it = dev->db.bufferViews.find(key);
        if (it != dev->db.bufferViews.end()) {
            dev->table.DestroyBufferView(dev->handle, view, nullptr);
            return it->second;
        }
        dev->db.bufferViews[key] = view;
    }

    return view;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    VkResult result;
    VkLayerDispatchTable table;
    VkBufferCreateInfo create_info = *pCreateInfo;

    struct device *dev = get_device(device);
    if (!dev)
        return VK_ERROR_INITIALIZATION_FAILED;

    table = dev->table;

    bool isDescriptorBuffer =
        (create_info.usage &
         (VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
          VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT)) != 0;

    VkBufferUsageFlags descriptorBufferBits =
        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
        VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;

    if (create_info.usage & descriptorBufferBits) {
        create_info.usage &= ~descriptorBufferBits;
        create_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    result = table.CreateBuffer(device, &create_info, pAllocator, pBuffer);
    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to create buffer, res %d", result);
        return result;
    }

    auto buf = std::make_unique<struct buffer>();
    buf->handle = *pBuffer;
    buf->size = pCreateInfo->size;
    buf->device = dev;
    buf->alloc = pAllocator;
    buf->id = 0;
    buf->isDescriptorBuffer = isDescriptorBuffer;

    {
        scoped_lock l(global_lock);
        buffersMap[*pBuffer] = std::move(buf);
    }

    return VK_SUCCESS;
}

static void BindBufferDeviceAddress(struct device *dev, VkBuffer buffer,
                                    struct buffer *buf) {
    if (!dev->table.GetBufferDeviceAddress) {
        Logger::log("error",
                    "GetBufferDeviceAddress not supported, cannot proceed");
        return;
    }

    VkBufferDeviceAddressInfo addrInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer};
    buf->deviceAddress =
        dev->table.GetBufferDeviceAddress(dev->handle, &addrInfo);
    if (!buf->deviceAddress) {
        Logger::log("error", "Buffer %p device address is nullptr", buffer);
        return;
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.addressRangeStarts[buf->deviceAddress] = buffer;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_BindBufferMemory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
    VkDeviceSize memoryOffset) {
    VkResult result;
    VkLayerDispatchTable table;

    scoped_lock l(global_lock);

    struct device *dev = get_device(device);
    if (!dev)
        return VK_ERROR_INITIALIZATION_FAILED;

    table = dev->table;

    result = table.BindBufferMemory(device, buffer, memory, memoryOffset);
    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to bind buffer memory, res %d", result);
        return result;
    }

    struct buffer *buf = find_buffer(buffer);
    if (!buf) {
        Logger::log("error", "BindBufferMemory: Buffer %p not found", buffer);
        return VK_SUCCESS;
    }

    buf->memory = memory;
    buf->offset = memoryOffset;
    if (buf->isDescriptorBuffer) {
        descriptorBufferMemoryCounts[memory]++;
    }
    BindBufferDeviceAddress(dev, buffer, buf);

    return VK_SUCCESS;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_BindBufferMemory2(
    VkDevice device, uint32_t bindInfoCount,
    const VkBindBufferMemoryInfo *pBindInfos) {
    VkResult result;
    VkLayerDispatchTable table;

    scoped_lock l(global_lock);

    struct device *dev = get_device(device);
    if (!dev)
        return VK_ERROR_INITIALIZATION_FAILED;

    table = dev->table;

    // Don't emulate with BindBufferMemory in case pBindInfos has a pNext
    result = table.BindBufferMemory2(device, bindInfoCount, pBindInfos);
    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to bind buffer memory, res %d", result);
        return result;
    }

    for (uint32_t i = 0; i < bindInfoCount; i++) {
        struct buffer *buf = find_buffer(pBindInfos[i].buffer);

        if (!buf) {
            Logger::log("error", "BindBufferMemory2: Buffer %p not found",
                        pBindInfos[i].buffer);
            continue;
        }
        buf->memory = pBindInfos[i].memory;
        buf->offset = pBindInfos[i].memoryOffset;
        if (buf->isDescriptorBuffer) {
            descriptorBufferMemoryCounts[pBindInfos[i].memory]++;
        }
        BindBufferDeviceAddress(dev, pBindInfos[i].buffer, buf);
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_DestroyBuffer(
    VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *pAllocator) {
    scoped_lock l(global_lock);

    struct device *dev = get_device(device);
    struct buffer *buf = find_buffer(buffer);
    if (!dev || !buf)
        return;

    if (buf->isDescriptorBuffer && buf->memory != VK_NULL_HANDLE) {
        auto it = descriptorBufferMemoryCounts.find(buf->memory);
        if (it != descriptorBufferMemoryCounts.end()) {
            it->second--;
            if (it->second <= 0) {
                descriptorBufferMemoryCounts.erase(it);
            }
        }
    }

    if (buf->deviceAddress != 0) {
        std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
        dev->db.addressRangeStarts.erase(buf->deviceAddress);
    }

    dev->table.DestroyBuffer(device, buffer, pAllocator);
    buffersMap.erase(buffer);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_MapMemory(
    VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
    VkDeviceSize size, VkMemoryMapFlags flags, void **ppData) {
    struct device *dev = get_device(device);

    {
        // If this is already cached (descriptor buffer memory), return early
        std::shared_lock<std::shared_mutex> lock(dev->db.mutex); // reader
        auto it = dev->db.mappedMemory.find(memory);
        if (it != dev->db.mappedMemory.end()) {
            *ppData = static_cast<char *>(it->second) + offset;
            return VK_SUCCESS;
        }
    }

    VkResult result =
        dev->table.MapMemory(device, memory, offset, size, flags, ppData);
    if (result != VK_SUCCESS) {
        Logger::log("error", "vkMapMemory failed, result: %d", result);
        return result;
    }

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    void *baseAddress = static_cast<char *>(*ppData) - offset;
    dev->db.mappedMemory[memory] = baseAddress;

    return result;
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    struct device *dev = get_device(device);

    // Keep descriptor buffer memory bound indefinitely for emulation
    bool isDescriptorMemory = false;
    {
        scoped_lock l(global_lock);
        if (descriptorBufferMemoryCounts.find(memory) !=
            descriptorBufferMemoryCounts.end()) {
            isDescriptorMemory = true;
        }
    }
    if (isDescriptorMemory) {
        return;
    }

    dev->table.UnmapMemory(device, memory);

    std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
    dev->db.mappedMemory.erase(memory);
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_FreeMemory(VkDevice device, VkDeviceMemory memory,
                                 const VkAllocationCallbacks *pAllocator) {
    if (memory == VK_NULL_HANDLE) {
        return;
    }

    struct device *dev = get_device(device);

    dev->table.FreeMemory(device, memory, pAllocator);

    {
        std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
        dev->db.mappedMemory.erase(memory);
    }

    {
        scoped_lock l(global_lock);
        descriptorBufferMemoryCounts.erase(memory);
    }
}

VK_LAYER_EXPORT VkDeviceAddress VKAPI_CALL
DescriptorBufferLayer_GetBufferDeviceAddress(
    VkDevice device, const VkBufferDeviceAddressInfo *pInfo) {
    struct device *dev = get_device(device);

    VkDeviceAddress address = dev->table.GetBufferDeviceAddress(device, pInfo);

    scoped_lock l(global_lock);
    struct buffer *buf = find_buffer(pInfo->buffer);
    if (buf) {
        std::unique_lock<std::shared_mutex> lock(dev->db.mutex); // writer
        if (buf->deviceAddress != 0) {
            dev->db.addressRangeStarts.erase(buf->deviceAddress);
        }
        buf->deviceAddress = address;
        dev->db.addressRangeStarts[address] = pInfo->buffer;
    }
    return address;
}
