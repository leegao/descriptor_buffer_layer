#include "buffer.hpp"
#include "command_buffer.hpp"
#include "descriptor_buffer.hpp"
#include "descriptors.hpp"
#include "layer.hpp"
#include "logger.hpp"
#include "pipelines.hpp"

#include <cstdint>
#include <cstring>
#include <deque>
#include <variant>

DescriptorSetCache &GetTLDescriptorSetCache(device *dev) {
    thread_local std::unordered_map<uint32_t, DescriptorSetCache> tlsCaches;
    return tlsCaches[dev->deviceId];
}

size_t hash_bytes(const void *data, size_t size) {
    std::string_view sv(reinterpret_cast<const char *>(data), size);
    return std::hash<std::string_view>{}(sv);
}

std::optional<VkDescriptorSet> DescriptorSetCache::Find(const Key &key) {
    auto it = map.find(key);
    if (it == map.end())
        return std::nullopt;
    lru.splice(lru.begin(), lru, it->second.second);
    return it->second.first;
}

void DescriptorSetCache::Insert(const Key &key, VkDescriptorSet set) {
    if (auto it = map.find(key); it != map.end()) {
        it->second.first = set;
        lru.splice(lru.begin(), lru, it->second.second);
        return;
    }

    if (map.size() >= kDescriptorSetCacheCapacity) {
        // TODO: clean up and free the descriptors associated with the evicted
        // entry
        // Key oldestKey = lru.back();
        // map.erase(oldestKey);
        // lru.pop_back();
    }

    lru.push_front(key);
    map.emplace(key, std::make_pair(set, lru.begin()));
}

void DescriptorSetCache::Clear() {
    // TODO: clean up during vkDestroyDevice
    lru.clear();
    map.clear();
}

std::optional<VkBuffer> FindBufferForAddress(struct device *dev,
                                             VkDeviceAddress address) {
    auto it = dev->db.addressRangeStarts.upper_bound(address); // O(logn)
    if (it == dev->db.addressRangeStarts.begin())
        return std::nullopt;
    --it;

    const VkDeviceAddress base = it->first;
    const VkBuffer buffer = it->second;

    struct buffer *buf = find_buffer(buffer);
    if (!buf)
        return std::nullopt;

    if (address < base || address >= base + buf->size)
        return std::nullopt;
    return buffer;
}

using VkDescriptorInfoStorage =
    std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo, VkBufferView>;

// Must be called with a shared_lock on dev->db.mutex
static void
AddUpdate(struct device *dev, VkDescriptorSet dstSet, uint32_t binding,
          uint32_t element, VkDescriptorType descriptorType,
          const EmulatedDescriptor &descriptor,
          std::vector<VkWriteDescriptorSet> &updates,
          std::deque<VkDescriptorInfoStorage> &descriptorInfoStorage) {

    VkWriteDescriptorSet updateDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dstSet,
        .dstBinding = binding,
        .dstArrayElement = element,
        .descriptorCount = 1,
        .descriptorType = descriptorType,
    };

    if (static_cast<EmulatedType>(descriptor.type) == EmulatedType::kNone) {
        Logger::log("error", "AddUpdate: Descriptor has an unknown type");
        return;
    }

    switch (static_cast<EmulatedType>(descriptor.type)) {
    case EmulatedType::kBuffer: {
        auto buffer = FindBufferForAddress(dev, descriptor.handle);
        if (!buffer) {
            Logger::log("error", "AddUpdate: untracked buffer address: %p",
                        descriptor.handle);
            return;
        }

        struct buffer *buf = find_buffer(*buffer);
        if (!buf) {
            Logger::log("error", "AddUpdate: buffer not found: %p", *buffer);
            return;
        }

        VkDeviceSize relativeOffset =
            static_cast<VkDeviceSize>(descriptor.handle - buf->deviceAddress);

        descriptorInfoStorage.push_back(VkDescriptorBufferInfo{
            .buffer = *buffer,
            .offset = relativeOffset,
            .range = descriptor.range == 0 ? VK_WHOLE_SIZE : descriptor.range,
        });

        updateDescriptorSet.pBufferInfo =
            &std::get<VkDescriptorBufferInfo>(descriptorInfoStorage.back());
        updates.push_back(updateDescriptorSet);
        break;
    }
    case EmulatedType::kImage: {
        descriptorInfoStorage.push_back(VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = reinterpret_cast<VkImageView>(descriptor.handle),
            .imageLayout =
                static_cast<VkImageLayout>(descriptor.imageLayoutOrFormat),
        });
        updateDescriptorSet.pImageInfo =
            &std::get<VkDescriptorImageInfo>(descriptorInfoStorage.back());
        updates.push_back(updateDescriptorSet);
        break;
    }
    case EmulatedType::kSampler: {
        descriptorInfoStorage.push_back(VkDescriptorImageInfo{
            .sampler = reinterpret_cast<VkSampler>(descriptor.handle),
            .imageView = VK_NULL_HANDLE,
        });
        updateDescriptorSet.pImageInfo =
            &std::get<VkDescriptorImageInfo>(descriptorInfoStorage.back());
        updates.push_back(updateDescriptorSet);
        break;
    }
    case EmulatedType::kCombinedImageSampler: {
        descriptorInfoStorage.push_back(VkDescriptorImageInfo{
            .sampler = reinterpret_cast<VkSampler>(descriptor.handle2),
            .imageView = reinterpret_cast<VkImageView>(descriptor.handle),
            .imageLayout =
                static_cast<VkImageLayout>(descriptor.imageLayoutOrFormat),
        });
        updateDescriptorSet.pImageInfo =
            &std::get<VkDescriptorImageInfo>(descriptorInfoStorage.back());
        updates.push_back(updateDescriptorSet);
        break;
    }
    case EmulatedType::kTexelBuffer: {
        auto buffer = FindBufferForAddress(dev, descriptor.handle);
        if (!buffer) {
            Logger::log("error",
                        "AddUpdate: untracked texel buffer address: %p",
                        descriptor.handle);
            return;
        }

        struct buffer *buf = find_buffer(*buffer);
        if (!buf) {
            Logger::log("error", "AddUpdate: texel buffer not found: %p",
                        *buffer);
            return;
        }

        auto format = static_cast<VkFormat>(descriptor.imageLayoutOrFormat);
        auto relativeOffset =
            static_cast<VkDeviceSize>(descriptor.handle - buf->deviceAddress);
        auto range = descriptor.range == 0 ? VK_WHOLE_SIZE : descriptor.range;

        VkBufferView bufferView = get_staging_buffer_view(
            dev, *buffer, format, relativeOffset, range);
        if (bufferView == VK_NULL_HANDLE) {
            Logger::log(
                "error",
                "AddUpdate: failed to create texel staging buffer view");
            return;
        }

        descriptorInfoStorage.push_back(bufferView);
        updateDescriptorSet.pTexelBufferView =
            &std::get<VkBufferView>(descriptorInfoStorage.back());
        updates.push_back(updateDescriptorSet);
        break;
    }
    case EmulatedType::kNone:
        break;
    }
}

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint) {
    DescriptorBufferBindingsState state;
    {
        scoped_lock lock(global_lock);
        state = cb->descriptorBufferState;
    }

    VkPipelineLayout activeLayout = VK_NULL_HANDLE;
    VkPipeline boundPipeline = (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
                                   ? state.currentGraphicsPipeline
                                   : state.currentComputePipeline;

    std::shared_lock<std::shared_mutex> deviceLock(dev->db.mutex);

    if (boundPipeline != VK_NULL_HANDLE) {
        auto pipIt = dev->db.pipelines.find(boundPipeline);
        if (pipIt != dev->db.pipelines.end()) {
            if (!pipIt->second.usesDescriptorBuffers) {
                return;
            }
            activeLayout = pipIt->second.layout;
        }
    }

    if (activeLayout == VK_NULL_HANDLE) {
        Logger::log("error",
                    "ResolveAndBindDescriptorSets: no bound pipeline found, "
                    "falling back to currentPipelineLayout: %p",
                    state.currentPipelineLayout);
        activeLayout = state.currentPipelineLayout;
    }

    if (activeLayout == VK_NULL_HANDLE) {
        Logger::log("error",
                    "ResolveAndBindDescriptorSets: no bound pipeline found");
        return;
    }

    auto plIt = dev->db.pipelineLayouts.find(activeLayout);
    if (plIt == dev->db.pipelineLayouts.end()) {
        Logger::log(
            "error",
            "ResolveAndBindDescriptorSets: VkPipelineLayout %p not found",
            activeLayout);
        return;
    }
    const TrackedPipelineLayout &pipelineLayoutInfo = plIt->second;

    for (uint32_t setIndex = 0; setIndex < pipelineLayoutInfo.setLayouts.size();
         ++setIndex) {
        if (setIndex >= kMaxBoundSets || !state.setIsBound[setIndex]) {
            Logger::log("error",
                        "ResolveAndBindDescriptorSets: set %u is not bound",
                        setIndex);
            continue;
        }

        const VkDescriptorSetLayout layoutHandle =
            pipelineLayoutInfo.setLayouts[setIndex];
        auto layoutIt = dev->db.descriptorSetLayouts.find(layoutHandle);
        if (layoutIt == dev->db.descriptorSetLayouts.end()) {
            Logger::log("error",
                        "ResolveAndBindDescriptorSets: layout %p not found",
                        layoutHandle);
            continue;
        }
        const auto &layoutInfo = layoutIt->second;
        if (layoutInfo.totalSize == 0) {
            Logger::log(
                "error",
                "ResolveAndBindDescriptorSets: layout %p has totalSize == 0",
                layoutHandle);
            continue;
        }

        const uint32_t bufferIndex = state.bufferIndexForSet[setIndex];
        if (bufferIndex >= kMaxBoundSets) {
            Logger::log(
                "error",
                "ResolveAndBindDescriptorSets: bufferIndex %u is out of bounds",
                bufferIndex);
            continue;
        }
        const VkBuffer sourceBuffer =
            state.boundBuffersByBindingIndex[bufferIndex];
        if (sourceBuffer == VK_NULL_HANDLE) {
            Logger::log("error",
                        "ResolveAndBindDescriptorSets: sourceBuffer %p is null",
                        sourceBuffer);
            continue;
        }

        void *hostBase = get_host_pointer(dev, sourceBuffer);
        if (hostBase == nullptr) {
            Logger::log("error", "Descriptor buffer is not host-mapped");
            continue;
        }

        struct buffer *buf = find_buffer(sourceBuffer);
        if (buf) {
            VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = buf->memory,
                .offset = buf->offset,
                .size = buf->size,
            };
            dev->table.InvalidateMappedMemoryRanges(dev->handle, 1, &range);
        }

        const char *data =
            static_cast<const char *>(hostBase) + state.offsetForSet[setIndex];
        const auto payloadHash =
            hash_bytes(data, static_cast<size_t>(layoutInfo.totalSize));
        DescriptorSetCache::Key cacheKey{layoutHandle, payloadHash};
        auto &cache = GetTLDescriptorSetCache(dev);

        VkDescriptorSet resolvedSet = VK_NULL_HANDLE;
        bool cacheHit = false;
        if (auto cached = cache.Find(cacheKey)) {
            resolvedSet = *cached;
            cacheHit = true;
        }

        if (cacheHit) {
            dev->table.CmdBindDescriptorSets(cb->handle, bindPoint,
                                             activeLayout, setIndex, 1,
                                             &resolvedSet, 0, nullptr);
            return;
        }

        VkDescriptorPool assignedPool = VK_NULL_HANDLE;
        VkResult allocResult = dev->descriptorSetAllocator->allocate(
            layoutHandle, &assignedPool, &resolvedSet);

        if (allocResult != VK_SUCCESS) {
            Logger::log("error", "Descriptor pool exhausted");
            continue;
        }

        std::vector<VkWriteDescriptorSet> updates;
        std::deque<VkDescriptorInfoStorage> descriptorInfoStorage;

        VkDeviceSize cursor = 0;
        for (const auto &binding : layoutInfo.bindings) {
            for (int element = 0; element < binding.descriptorCount;
                 ++element) {
                EmulatedDescriptor descriptor;
                if (cursor + sizeof(EmulatedDescriptor) <=
                    layoutInfo.totalSize) {
                    std::memcpy(&descriptor, data + cursor,
                                sizeof(EmulatedDescriptor));
                } else {
                    std::memset(&descriptor, 0, sizeof(EmulatedDescriptor));
                }
                cursor += kDescriptorSize;

                if (descriptor.magic != kLayerMagic) {
                    Logger::log("error",
                                "Descriptor magic mismatch at offset %lu",
                                cursor - sizeof(EmulatedDescriptor));
                    continue;
                }

                AddUpdate(dev, resolvedSet, binding.binding, element,
                          binding.descriptorType, descriptor, updates,
                          descriptorInfoStorage);
            }
        }

        if (!updates.empty()) {
            dev->table.UpdateDescriptorSets(
                dev->handle, static_cast<uint32_t>(updates.size()),
                updates.data(), 0, nullptr);
        }

        cache.Insert(cacheKey, resolvedSet);
        dev->table.CmdBindDescriptorSets(cb->handle, bindPoint, activeLayout,
                                         setIndex, 1, &resolvedSet, 0, nullptr);
    }
}
