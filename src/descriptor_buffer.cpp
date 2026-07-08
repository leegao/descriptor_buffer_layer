
#include "descriptor_buffer.hpp"
#include "layer.hpp"

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

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint) {
    // TODO: implement me
}
