
#include "descriptor_buffer.hpp"
#include "layer.hpp"

std::optional<VkBuffer> FindBufferForAddress(struct device *dev,
                                             VkDeviceAddress address) {
    // TODO: implement me
    return std::nullopt;
}

void ResolveAndBindDescriptorSets(struct device *dev, struct command_buffer *cb,
                                  VkPipelineBindPoint bindPoint) {
    // TODO: implement me
}
