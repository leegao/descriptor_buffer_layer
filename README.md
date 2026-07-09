# descriptor_buffer_layer

Emulates `VK_EXT_descriptor_buffer`

Originally meant for vkd3d, but is [not necessary](https://github.com/HansKristian-Work/vkd3d-proton/blob/ce3c862c26ef5b1b74a3a24cf637bddb1adb7a7c/VP_D3D12_VKD3D_PROTON_profile.json#L500) for it to work.

Enable by `ENABLE_COMPAT_DESCRIPTOR_BUFFER_LAYER=1` or `VK_INSTANCE_LAYERS=VK_LAYER_COMPAT_DescriptorBufferLayer`

Test with Vulkan-Samples: https://github.com/leegao/Vulkan-Samples/releases
