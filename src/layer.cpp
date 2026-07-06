#include "layer.hpp"

#include "logger.hpp"
#include "vulkan/vk_layer.h"

#include <cmath>
#include <memory>
#include <unistd.h>
#include <utility>
#include <vulkan/vulkan.h>

std::unordered_map<void *, VkLayerInstanceDispatchTable> instanceDispatch;
std::unordered_map<void *, VkInstance> instanceMap;
std::unordered_map<void *, VkPhysicalDeviceFeatures> featuresMap;
std::unordered_map<void *, VkPhysicalDeviceProperties2> propertiesMap;
std::unordered_map<void *, VkPhysicalDeviceDriverProperties>
    driverPropertiesMap;
std::unordered_map<void *, std::shared_ptr<struct device>> deviceMap;

std::mutex global_lock;

#define GETPROCADDR(func)                                                      \
    if (!strcmp(pName, "vk" #func))                                            \
        return (PFN_vkVoidFunction) & DescriptorBufferLayer_##func;

struct device *get_device(VkDevice device) {
    auto it = deviceMap.find(GetKey(device));

    if (it == deviceMap.end())
        return nullptr;

    return it->second.get();
}

SyncPool::~SyncPool() {
    auto dev = get_device(device);
    if (!dev)
        return;
    for (auto f : freeFences)
        dev->table.DestroyFence(device, f, nullptr);
    for (auto s : freeSemaphores)
        dev->table.DestroySemaphore(device, s, nullptr);
}

std::pair<VkSemaphore, VkFence> SyncPool::Acquire() {
    auto dev = get_device(device);
    if (!dev)
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};

    VkFence fence;
    if (!freeFences.empty()) {
        fence = freeFences.back();
        freeFences.pop_back();
        dev->table.ResetFences(device, 1, &fence);
    } else {
        VkFenceCreateInfo info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        dev->table.CreateFence(device, &info, nullptr, &fence);
    }

    VkSemaphore sem;
    if (!freeSemaphores.empty()) {
        sem = freeSemaphores.back();
        freeSemaphores.pop_back();
    } else {
        VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        dev->table.CreateSemaphore(device, &info, nullptr, &sem);
    }
    return {sem, fence};
}

void DescriptorSetAllocator::cleanup() {
    if (activePool != VK_NULL_HANDLE) {
        device->table.DestroyDescriptorPool(device->handle, activePool,
                                            nullptr);
        activePool = VK_NULL_HANDLE;
    }
    for (auto pool : exhaustedPools) {
        device->table.DestroyDescriptorPool(device->handle, pool, nullptr);
    }
    exhaustedPools.clear();
}

VkResult DescriptorSetAllocator::allocate(VkDescriptorSetLayout layout,
                                          VkDescriptorPool *pool,
                                          VkDescriptorSet *descriptors) {
    // Both the Free/AllocateDescriptorSets and access to active/exhausted pools
    // must be synchronized
    scoped_lock l(lock);
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = activePool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };

    VkResult result = device->table.AllocateDescriptorSets(
        device->handle, &alloc_info, descriptors);
    VkDescriptorPool initial_pool = alloc_info.descriptorPool;

    // Check through all the previously exhausted pools to see if one of them
    // has capacity
    while (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
           result == VK_ERROR_FRAGMENTED_POOL) {
        exhaustedPools.push_back(activePool);
        activePool = exhaustedPools.front();
        exhaustedPools.erase(exhaustedPools.begin());
        if (activePool == initial_pool) {
            // we've looped back to the initial pool, time to allocate a new one
            break;
        }
        alloc_info.descriptorPool = activePool;
        result = device->table.AllocateDescriptorSets(device->handle,
                                                      &alloc_info, descriptors);
    }

    // None of the pools have capacity, allocate a new one
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
        result == VK_ERROR_FRAGMENTED_POOL) {
        result = createNewPool(&alloc_info.descriptorPool);
        if (result != VK_SUCCESS) {
            return result;
        }
        exhaustedPools.push_back(activePool);
        activePool = alloc_info.descriptorPool;
        result = device->table.AllocateDescriptorSets(device->handle,
                                                      &alloc_info, descriptors);
    }
    ++allocated_count;
    *pool = alloc_info.descriptorPool;
    return result;
}

void DescriptorSetAllocator::free(VkDescriptorPool pool,
                                  VkDescriptorSet descriptors) {
    scoped_lock l(lock);
    --allocated_count;
    device->table.FreeDescriptorSets(device->handle, pool, 1, &descriptors);
}

VkResult
DescriptorSetAllocator::createNewPool(VkDescriptorPool *descriptor_pool) {
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = poolSizes.maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.sizes.size()),
        .pPoolSizes = poolSizes.sizes.data(),
    };

    auto result = device->table.CreateDescriptorPool(device->handle, &pool_info,
                                                     nullptr, descriptor_pool);
    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to allocate new descriptor pool: %d",
                    result);
    }
    return result;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_CreateInstance(
    const VkInstanceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *layerCreateInfo =
        (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    VkResult result;

    while (layerCreateInfo &&
           (layerCreateInfo->sType !=
                VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO |
            layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
        layerCreateInfo = (VkLayerInstanceCreateInfo *)layerCreateInfo->pNext;
    }

    if (!layerCreateInfo)
        return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gip =
        layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

    PFN_vkCreateInstance createInstance =
        (PFN_vkCreateInstance)gip(VK_NULL_HANDLE, "vkCreateInstance");
    result = createInstance(pCreateInfo, pAllocator, pInstance);

    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to create instance, res %d", result);
        return result;
    }

    VkLayerInstanceDispatchTable table;
    table.GetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)gip(*pInstance, "vkGetInstanceProcAddr");
    table.DestroyInstance =
        (PFN_vkDestroyInstance)gip(*pInstance, "vkDestroyInstance");
    table.EnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)gip(
        *pInstance, "vkEnumeratePhysicalDevices");
    table.GetPhysicalDeviceMemoryProperties =
        (PFN_vkGetPhysicalDeviceMemoryProperties)gip(
            *pInstance, "vkGetPhysicalDeviceMemoryProperties");
    table.GetPhysicalDeviceFormatProperties =
        (PFN_vkGetPhysicalDeviceFormatProperties)gip(
            *pInstance, "vkGetPhysicalDeviceFormatProperties");
    table.GetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)gip(
        *pInstance, "vkGetPhysicalDeviceProperties");
    table.GetPhysicalDeviceProperties2 =
        (PFN_vkGetPhysicalDeviceProperties2)gip(
            *pInstance, "vkGetPhysicalDeviceProperties2");
    table.GetPhysicalDeviceImageFormatProperties =
        (PFN_vkGetPhysicalDeviceImageFormatProperties)gip(
            *pInstance, "vkGetPhysicalDeviceImageFormatProperties");
    table.GetPhysicalDeviceImageFormatProperties2 =
        (PFN_vkGetPhysicalDeviceImageFormatProperties2)gip(
            *pInstance, "vkGetPhysicalDeviceImageFormatProperties2");
    table.GetPhysicalDeviceFeatures = (PFN_vkGetPhysicalDeviceFeatures)gip(
        *pInstance, "vkGetPhysicalDeviceFeatures");
    table.GetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gip(
        *pInstance, "vkGetPhysicalDeviceFeatures2");
    table.GetPhysicalDeviceQueueFamilyProperties =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gip(
            *pInstance, "vkGetPhysicalDeviceQueueFamilyProperties");

    {
        scoped_lock l(global_lock);
        instanceDispatch[GetKey(*pInstance)] = table;
        instanceMap[GetKey(*pInstance)] = *pInstance;
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_DestroyInstance(
    VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    scoped_lock l(global_lock);
    if (!instance)
        return;

    VkLayerInstanceDispatchTable table = instanceDispatch[GetKey(instance)];
    table.DestroyInstance(instance, pAllocator);
    instanceMap.erase(GetKey(instance));
    instanceDispatch.erase(GetKey(instance));
}

VK_LAYER_EXPORT VkResult VKAPI_CALL
DescriptorBufferLayer_EnumeratePhysicalDevices(
    VkInstance instance, uint32_t *pPhysicalDeviceCount,
    VkPhysicalDevice *pPhysicalDevices) {
    scoped_lock l(global_lock);

    VkResult result;

    result = instanceDispatch[GetKey(instance)].EnumeratePhysicalDevices(
        instance, pPhysicalDeviceCount, pPhysicalDevices);

    if (result != VK_SUCCESS || *pPhysicalDeviceCount < 1 ||
        pPhysicalDevices == nullptr)
        return result;

    for (uint32_t index = 0; index < *pPhysicalDeviceCount; index++) {
        VkPhysicalDeviceFeatures features{};
        instanceDispatch[GetKey(instance)].GetPhysicalDeviceFeatures(
            pPhysicalDevices[index], &features);

        VkPhysicalDeviceDriverProperties driverProperties{};
        VkPhysicalDeviceProperties2 props2{};
        driverProperties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &driverProperties;
        instanceDispatch[GetKey(instance)].GetPhysicalDeviceProperties2(
            pPhysicalDevices[index], &props2);

        featuresMap[GetKey(pPhysicalDevices[index])] = features;
        propertiesMap[GetKey(pPhysicalDevices[index])] = props2;
        driverPropertiesMap[GetKey(pPhysicalDevices[index])] = driverProperties;
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_GetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    scoped_lock l(global_lock);

    instanceDispatch[GetKey(physicalDevice)].GetPhysicalDeviceFeatures(
        physicalDevice, pFeatures);
    // pFeatures->textureCompressionBC = true;
}

VK_LAYER_EXPORT void VKAPI_CALL
DescriptorBufferLayer_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
    scoped_lock l(global_lock);

    instanceDispatch[GetKey(physicalDevice)].GetPhysicalDeviceFeatures2(
        physicalDevice, pFeatures);
    // pFeatures->features.textureCompressionBC = true;
}

void FinalizerThread(struct device *dev) {
    while (!dev->stop_thread) {
        {
            std::vector<std::unique_ptr<StagingResources>> queue;
            {
                std::unique_lock<std::mutex> lock(global_lock);
                dev->hasCleanupWork.wait(lock, [dev] {
                    return dev->stop_thread ||
                           !dev->stagingResourcesQueue.empty();
                });

                if (dev->stop_thread) {
                    return;
                }

                if (!dev->stagingResourcesQueue.empty()) {
                    std::swap(queue, dev->stagingResourcesQueue);
                }
            }

            while (!queue.empty()) {
                if (dev->stop_thread)
                    return;
                std::unique_ptr<StagingResources> stagingResources =
                    std::move(queue.back());
                queue.pop_back();
                if (stagingResources) {
                    stagingResources->WaitForCompletion();
                    stagingResources->Cleanup();
                }
            }
        }
    }
}

struct BufferFeatures {
    bool shaderBufferInt64Atomics;
    bool shaderSharedInt64Atomics;
    bool storageBuffer8BitAccess;
    bool storageBuffer16BitAccess;
    bool shaderInt8;
    bool shaderFloat16;
};

BufferFeatures CheckForBufferFeatureSupport(VkInstance instance,
                                            VkPhysicalDevice physicalDevice) {
    auto props = propertiesMap[GetKey(physicalDevice)];
    if (props.properties.apiVersion < VK_API_VERSION_1_1) {
        // 1.0 does not support 8bit
        return {
            .shaderBufferInt64Atomics = false,
            .shaderSharedInt64Atomics = false,
            .storageBuffer8BitAccess = false,
            .storageBuffer16BitAccess = false,
            .shaderInt8 = false,
            .shaderFloat16 = false,
        };
    }

    VkPhysicalDeviceShaderAtomicInt64Features int64Atomics = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    };

    VkPhysicalDevice8BitStorageFeaturesKHR storage8Bit = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
        .pNext = &int64Atomics};

    VkPhysicalDevice16BitStorageFeaturesKHR storage16Bit = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
        .pNext = &storage8Bit};

    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR arithmeticFloat16Int8 = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
        .pNext = &storage16Bit,
    };

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &arithmeticFloat16Int8,
    };

    instanceDispatch[GetKey(instance)].GetPhysicalDeviceFeatures2(
        physicalDevice, &features2);
    return {
        .shaderBufferInt64Atomics =
            (int64Atomics.shaderBufferInt64Atomics == VK_TRUE),
        .shaderSharedInt64Atomics =
            (int64Atomics.shaderSharedInt64Atomics == VK_TRUE),
        .storageBuffer8BitAccess =
            (storage8Bit.storageBuffer8BitAccess == VK_TRUE),
        .storageBuffer16BitAccess =
            (storage16Bit.storageBuffer16BitAccess == VK_TRUE),
        .shaderInt8 = (arithmeticFloat16Int8.shaderInt8 == VK_TRUE),
        .shaderFloat16 = (arithmeticFloat16Int8.shaderFloat16 == VK_TRUE),
    };
}

VkPhysicalDeviceFaultFeaturesEXT
CheckForFaultSupport(VkInstance instance, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceFaultFeaturesEXT faultFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
    };
    VkPhysicalDeviceFeatures2 features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &faultFeatures,
    };

    instanceDispatch[GetKey(instance)].GetPhysicalDeviceFeatures2(
        physicalDevice, &features);

    return faultFeatures;
}

VK_LAYER_EXPORT VkResult VKAPI_CALL DescriptorBufferLayer_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    VkResult result;
    VkLayerDeviceCreateInfo *layerCreateInfo =
        (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    VkDeviceCreateInfo createInfo = *pCreateInfo;

    while (layerCreateInfo &&
           (layerCreateInfo->sType !=
                VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
            layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
        layerCreateInfo = (VkLayerDeviceCreateInfo *)layerCreateInfo->pNext;
    }

    if (layerCreateInfo == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gipa =
        layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa =
        layerCreateInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    bool has_more_layers = layerCreateInfo->u.pLayerInfo->pNext != nullptr;
    layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

    VkInstance instance = instanceMap[GetKey(physicalDevice)];
    if (instance == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDeviceMemoryProperties memoryProps{};
    uint32_t idx;

    instanceDispatch[GetKey(instance)].GetPhysicalDeviceMemoryProperties(
        physicalDevice, &memoryProps);
    for (idx = 0; idx < memoryProps.memoryTypeCount; idx++) {
        if (memoryProps.memoryTypes[idx].propertyFlags &
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            break;
    }

    auto profile_transfers = getenv("COMPAT_DB_PROFILE_TRANSFERS")
                                 ? atoi(getenv("COMPAT_DB_PROFILE_TRANSFERS"))
                                 : 0;
    auto sample_gpu_counters =
        getenv("COMPAT_DB_SAMPLE_GPU_COUNTERS")
            ? atoi(getenv("COMPAT_DB_SAMPLE_GPU_COUNTERS"))
            : 0;

    auto memoryIndex = idx < memoryProps.memoryTypeCount ? idx : UINT32_MAX;
    auto queriedFaultFeatures = CheckForFaultSupport(instance, physicalDevice);
    bool hasFaultSupport = queriedFaultFeatures.deviceFault;

    VkBaseOutStructure *ext = (VkBaseOutStructure *)createInfo.pNext;
    VkPhysicalDeviceFeatures2 *appFeatures2 = nullptr;
    VkPhysicalDeviceFaultFeaturesEXT *appFaultFeatures = nullptr;

    while (ext) {
        if (ext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            appFeatures2 = (VkPhysicalDeviceFeatures2 *)ext;
        } else if (ext->sType ==
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT) {
            appFaultFeatures = (VkPhysicalDeviceFaultFeaturesEXT *)ext;
        }
        ext = ext->pNext;
    }

    VkPhysicalDeviceProperties2 physProps =
        propertiesMap[GetKey(physicalDevice)];
    uint32_t apiVersion = physProps.properties.apiVersion;

    std::vector<const char *> enabledExtensions;
    if (createInfo.ppEnabledExtensionNames &&
        createInfo.enabledExtensionCount > 0) {
        enabledExtensions.assign(createInfo.ppEnabledExtensionNames,
                                 createInfo.ppEnabledExtensionNames +
                                     createInfo.enabledExtensionCount);
    }

    auto hasExtension = [&](const char *name) {
        for (const auto &extName : enabledExtensions) {
            if (strcmp(extName, name) == 0)
                return true;
        }
        return false;
    };

    if (hasFaultSupport && !hasExtension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
        Logger::log("info",
                    "Adding extension " VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        enabledExtensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    }

    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledExtensions.size());

    VkPhysicalDeviceFaultFeaturesEXT layerFaultFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
        .deviceFault = VK_TRUE,
        .deviceFaultVendorBinary = queriedFaultFeatures.deviceFaultVendorBinary,
    };

    if (hasFaultSupport) {
        if (appFaultFeatures) {
            appFaultFeatures->deviceFault = VK_TRUE;
            appFaultFeatures->deviceFaultVendorBinary =
                queriedFaultFeatures.deviceFaultVendorBinary;
        } else {
            Logger::log("info", "Enabling VK_EXT_device_fault features");
            layerFaultFeatures.pNext = (void *)createInfo.pNext;
            createInfo.pNext = &layerFaultFeatures;
        }
    }

    VkPhysicalDeviceFeatures enabledFeatures;
    if (createInfo.pEnabledFeatures) {
        enabledFeatures = *createInfo.pEnabledFeatures;
        // enabledFeatures.textureCompressionBC &=
        //     featuresMap[GetKey(physicalDevice)].textureCompressionBC;

        createInfo.pEnabledFeatures = &enabledFeatures;
    } else if (appFeatures2) {
        // appFeatures2->features.textureCompressionBC &=
        //     featuresMap[GetKey(physicalDevice)].textureCompressionBC;
    }

    PFN_vkCreateDevice createDevice =
        (PFN_vkCreateDevice)gipa(instance, "vkCreateDevice");
    result = createDevice(physicalDevice, &createInfo, pAllocator, pDevice);

    if (result != VK_SUCCESS) {
        Logger::log("error", "Failed to create device, res %d", result);
        return result;
    }

    VkLayerDispatchTable table;
    table.GetDeviceProcAddr =
        (PFN_vkGetDeviceProcAddr)gdpa(*pDevice, "vkGetDeviceProcAddr");
    table.DestroyDevice =
        (PFN_vkDestroyDevice)gdpa(*pDevice, "vkDestroyDevice");
    table.AllocateMemory =
        (PFN_vkAllocateMemory)gdpa(*pDevice, "vkAllocateMemory");
    table.FreeMemory = (PFN_vkFreeMemory)gdpa(*pDevice, "vkFreeMemory");
    table.CreateImage = (PFN_vkCreateImage)gdpa(*pDevice, "vkCreateImage");
    table.CreateImageView =
        (PFN_vkCreateImageView)gdpa(*pDevice, "vkCreateImageView");
    table.DestroyImage = (PFN_vkDestroyImage)gdpa(*pDevice, "vkDestroyImage");
    table.DestroyImageView =
        (PFN_vkDestroyImageView)gdpa(*pDevice, "vkDestroyImageView");
    table.CreateBuffer = (PFN_vkCreateBuffer)gdpa(*pDevice, "vkCreateBuffer");
    table.BindBufferMemory =
        (PFN_vkBindBufferMemory)gdpa(*pDevice, "vkBindBufferMemory");
    table.BindBufferMemory2 =
        (PFN_vkBindBufferMemory2)gdpa(*pDevice, "vkBindBufferMemory2");
    table.DestroyBuffer =
        (PFN_vkDestroyBuffer)gdpa(*pDevice, "vkDestroyBuffer");
    table.AllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)gdpa(
        *pDevice, "vkAllocateCommandBuffers");
    table.CreateCommandPool =
        (PFN_vkCreateCommandPool)gdpa(*pDevice, "vkCreateCommandPool");
    table.DestroyCommandPool =
        (PFN_vkDestroyCommandPool)gdpa(*pDevice, "vkDestroyCommandPool");
    table.GetDeviceQueue =
        (PFN_vkGetDeviceQueue)gdpa(*pDevice, "vkGetDeviceQueue");
    table.CreateFence = (PFN_vkCreateFence)gdpa(*pDevice, "vkCreateFence");
    table.ResetFences = (PFN_vkResetFences)gdpa(*pDevice, "vkResetFences");
    table.DestroyFence = (PFN_vkDestroyFence)gdpa(*pDevice, "vkDestroyFence");
    table.WaitForFences =
        (PFN_vkWaitForFences)gdpa(*pDevice, "vkWaitForFences");
    table.CreateSemaphore =
        (PFN_vkCreateSemaphore)gdpa(*pDevice, "vkCreateSemaphore");
    table.DestroySemaphore =
        (PFN_vkDestroySemaphore)gdpa(*pDevice, "vkDestroySemaphore");
    table.DeviceWaitIdle =
        (PFN_vkDeviceWaitIdle)gdpa(*pDevice, "vkDeviceWaitIdle");
    table.BeginCommandBuffer =
        (PFN_vkBeginCommandBuffer)gdpa(*pDevice, "vkBeginCommandBuffer");
    table.ResetCommandBuffer =
        (PFN_vkResetCommandBuffer)gdpa(*pDevice, "vkResetCommandBuffer");
    table.EndCommandBuffer =
        (PFN_vkEndCommandBuffer)gdpa(*pDevice, "vkEndCommandBuffer");
    table.QueueSubmit = (PFN_vkQueueSubmit)gdpa(*pDevice, "vkQueueSubmit");
    table.QueueSubmit2 = (PFN_vkQueueSubmit2)gdpa(*pDevice, "vkQueueSubmit2");
    table.QueueWaitIdle =
        (PFN_vkQueueWaitIdle)gdpa(*pDevice, "vkQueueWaitIdle");
    table.FreeCommandBuffers =
        (PFN_vkFreeCommandBuffers)gdpa(*pDevice, "vkFreeCommandBuffers");
    table.CreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)gdpa(
        *pDevice, "vkCreateDescriptorSetLayout");
    table.CreateShaderModule =
        (PFN_vkCreateShaderModule)gdpa(*pDevice, "vkCreateShaderModule");
    table.CreatePipelineLayout =
        (PFN_vkCreatePipelineLayout)gdpa(*pDevice, "vkCreatePipelineLayout");
    table.CreateComputePipelines = (PFN_vkCreateComputePipelines)gdpa(
        *pDevice, "vkCreateComputePipelines");
    table.CreateDescriptorPool =
        (PFN_vkCreateDescriptorPool)gdpa(*pDevice, "vkCreateDescriptorPool");
    table.AllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)gdpa(
        *pDevice, "vkAllocateDescriptorSets");
    table.UpdateDescriptorSets =
        (PFN_vkUpdateDescriptorSets)gdpa(*pDevice, "vkUpdateDescriptorSets");
    table.CmdBindPipeline =
        (PFN_vkCmdBindPipeline)gdpa(*pDevice, "vkCmdBindPipeline");
    table.CmdPushConstants =
        (PFN_vkCmdPushConstants)gdpa(*pDevice, "vkCmdPushConstants");
    table.CmdPushConstants2 =
        (PFN_vkCmdPushConstants2)gdpa(*pDevice, "vkCmdPushConstants2");
    table.CmdBindDescriptorSets =
        (PFN_vkCmdBindDescriptorSets)gdpa(*pDevice, "vkCmdBindDescriptorSets");
    table.CmdBindDescriptorSets2 = (PFN_vkCmdBindDescriptorSets2)gdpa(
        *pDevice, "vkCmdBindDescriptorSets2");
    table.CmdDispatch = (PFN_vkCmdDispatch)gdpa(*pDevice, "vkCmdDispatch");
    table.CmdCopyBufferToImage =
        (PFN_vkCmdCopyBufferToImage)gdpa(*pDevice, "vkCmdCopyBufferToImage");
    table.CmdCopyBufferToImage2 =
        (PFN_vkCmdCopyBufferToImage2)gdpa(*pDevice, "vkCmdCopyBufferToImage2");
    table.CmdCopyBuffer =
        (PFN_vkCmdCopyBuffer)gdpa(*pDevice, "vkCmdCopyBuffer");
    table.CmdCopyImage2 =
        (PFN_vkCmdCopyImage2)gdpa(*pDevice, "vkCmdCopyImage2");
    table.CmdCopyImage = (PFN_vkCmdCopyImage)gdpa(*pDevice, "vkCmdCopyImage");
    table.CmdCopyImageToBuffer =
        (PFN_vkCmdCopyImageToBuffer)gdpa(*pDevice, "vkCmdCopyImageToBuffer");
    table.CmdPipelineBarrier =
        (PFN_vkCmdPipelineBarrier)gdpa(*pDevice, "vkCmdPipelineBarrier");
    table.DestroyDescriptorPool =
        (PFN_vkDestroyDescriptorPool)gdpa(*pDevice, "vkDestroyDescriptorPool");
    table.DestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)gdpa(
        *pDevice, "vkDestroyDescriptorSetLayout");
    table.FreeDescriptorSets =
        (PFN_vkFreeDescriptorSets)gdpa(*pDevice, "vkFreeDescriptorSets");
    table.DestroyPipelineLayout =
        (PFN_vkDestroyPipelineLayout)gdpa(*pDevice, "vkDestroyPipelineLayout");
    table.DestroyPipeline =
        (PFN_vkDestroyPipeline)gdpa(*pDevice, "vkDestroyPipeline");
    table.DestroyShaderModule =
        (PFN_vkDestroyShaderModule)gdpa(*pDevice, "vkDestroyShaderModule");
    table.MapMemory = (PFN_vkMapMemory)gdpa(*pDevice, "vkMapMemory");
    table.UnmapMemory = (PFN_vkUnmapMemory)gdpa(*pDevice, "vkUnmapMemory");
    table.InvalidateMappedMemoryRanges =
        (PFN_vkInvalidateMappedMemoryRanges)gdpa(
            *pDevice, "vkInvalidateMappedMemoryRanges");
    table.GetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)gdpa(
        *pDevice, "vkGetBufferMemoryRequirements");
    table.CreateQueryPool =
        (PFN_vkCreateQueryPool)gdpa(*pDevice, "vkCreateQueryPool");
    table.CmdResetQueryPool =
        (PFN_vkCmdResetQueryPool)gdpa(*pDevice, "vkCmdResetQueryPool");
    table.CmdWriteTimestamp =
        (PFN_vkCmdWriteTimestamp)gdpa(*pDevice, "vkCmdWriteTimestamp");
    table.GetQueryPoolResults =
        (PFN_vkGetQueryPoolResults)gdpa(*pDevice, "vkGetQueryPoolResults");
    table.DestroyQueryPool =
        (PFN_vkDestroyQueryPool)gdpa(*pDevice, "vkDestroyQueryPool");
    table.DeviceSetApiDumpState = (void (*)(VkDevice, bool))gdpa(
        *pDevice, "vkDeviceSetApiDumpState"); // exposed by custom ApiDump layer
    table.GetDeviceFaultInfoEXT =
        (PFN_vkGetDeviceFaultInfoEXT)gdpa(*pDevice, "vkGetDeviceFaultInfoEXT");

    if (table.DeviceSetApiDumpState) {
        Logger::log("info", "[DEBUG] vkDeviceSetApiDumpState "
                            "(VK_LAYER_LUNARG_api_dump) is enabled");
    }

    uint32_t queueCount;
    VkQueue queue;

    std::vector<VkQueueFamilyProperties> queueProps;
    instanceDispatch[GetKey(instance)].GetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queueCount, nullptr);
    queueProps.resize(queueCount);
    instanceDispatch[GetKey(instance)].GetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queueCount, queueProps.data());

    uint32_t i = 0;
    for (const auto &family : queueProps) {
        if (family.queueFlags & VK_QUEUE_COMPUTE_BIT)
            break;
        i++;
    }

    table.GetDeviceQueue(*pDevice, i, 0, &queue);

    auto device = std::make_shared<struct device>();
    device->handle = *pDevice;
    device->physical = physicalDevice;
    device->props2 = propertiesMap[GetKey(physicalDevice)];
    device->driverProps = driverPropertiesMap[GetKey(physicalDevice)];
    device->features = featuresMap[GetKey(physicalDevice)];
    device->table = table;
    device->memoryIndex = memoryIndex;
    device->queue = queue;
    device->queueFamilyIndex = i;
    device->alloc = pAllocator;
    device->profile_transfers = profile_transfers;
    device->sample_gpu_counters = sample_gpu_counters;
    device->has_more_layers = has_more_layers;

    device->syncPool = std::make_unique<SyncPool>(device->handle);
    uint32_t buffer_multiplier = 1u;
    const DescriptorSetAllocator::PoolSizes default_pool_sizes{
        .sizes = {{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                   .descriptorCount = 256u},
                  {
                      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      .descriptorCount = 256u * buffer_multiplier,
                  }},
        .maxSets = 256u,
    };
    device->descriptorSetAllocator = std::make_unique<DescriptorSetAllocator>(
        device.get(), default_pool_sizes);

    device->stop_thread = false;
    device->finalizer_thread = std::thread(FinalizerThread, device.get());

    {
        scoped_lock l(global_lock);
        deviceMap[GetKey(*pDevice)] = device;
    }

    return VK_SUCCESS;
}

VK_LAYER_EXPORT void VKAPI_CALL DescriptorBufferLayer_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks *pAllocator) {
    struct device *dev;
    {
        scoped_lock l(global_lock);
        dev = get_device(device);
        if (!dev)
            return;
    }

    dev->stop_thread = true;
    dev->hasCleanupWork.notify_all();
    dev->table.DeviceWaitIdle(device);
    if (dev->finalizer_thread.joinable()) {
        dev->finalizer_thread.join();
    }

    scoped_lock l(global_lock);
    for (auto &stagingResources : dev->stagingResourcesQueue) {
        stagingResources->Cleanup();
    }
    dev->stagingResourcesQueue.clear();
    dev->syncPool.reset();
    dev->descriptorSetAllocator->cleanup();
    dev->descriptorSetAllocator.reset();
    if (device != VK_NULL_HANDLE)
        dev->table.DestroyDevice(device, pAllocator);

    deviceMap.erase(GetKey(device));
}

VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
DescriptorBufferLayer_GetDeviceProcAddr(VkDevice device, const char *pName) {
    GETPROCADDR(DestroyDevice);
    GETPROCADDR(CreateBuffer);
    GETPROCADDR(BindBufferMemory);
    GETPROCADDR(BindBufferMemory2);
    GETPROCADDR(DestroyBuffer);
    GETPROCADDR(AllocateCommandBuffers);
    GETPROCADDR(FreeCommandBuffers);
    GETPROCADDR(BeginCommandBuffer);
    GETPROCADDR(ResetCommandBuffer);
    GETPROCADDR(CmdBindPipeline);
    GETPROCADDR(CmdBindDescriptorSets);
    GETPROCADDR(CmdBindDescriptorSets2);
    GETPROCADDR(CmdPushConstants);
    GETPROCADDR(CmdPushConstants2);
    GETPROCADDR(GetDeviceQueue);
    GETPROCADDR(QueueSubmit);
    GETPROCADDR(QueueSubmit2);

    {
        scoped_lock l(global_lock);
        struct device *dev = get_device(device);
        if (!dev)
            return NULL;

        return dev->table.GetDeviceProcAddr(device, pName);
    }
}

VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
DescriptorBufferLayer_GetInstanceProcAddr(VkInstance instance,
                                          const char *pName) {
    GETPROCADDR(CreateInstance);
    GETPROCADDR(EnumeratePhysicalDevices)
    GETPROCADDR(GetPhysicalDeviceFeatures);
    GETPROCADDR(GetPhysicalDeviceFeatures2);
    GETPROCADDR(DestroyInstance);
    GETPROCADDR(CreateDevice);

    {
        scoped_lock l(global_lock);
        VkLayerInstanceDispatchTable table = instanceDispatch[GetKey(instance)];
        return table.GetInstanceProcAddr(instance, pName);
    }
}
