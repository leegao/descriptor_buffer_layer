#ifndef LAYER_HPP
#define LAYER_HPP

#include "logger.hpp"
#include "staging_resources.hpp"
#include "vk_func.hpp"
#include "vulkan/vk_layer.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#undef VK_LAYER_EXPORT
#if defined(WIN32)
#define VK_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define VK_LAYER_EXPORT extern "C"
#endif

#define VK_DRIVER_ID_QUALCOMM_PROPRIETARY 8
#define VK_DRIVER_ID_ARM_PROPRIETARY 9
#define VK_DRIVER_ID_MESA_TURNIP 18
#define VK_DRIVER_ID_SAMSUNG_PROPRIETARY 21

template <typename T> void *GetKey(T item) { return *(void **)item; }

constexpr size_t kDescriptorAlignment = 64;

extern std::mutex global_lock;
typedef std::lock_guard<std::mutex> scoped_lock;

struct TrackedDescriptorSetLayout;
struct TrackedPipelineLayout;

class SyncPool {
  public:
    explicit SyncPool(VkDevice device) : device(device) {}
    ~SyncPool();

    std::pair<VkSemaphore, VkFence> Acquire();

    void Release(VkSemaphore sem, VkFence fence) {
        freeSemaphores.push_back(sem);
        freeFences.push_back(fence);
    }

  private:
    VkDevice device;
    std::vector<VkFence> freeFences;
    std::vector<VkSemaphore> freeSemaphores;
};

class DescriptorSetAllocator {
  public:
    struct PoolSizes {
        std::vector<VkDescriptorPoolSize> sizes;
        uint32_t maxSets = 100;
    };

    explicit DescriptorSetAllocator(struct device *device,
                                    const PoolSizes &defaultSizes)
        : device(device), poolSizes(defaultSizes) {
        createNewPool(&activePool);
    }
    ~DescriptorSetAllocator() { cleanup(); }

    void cleanup();
    VkResult allocate(VkDescriptorSetLayout layout, VkDescriptorPool *pool,
                      VkDescriptorSet *descriptors);
    void free(VkDescriptorPool pool, VkDescriptorSet descriptors);
    uint64_t allocatedCount() const { return allocated_count; }

  private:
    VkResult createNewPool(VkDescriptorPool *descriptor_pool);

    struct device *device = nullptr;
    VkDescriptorPool activePool = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> exhaustedPools;
    PoolSizes poolSizes;
    std::mutex lock;
    std::atomic_uint64_t allocated_count = 0;
};

struct DescriptorBufferEmulationState {
    std::shared_mutex mutex;
    std::unordered_map<VkDescriptorSetLayout, TrackedDescriptorSetLayout>
        descriptorSetLayouts;
    std::unordered_map<VkPipelineLayout, TrackedPipelineLayout> pipelineLayouts;
    std::unordered_map<VkPipeline, VkPipelineLayout> pipelines;
    std::unordered_map<VkDeviceMemory, void *> mappedMemory;
    std::map<VkDeviceAddress, VkBuffer> addressRangeStarts;
};

struct device {
    VkDevice handle;
    VkPhysicalDevice physical;
    VkPhysicalDeviceProperties2 props2;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceDriverProperties driverProps;
    VkLayerDispatchTable table;
    VkQueue queue;
    uint32_t queueFamilyIndex;
    uint32_t memoryIndex;
    int profile_transfers = 0;
    int sample_gpu_counters = 0;
    const VkAllocationCallbacks *alloc;
    std::unique_ptr<SyncPool> syncPool;
    std::unique_ptr<DescriptorSetAllocator> descriptorSetAllocator;
    std::vector<std::unique_ptr<StagingResources>> stagingResourcesQueue;
    std::condition_variable hasCleanupWork;
    std::thread finalizer_thread;
    std::atomic_bool stop_thread{false};
    std::string dump_buffers_path;
    bool has_more_layers = false;
    DescriptorBufferEmulationState db;
};

struct device *get_device(VkDevice);

#endif // LAYER_HPP
