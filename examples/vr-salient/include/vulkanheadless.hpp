#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

class VulkanHeadless
{

private:
    cudaMipmappedArray_t cudaMipmap;
    cudaArray_t cudaArray;

    VkSemaphore vulkanRenderingDone;
    cudaExternalMemory_t cudaExtMemory;
    cudaExternalSemaphore_t cudaExtSemaphore;

    uint32_t indexCount;

    uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties);
    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr);

    void initCudaResources(VkDeviceSize size);
    void destroyCudaResources();

    /* Submit command buffer to a queue and wait for fence until queue operations have been finished */
    void submitWork(VkCommandBuffer cmdBuffer, VkQueue queue);

    VkCommandBufferBeginInfo renderCmdBufBeginInfo;
    VkClearValue renderClearValues[2];
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    VkViewport viewport = {};
    VkRect2D scissor = {};
    VkDeviceSize vertexBufferOffsets[1] = {0};
    VkSubmitInfo submitInfo{};
    cudaExternalSemaphoreWaitParams waitParams = {};

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t queueFamilyIndex;
    VkPipelineCache pipelineCache;
    VkQueue queue;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    std::vector<VkShaderModule> shaderModules;
    VkBuffer vertexBuffer, indexBuffer;
    VkDeviceMemory vertexMemory, indexMemory;

    cudaTextureObject_t cudaTexture;

    struct FrameBufferAttachment
    {
        VkImage image;
        VkDeviceMemory memory;
        VkImageView view;
    };

    VkFramebuffer framebuffer;
    FrameBufferAttachment colorAttachment, depthAttachment;
    VkRenderPass renderPass;

    VkDebugReportCallbackEXT debugReportCallback{};

    void prepareRenderStructs();

public:
    bool isValid;

    struct Vertex
    {
        float position[3];
    };

    const int32_t width, height;

    VulkanHeadless(const int32_t width, const int32_t height, std::vector<Vertex> vertices, std::vector<uint32_t> indices, uint8_t requestedUUID[VK_UUID_SIZE] = NULL);
    ~VulkanHeadless();

    void render(float *mvpMatrix, cudaStream_t stream);

    inline cudaTextureObject_t *getCudaTexture()
    {
        return isValid ? &cudaTexture : nullptr;
    }
};