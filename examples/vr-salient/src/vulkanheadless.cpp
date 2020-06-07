/* NBs:

https://stackoverflow.com/a/55547739
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1g0012448b208196bc60c2307eaeaa3ff7
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleVulkan/vulkanCUDASinewave.cu

*/


/*
* Vulkan Example - Minimal headless rendering example
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "assets.hpp"
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include "vulkanheadless.hpp"

#define VK_CHECK_RESULT(f)																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        std::cout << "Fatal : VkResult is \"" << vks::tools::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

#ifdef CUDAERRORCHECKS
#define cudaErrorCheck() __cudaErrorCheck(__func__, __FILE__, __LINE__)
#else
#define cudaErrorCheck() ((void)0)
#endif

void __cudaErrorCheck(char const* const func, const char* const file, int const line)
{
        auto code = cudaGetLastError();
        if (cudaSuccess == code)
        {
            cudaDeviceSynchronize();
            code = cudaGetLastError();
        }
        if (cudaSuccess != code)
        {
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, code, cudaGetErrorString(code), func);
            exit(code);
        }
}


#define BUFFER_ELEMENTS 32

#define LOG(...) printf(__VA_ARGS__)

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objectType,
    uint64_t object,
    size_t location,
    int32_t messageCode,
    const char* pLayerPrefix,
    const char* pMessage,
    void* pUserData)
{
    LOG("[VALIDATION]: %s - %s\n", pLayerPrefix, pMessage);
    return VK_FALSE;
}

namespace vks
{
    namespace initializers
    {

        inline VkMemoryAllocateInfo memoryAllocateInfo()
        {
            VkMemoryAllocateInfo memAllocInfo{};
            memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            return memAllocInfo;
        }


        inline VkCommandBufferAllocateInfo commandBufferAllocateInfo(
            VkCommandPool commandPool,
            VkCommandBufferLevel level,
            uint32_t bufferCount)
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.commandPool = commandPool;
            commandBufferAllocateInfo.level = level;
            commandBufferAllocateInfo.commandBufferCount = bufferCount;
            return commandBufferAllocateInfo;
        }

        inline VkCommandPoolCreateInfo commandPoolCreateInfo()
        {
            VkCommandPoolCreateInfo cmdPoolCreateInfo{};
            cmdPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            return cmdPoolCreateInfo;
        }

        inline VkCommandBufferBeginInfo commandBufferBeginInfo()
        {
            VkCommandBufferBeginInfo cmdBufferBeginInfo{};
            cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            return cmdBufferBeginInfo;
        }

        inline VkRenderPassBeginInfo renderPassBeginInfo()
        {
            VkRenderPassBeginInfo renderPassBeginInfo{};
            renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            return renderPassBeginInfo;
        }

        inline VkRenderPassCreateInfo renderPassCreateInfo()
        {
            VkRenderPassCreateInfo renderPassCreateInfo{};
            renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            return renderPassCreateInfo;
        }

        /** @brief Initialize an image memory barrier with no image transfer ownership */
        inline VkImageMemoryBarrier imageMemoryBarrier()
        {
            VkImageMemoryBarrier imageMemoryBarrier{};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            return imageMemoryBarrier;
        }

        inline VkImageCreateInfo imageCreateInfo()
        {
            VkImageCreateInfo imageCreateInfo{};
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            return imageCreateInfo;
        }


        inline VkImageViewCreateInfo imageViewCreateInfo()
        {
            VkImageViewCreateInfo imageViewCreateInfo{};
            imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            return imageViewCreateInfo;
        }

        inline VkFramebufferCreateInfo framebufferCreateInfo()
        {
            VkFramebufferCreateInfo framebufferCreateInfo{};
            framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            return framebufferCreateInfo;
        }

        inline VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0)
        {
            VkFenceCreateInfo fenceCreateInfo{};
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = flags;
            return fenceCreateInfo;
        }


        inline VkSubmitInfo submitInfo()
        {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            return submitInfo;
        }

        inline VkViewport viewport(
            float width,
            float height,
            float minDepth,
            float maxDepth)
        {
            VkViewport viewport{};
            viewport.width = width;
            viewport.height = height;
            viewport.minDepth = minDepth;
            viewport.maxDepth = maxDepth;
            return viewport;
        }

        inline VkBufferCreateInfo bufferCreateInfo()
        {
            VkBufferCreateInfo bufCreateInfo{};
            bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            return bufCreateInfo;
        }

        inline VkBufferCreateInfo bufferCreateInfo(
            VkBufferUsageFlags usage,
            VkDeviceSize size)
        {
            VkBufferCreateInfo bufCreateInfo{};
            bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufCreateInfo.usage = usage;
            bufCreateInfo.size = size;
            return bufCreateInfo;
        }

        inline VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
            const VkDescriptorSetLayoutBinding* pBindings,
            uint32_t bindingCount)
        {
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
            descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCreateInfo.pBindings = pBindings;
            descriptorSetLayoutCreateInfo.bindingCount = bindingCount;
            return descriptorSetLayoutCreateInfo;
        }

        inline VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
            const std::vector<VkDescriptorSetLayoutBinding>& bindings)
        {
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
            descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCreateInfo.pBindings = bindings.data();
            descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            return descriptorSetLayoutCreateInfo;
        }

        inline VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            const VkDescriptorSetLayout* pSetLayouts,
            uint32_t setLayoutCount = 1)
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = setLayoutCount;
            pipelineLayoutCreateInfo.pSetLayouts = pSetLayouts;
            return pipelineLayoutCreateInfo;
        }

        inline VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            uint32_t setLayoutCount = 1)
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = setLayoutCount;
            return pipelineLayoutCreateInfo;
        }

        inline VkVertexInputBindingDescription vertexInputBindingDescription(
            uint32_t binding,
            uint32_t stride,
            VkVertexInputRate inputRate)
        {
            VkVertexInputBindingDescription vInputBindDescription{};
            vInputBindDescription.binding = binding;
            vInputBindDescription.stride = stride;
            vInputBindDescription.inputRate = inputRate;
            return vInputBindDescription;
        }

        inline VkVertexInputAttributeDescription vertexInputAttributeDescription(
            uint32_t binding,
            uint32_t location,
            VkFormat format,
            uint32_t offset)
        {
            VkVertexInputAttributeDescription vInputAttribDescription{};
            vInputAttribDescription.location = location;
            vInputAttribDescription.binding = binding;
            vInputAttribDescription.format = format;
            vInputAttribDescription.offset = offset;
            return vInputAttribDescription;
        }

        inline VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo()
        {
            VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo{};
            pipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            return pipelineVertexInputStateCreateInfo;
        }

        inline VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(
            VkPrimitiveTopology topology,
            VkPipelineInputAssemblyStateCreateFlags flags,
            VkBool32 primitiveRestartEnable)
        {
            VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo{};
            pipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            pipelineInputAssemblyStateCreateInfo.topology = topology;
            pipelineInputAssemblyStateCreateInfo.flags = flags;
            pipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = primitiveRestartEnable;
            return pipelineInputAssemblyStateCreateInfo;
        }

        inline VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo(
            VkPolygonMode polygonMode,
            VkCullModeFlags cullMode,
            VkFrontFace frontFace,
            VkPipelineRasterizationStateCreateFlags flags = 0)
        {
            VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo{};
            pipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            pipelineRasterizationStateCreateInfo.polygonMode = polygonMode;
            pipelineRasterizationStateCreateInfo.cullMode = cullMode;
            pipelineRasterizationStateCreateInfo.frontFace = frontFace;
            pipelineRasterizationStateCreateInfo.flags = flags;
            pipelineRasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
            pipelineRasterizationStateCreateInfo.lineWidth = 1.0f;
            return pipelineRasterizationStateCreateInfo;
        }

        inline VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState(
            VkColorComponentFlags colorWriteMask,
            VkBool32 blendEnable)
        {
            VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState{};
            pipelineColorBlendAttachmentState.colorWriteMask = colorWriteMask;
            pipelineColorBlendAttachmentState.blendEnable = blendEnable;
            return pipelineColorBlendAttachmentState;
        }

        inline VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo(
            uint32_t attachmentCount,
            const VkPipelineColorBlendAttachmentState* pAttachments)
        {
            VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo{};
            pipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            pipelineColorBlendStateCreateInfo.attachmentCount = attachmentCount;
            pipelineColorBlendStateCreateInfo.pAttachments = pAttachments;
            return pipelineColorBlendStateCreateInfo;
        }

        inline VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo(
            VkBool32 depthTestEnable,
            VkBool32 depthWriteEnable,
            VkCompareOp depthCompareOp)
        {
            VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo{};
            pipelineDepthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            pipelineDepthStencilStateCreateInfo.depthTestEnable = depthTestEnable;
            pipelineDepthStencilStateCreateInfo.depthWriteEnable = depthWriteEnable;
            pipelineDepthStencilStateCreateInfo.depthCompareOp = depthCompareOp;
            pipelineDepthStencilStateCreateInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
            return pipelineDepthStencilStateCreateInfo;
        }

        inline VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo(
            uint32_t viewportCount,
            uint32_t scissorCount,
            VkPipelineViewportStateCreateFlags flags = 0)
        {
            VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo{};
            pipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            pipelineViewportStateCreateInfo.viewportCount = viewportCount;
            pipelineViewportStateCreateInfo.scissorCount = scissorCount;
            pipelineViewportStateCreateInfo.flags = flags;
            return pipelineViewportStateCreateInfo;
        }

        inline VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo(
            VkSampleCountFlagBits rasterizationSamples,
            VkPipelineMultisampleStateCreateFlags flags = 0)
        {
            VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo{};
            pipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            pipelineMultisampleStateCreateInfo.rasterizationSamples = rasterizationSamples;
            pipelineMultisampleStateCreateInfo.flags = flags;
            return pipelineMultisampleStateCreateInfo;
        }

        inline VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(
            const VkDynamicState* pDynamicStates,
            uint32_t dynamicStateCount,
            VkPipelineDynamicStateCreateFlags flags = 0)
        {
            VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{};
            pipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            pipelineDynamicStateCreateInfo.pDynamicStates = pDynamicStates;
            pipelineDynamicStateCreateInfo.dynamicStateCount = dynamicStateCount;
            pipelineDynamicStateCreateInfo.flags = flags;
            return pipelineDynamicStateCreateInfo;
        }

        inline VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(
            const std::vector<VkDynamicState>& pDynamicStates,
            VkPipelineDynamicStateCreateFlags flags = 0)
        {
            VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{};
            pipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            pipelineDynamicStateCreateInfo.pDynamicStates = pDynamicStates.data();
            pipelineDynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(pDynamicStates.size());
            pipelineDynamicStateCreateInfo.flags = flags;
            return pipelineDynamicStateCreateInfo;
        }


        inline VkGraphicsPipelineCreateInfo pipelineCreateInfo(
            VkPipelineLayout layout,
            VkRenderPass renderPass,
            VkPipelineCreateFlags flags = 0)
        {
            VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
            pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCreateInfo.layout = layout;
            pipelineCreateInfo.renderPass = renderPass;
            pipelineCreateInfo.flags = flags;
            pipelineCreateInfo.basePipelineIndex = -1;
            pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
            return pipelineCreateInfo;
        }

        inline VkGraphicsPipelineCreateInfo pipelineCreateInfo()
        {
            VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
            pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCreateInfo.basePipelineIndex = -1;
            pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
            return pipelineCreateInfo;
        }


        inline VkPushConstantRange pushConstantRange(
            VkShaderStageFlags stageFlags,
            uint32_t size,
            uint32_t offset)
        {
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = stageFlags;
            pushConstantRange.offset = offset;
            pushConstantRange.size = size;
            return pushConstantRange;
        }

    }

    namespace tools
    {
        bool errorModeSilent = false;

        std::string errorString(VkResult errorCode)
        {
            switch (errorCode)
            {
#define STR(r) case VK_ ##r: return #r
                STR(NOT_READY);
                STR(TIMEOUT);
                STR(EVENT_SET);
                STR(EVENT_RESET);
                STR(INCOMPLETE);
                STR(ERROR_OUT_OF_HOST_MEMORY);
                STR(ERROR_OUT_OF_DEVICE_MEMORY);
                STR(ERROR_INITIALIZATION_FAILED);
                STR(ERROR_DEVICE_LOST);
                STR(ERROR_MEMORY_MAP_FAILED);
                STR(ERROR_LAYER_NOT_PRESENT);
                STR(ERROR_EXTENSION_NOT_PRESENT);
                STR(ERROR_FEATURE_NOT_PRESENT);
                STR(ERROR_INCOMPATIBLE_DRIVER);
                STR(ERROR_TOO_MANY_OBJECTS);
                STR(ERROR_FORMAT_NOT_SUPPORTED);
                STR(ERROR_SURFACE_LOST_KHR);
                STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
                STR(SUBOPTIMAL_KHR);
                STR(ERROR_OUT_OF_DATE_KHR);
                STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
                STR(ERROR_VALIDATION_FAILED_EXT);
                STR(ERROR_INVALID_SHADER_NV);
#undef STR
            default:
                return "UNKNOWN_ERROR";
            }
        }

        VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat)
        {
            // Since all depth formats may be optional, we need to find a suitable depth format to use
            // Start with the highest precision packed format
            std::vector<VkFormat> depthFormats = {
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D24_UNORM_S8_UINT,
                VK_FORMAT_D16_UNORM_S8_UINT,
                VK_FORMAT_D16_UNORM
            };

            for (auto& format : depthFormats)
            {
                VkFormatProperties formatProps;
                vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
                // Format must support depth stencil attachment for optimal tiling
                if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                {
                    *depthFormat = format;
                    return true;
                }
            }

            return false;
        }


        void insertImageMemoryBarrier(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask,
            VkImageSubresourceRange subresourceRange)
        {
            VkImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
            imageMemoryBarrier.srcAccessMask = srcAccessMask;
            imageMemoryBarrier.dstAccessMask = dstAccessMask;
            imageMemoryBarrier.oldLayout = oldImageLayout;
            imageMemoryBarrier.newLayout = newImageLayout;
            imageMemoryBarrier.image = image;
            imageMemoryBarrier.subresourceRange = subresourceRange;

            vkCmdPipelineBarrier(
                cmdbuffer,
                srcStageMask,
                dstStageMask,
                0,
                0, nullptr,
                0, nullptr,
                1, &imageMemoryBarrier);
        }


        VkShaderModule loadShader(const char* shaderCode, const size_t shaderCodeLength, VkDevice device)
        {
            VkShaderModule shaderModule;
            VkShaderModuleCreateInfo moduleCreateInfo{};
            moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleCreateInfo.codeSize = shaderCodeLength;
            moduleCreateInfo.pCode = (uint32_t*)shaderCode;

            VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

            return shaderModule;
        }

        /** Get CUDA memory object of the VkImage */
        cudaExternalMemory_t getVkCudaMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size) {
            cudaExternalMemory_t externalMem;
            cudaExternalMemoryHandleDesc memDesc = { };
            memset(&memDesc, 0, sizeof(memDesc));
            memDesc.size = size;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
            VkMemoryGetWin32HandleInfoKHR hInfo = { };
            hInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            hInfo.memory = memory;
            hInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
            PFN_vkGetMemoryWin32HandleKHR func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
#else
            memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
            VkMemoryGetFdInfoKHR hInfo = { };
            hInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            hInfo.memory = memory;
            hInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
            PFN_vkGetMemoryFdKHR func = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
#endif
            if (func == NULL) {
                std::cout << "Failed to locate function vkGetMemoryXxxKHR" << std::endl;
                assert(false);
            }
            hInfo.pNext = NULL;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            VK_CHECK_RESULT(func(device, &hInfo, &(memDesc.handle.win32.handle)));
#else
            VK_CHECK_RESULT(func(device, &hInfo, &(memDesc.handle.fd)));
#endif
            cudaImportExternalMemory(&externalMem, &memDesc);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            CloseHandle(memDesc.handle.win32.handle);
#endif
            cudaErrorCheck();
            return externalMem;
        }
    }
}



uint32_t VulkanHeadless::getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
    for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        typeBits >>= 1;
    }
    return 0;
}

VkResult VulkanHeadless::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer* buffer, VkDeviceMemory* memory, VkDeviceSize size, void* data)
{
    // Create the buffer handle
    VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer));

    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    vkGetBufferMemoryRequirements(device, *buffer, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, memoryPropertyFlags);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, memory));

    if (data != nullptr) {
        void* mapped;
        VK_CHECK_RESULT(vkMapMemory(device, *memory, 0, size, 0, &mapped));
        memcpy(mapped, data, size);
        vkUnmapMemory(device, *memory);
    }

    VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *memory, 0));

    return VK_SUCCESS;
}

void VulkanHeadless::initCudaResources(VkDeviceSize size)
{
    cudaExtMemory = vks::tools::getVkCudaMemory(device, colorAttachment.memory, size);

    cudaExternalMemoryMipmappedArrayDesc vulkan_mipmap_desc;
    vulkan_mipmap_desc.offset = 0;
    vulkan_mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    vulkan_mipmap_desc.extent.depth = 0;
    vulkan_mipmap_desc.extent.width = width;
    vulkan_mipmap_desc.extent.height = height;
    vulkan_mipmap_desc.flags = cudaArrayColorAttachment;
    vulkan_mipmap_desc.numLevels = 1;
    cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmap, cudaExtMemory, &vulkan_mipmap_desc);
    cudaErrorCheck();
    cudaGetMipmappedArrayLevel(&cudaArray, cudaMipmap, 0);
    cudaErrorCheck();

    cudaTextureDesc vulkan_texDesc;
    memset(&vulkan_texDesc, 0, sizeof(vulkan_texDesc));
    vulkan_texDesc.readMode = cudaReadModeElementType;    // Same type as content (i.e. not a normalized float)
    vulkan_texDesc.addressMode[0] = cudaAddressModeClamp; // clamp to coordinates to the image region
    vulkan_texDesc.addressMode[1] = cudaAddressModeClamp; // -/-
    vulkan_texDesc.filterMode = cudaFilterModePoint;     // nearest neighbour interpolation
    vulkan_texDesc.normalizedCoords = false;              // do not normalize to 0..1

    cudaResourceDesc vulkan_resourceDesc;
    memset(&vulkan_resourceDesc, 0, sizeof(vulkan_resourceDesc));
    vulkan_resourceDesc.resType = cudaResourceTypeArray;
    vulkan_resourceDesc.res.array.array = cudaArray;
    cudaCreateTextureObject(&cudaTexture, &vulkan_resourceDesc, &vulkan_texDesc, NULL);
    cudaErrorCheck();
}

void VulkanHeadless::destroyCudaResources()
{
    cudaDestroyTextureObject(cudaTexture);
    cudaErrorCheck();
    cudaFreeMipmappedArray(cudaMipmap);
    cudaErrorCheck();
    cudaDestroyExternalMemory(cudaExtMemory);
    cudaErrorCheck();
}

void VulkanHeadless::submitWork(VkCommandBuffer cmdBuffer, VkQueue queue)
{
    VkSubmitInfo submitInfo = vks::initializers::submitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo();
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
}

void VulkanHeadless::render(float* mvpMatrix)
{
    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    VkClearValue clearValues[2];
    clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = framebuffer;

    vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {};
    viewport.height = (float)height;
    viewport.width = (float)width;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    // Update dynamic scissor state
    VkRect2D scissor = {};
    scissor.extent.width = width;
    scissor.extent.height = height;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Render scene
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);


    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float)*16, mvpMatrix);

    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    submitWork(commandBuffer, queue);

    vkDeviceWaitIdle(device);
}

VulkanHeadless::VulkanHeadless(int32_t width, int32_t height, std::vector<Vertex> vertices, std::vector<uint32_t> indices)
    : width(width), height(height)
{
        LOG("Running headless rendering example\n");

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan headless example";
        appInfo.pEngineName = "VulkanHeadless";
        appInfo.apiVersion = VK_API_VERSION_1_0;

        /*
            Vulkan instance creation (without surface extensions)
        */
        VkInstanceCreateInfo instanceCreateInfo = {};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;

        uint32_t layerCount = 0;
        const char* validationLayers[] = { "VK_LAYER_LUNARG_standard_validation" };
        std::vector<const char*> instanceExtensions = {
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME };
        layerCount = 1;
#ifndef NDEBUG
        // Check if layers are available
        uint32_t instanceLayerCount;
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
        std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

        bool layersAvailable = true;
        for (auto layerName : validationLayers) {
            bool layerAvailable = false;
            for (auto instanceLayer : instanceLayers) {
                if (strcmp(instanceLayer.layerName, layerName) == 0) {
                    layerAvailable = true;
                    break;
                }
            }
            if (!layerAvailable) {
                layersAvailable = false;
                break;
            }
        }

        if (layersAvailable) {
            instanceCreateInfo.ppEnabledLayerNames = validationLayers;
            instanceCreateInfo.enabledLayerCount = layerCount;
            instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
#endif
        instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = &instanceExtensions[0];
        VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

#ifndef NDEBUG
        if (layersAvailable) {
            VkDebugReportCallbackCreateInfoEXT debugReportCreateInfo = {};
            debugReportCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            debugReportCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
            debugReportCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;

            // We have to explicitly load this function.
            PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
            assert(vkCreateDebugReportCallbackEXT);
            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &debugReportCreateInfo, nullptr, &debugReportCallback));
        }
#endif

        /*
            Vulkan device creation
        */
        uint32_t deviceCount = 0;
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
        std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
        physicalDevice = physicalDevices[0];

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        LOG("GPU: %s\n", deviceProperties.deviceName);

        // Request a single graphics queue
        const float defaultQueuePriority(0.0f);
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
            if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                queueFamilyIndex = i;
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = i;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
                break;
            }
        }
        // Create logical device
        std::vector<const char*> deviceExtensions = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
        };

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.ppEnabledExtensionNames = &deviceExtensions[0];
        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

        // Get a graphics queue
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        // Command pool
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));

        /*
            Prepare vertex and index buffers
        */
        {
            const VkDeviceSize vertexBufferSize = vertices.size() * sizeof(Vertex);
            const VkDeviceSize indexBufferSize = indices.size() * sizeof(uint32_t);
            indexCount = (uint32_t)indices.size();

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;

            // Command buffer for copy commands (reused)
            VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
            VkCommandBuffer copyCmd;
            VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));
            VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

            // Copy input data to VRAM using a staging buffer
            {
                // Vertices
                createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &stagingBuffer,
                    &stagingMemory,
                    vertexBufferSize,
                    vertices.data());

                createBuffer(
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    &vertexBuffer,
                    &vertexMemory,
                    vertexBufferSize);

                VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));
                VkBufferCopy copyRegion = {};
                copyRegion.size = vertexBufferSize;
                vkCmdCopyBuffer(copyCmd, stagingBuffer, vertexBuffer, 1, &copyRegion);
                VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

                submitWork(copyCmd, queue);

                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingMemory, nullptr);

                // Indices
                createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &stagingBuffer,
                    &stagingMemory,
                    indexBufferSize,
                    indices.data());

                createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    &indexBuffer,
                    &indexMemory,
                    indexBufferSize);

                VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));
                copyRegion.size = indexBufferSize;
                vkCmdCopyBuffer(copyCmd, stagingBuffer, indexBuffer, 1, &copyRegion);
                VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

                submitWork(copyCmd, queue);

                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingMemory, nullptr);
            }
        }

        /*
            Create framebuffer attachments
        */
        VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        VkFormat depthFormat;
        vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat);
        {
            // Color attachment
            VkImageCreateInfo image = vks::initializers::imageCreateInfo();
            image.imageType = VK_IMAGE_TYPE_2D;
            image.format = colorFormat;
            image.extent.width = width;
            image.extent.height = height;
            image.extent.depth = 1;
            image.mipLevels = 1;
            image.arrayLayers = 1;
            image.samples = VK_SAMPLE_COUNT_1_BIT;
            image.tiling = VK_IMAGE_TILING_OPTIMAL; // important for the mapping!
            image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // TODO: remove transfer?..
            
            VkExternalMemoryImageCreateInfoKHR extImageCreateInfo = {};
            extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            extImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            extImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
            image.pNext = &extImageCreateInfo;

            VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
            VkMemoryRequirements memReqs;

            VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &colorAttachment.image));
            vkGetImageMemoryRequirements(device, colorAttachment.image, &memReqs);
            memAlloc.allocationSize = memReqs.size;
            memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkExportMemoryAllocateInfoKHR exportInfo = {};
            exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
            memAlloc.pNext = &exportInfo;

            VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &colorAttachment.memory));
            VK_CHECK_RESULT(vkBindImageMemory(device, colorAttachment.image, colorAttachment.memory, 0));

            // give access to cuda
            initCudaResources(memAlloc.allocationSize);

            VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
            colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            colorImageView.format = colorFormat;
            colorImageView.subresourceRange = {};
            colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            colorImageView.subresourceRange.baseMipLevel = 0;
            colorImageView.subresourceRange.levelCount = 1;
            colorImageView.subresourceRange.baseArrayLayer = 0;
            colorImageView.subresourceRange.layerCount = 1;
            colorImageView.image = colorAttachment.image;
            VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &colorAttachment.view));

            // Depth stencil attachment
            image.format = depthFormat;
            image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

            VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &depthAttachment.image));
            vkGetImageMemoryRequirements(device, depthAttachment.image, &memReqs);
            memAlloc.allocationSize = memReqs.size;
            memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthAttachment.memory));
            VK_CHECK_RESULT(vkBindImageMemory(device, depthAttachment.image, depthAttachment.memory, 0));

            VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
            depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            depthStencilView.format = depthFormat;
            depthStencilView.flags = 0;
            depthStencilView.subresourceRange = {};
            depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            depthStencilView.subresourceRange.baseMipLevel = 0;
            depthStencilView.subresourceRange.levelCount = 1;
            depthStencilView.subresourceRange.baseArrayLayer = 0;
            depthStencilView.subresourceRange.layerCount = 1;
            depthStencilView.image = depthAttachment.image;
            VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &depthAttachment.view));
        }

        /*
            Create renderpass
        */
        {
            std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
            // Color attachment
            attchmentDescriptions[0].format = colorFormat;
            attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
            attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            // Depth attachment
            attchmentDescriptions[1].format = depthFormat;
            attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
            attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

            VkSubpassDescription subpassDescription = {};
            subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpassDescription.colorAttachmentCount = 1;
            subpassDescription.pColorAttachments = &colorReference;
            subpassDescription.pDepthStencilAttachment = &depthReference;

            // Use subpass dependencies for layout transitions
            std::array<VkSubpassDependency, 2> dependencies;

            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            // Create the actual renderpass
            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
            renderPassInfo.pAttachments = attchmentDescriptions.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpassDescription;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));

            VkImageView attachments[2];
            attachments[0] = colorAttachment.view;
            attachments[1] = depthAttachment.view;

            VkFramebufferCreateInfo framebufferCreateInfo = vks::initializers::framebufferCreateInfo();
            framebufferCreateInfo.renderPass = renderPass;
            framebufferCreateInfo.attachmentCount = 2;
            framebufferCreateInfo.pAttachments = attachments;
            framebufferCreateInfo.width = width;
            framebufferCreateInfo.height = height;
            framebufferCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &framebuffer));
        }

        /*
            Prepare graphics pipeline
        */
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {};
            VkDescriptorSetLayoutCreateInfo descriptorLayout =
                vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
                vks::initializers::pipelineLayoutCreateInfo(nullptr, 0);

            // MVP via push constant block
            VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 16, 0);
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

            VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
            pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));

            // Create pipeline
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
                vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

            VkPipelineRasterizationStateCreateInfo rasterizationState =
                vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE);

            VkPipelineColorBlendAttachmentState blendAttachmentState =
                vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

            VkPipelineColorBlendStateCreateInfo colorBlendState =
                vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

            VkPipelineDepthStencilStateCreateInfo depthStencilState =
                vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

            VkPipelineViewportStateCreateInfo viewportState =
                vks::initializers::pipelineViewportStateCreateInfo(1, 1);

            VkPipelineMultisampleStateCreateInfo multisampleState =
                vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);

            std::vector<VkDynamicState> dynamicStateEnables = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };
            VkPipelineDynamicStateCreateInfo dynamicState =
                vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

            VkGraphicsPipelineCreateInfo pipelineCreateInfo =
                vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

            std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

            pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
            pipelineCreateInfo.pRasterizationState = &rasterizationState;
            pipelineCreateInfo.pColorBlendState = &colorBlendState;
            pipelineCreateInfo.pMultisampleState = &multisampleState;
            pipelineCreateInfo.pViewportState = &viewportState;
            pipelineCreateInfo.pDepthStencilState = &depthStencilState;
            pipelineCreateInfo.pDynamicState = &dynamicState;
            pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineCreateInfo.pStages = shaderStages.data();

            // Vertex bindings an attributes
            // Binding description
            std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
                vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
            };

            // Attribute descriptions
            std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Position
                vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Color
            };

            VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
            vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
            vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
            vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
            vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

            pipelineCreateInfo.pVertexInputState = &vertexInputState;

            shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            shaderStages[0].pName = "main";
            shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            shaderStages[1].pName = "main";
            shaderStages[0].module = vks::tools::loadShader(assets::VERTEX_SHADER_SOURCE, sizeof(assets::VERTEX_SHADER_SOURCE), device);
            shaderStages[1].module = vks::tools::loadShader(assets::FRAGMENT_SHADER_SOURCE, sizeof(assets::FRAGMENT_SHADER_SOURCE), device);
            shaderModules = { shaderStages[0].module, shaderStages[1].module };
            VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
        }

        /*
            Command buffer creation
        */
        VkCommandBufferAllocateInfo cmdBufAllocateInfo =
            vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

        float mvpMatrix[16];
        memset(&mvpMatrix, 0, sizeof(mvpMatrix));
        mvpMatrix[0] = 1;
        mvpMatrix[5] = 1;
        mvpMatrix[10] = 1;
        mvpMatrix[15] = 1;
        render(mvpMatrix);

        vkQueueWaitIdle(queue);
    }

VulkanHeadless::~VulkanHeadless()
    {
        std::cout << "destroying VulkanHeadless" << std::endl;
        destroyCudaResources();

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexMemory, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexMemory, nullptr);
        vkDestroyImageView(device, colorAttachment.view, nullptr);
        vkDestroyImage(device, colorAttachment.image, nullptr);
        vkFreeMemory(device, colorAttachment.memory, nullptr);
        vkDestroyImageView(device, depthAttachment.view, nullptr);
        vkDestroyImage(device, depthAttachment.image, nullptr);
        vkFreeMemory(device, depthAttachment.memory, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        for (auto shadermodule : shaderModules) {
            vkDestroyShaderModule(device, shadermodule, nullptr);
        }
        vkDestroyDevice(device, nullptr);
#ifndef NDEBUG
        if (debugReportCallback) {
            PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
            assert(vkDestroyDebugReportCallback);
            vkDestroyDebugReportCallback(instance, debugReportCallback, nullptr);
        }
#endif
        vkDestroyInstance(instance, nullptr);
    }

