
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/../vr-salient/cmake")

# VR tracking devices
find_package(OpenVR REQUIRED)
# Depth camera
find_package(RealSense2 REQUIRED)
# Headless rendering of VR shaperone bounds into a depth buffer
find_package(Vulkan REQUIRED)

# CUDA Dependencies for the CXX code.
find_package(CUDAToolkit REQUIRED)

add_library(saber-salient SHARED
    src/saber-salient.cu
    ../vr-salient/src/vulkanheadless.cpp
)

find_path(unity_PluginAPI_INCLUDE_DIR
  NAMES IUnityGraphics.h
  HINTS
    "C:/Program Files/Unity/Editor/Data/PluginAPI"
  DOC "A path to UnityEngine headers for native plugins.")



set_target_properties(saber-salient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# STRING(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
# STRING(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
target_compile_options(saber-salient PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda >
  $<$<COMPILE_LANGUAGE:CXX>: /EHa /CLR >
)

# Copy static content: SPIR-V shader code and depth camera settings
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../vr-salient/assets/camera.settings.json" REALSENSE_CAMERA_SETTINGS)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../vr-salient/assets/triangle.vert.spv" VERTEX_SHADER_SOURCE_BYTES HEX)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../vr-salient/assets/triangle.frag.spv" FRAGMENT_SHADER_SOURCE_BYTES HEX)
string(REGEX REPLACE "\"" "\\\\\"" REALSENSE_CAMERA_SETTINGS "${REALSENSE_CAMERA_SETTINGS}")
string(REGEX REPLACE "\n\\s*" " " REALSENSE_CAMERA_SETTINGS "${REALSENSE_CAMERA_SETTINGS}")
string(REGEX REPLACE "(..)" "'\\\\x\\1', " VERTEX_SHADER_SOURCE "${VERTEX_SHADER_SOURCE_BYTES}")
string(REGEX REPLACE "(..)" "'\\\\x\\1', " FRAGMENT_SHADER_SOURCE "${FRAGMENT_SHADER_SOURCE_BYTES}")
file(WRITE "${CMAKE_BINARY_DIR}/include/assets.hpp" "#pragma once
namespace assets {
    const char REALSENSE_CAMERA_SETTINGS[] = \"${REALSENSE_CAMERA_SETTINGS}\";
    const char VERTEX_SHADER_SOURCE[] = {${VERTEX_SHADER_SOURCE}};
    const char FRAGMENT_SHADER_SOURCE[] = {${FRAGMENT_SHADER_SOURCE}};
}")


target_include_directories(saber-salient
  PUBLIC
    include
  PRIVATE
    ../vr-salient/include
    "${CMAKE_BINARY_DIR}/include"
    ${unity_PluginAPI_INCLUDE_DIR}
)

target_link_libraries(saber-salient
  PRIVATE
    real-salient
    OpenVR::OpenVR
    RealSense2::RealSense2
    Vulkan::Vulkan
    CUDA::cudart_static
)

add_custom_command(TARGET saber-salient POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:OpenVR::OpenVR> $<TARGET_FILE:RealSense2::RealSense2>
    $<TARGET_FILE_DIR:vr-salient>
)
