
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Display window
find_package(OpenCV REQUIRED COMPONENTS highgui)
# VR tracking devices
find_package(OpenVR REQUIRED)
# Depth camera
find_package(RealSense2 REQUIRED)
# Headless rendering of VR shaperone bounds into a depth buffer
find_package(Vulkan REQUIRED)

# CUDA Dependencies for the CXX code.
find_package(CUDAToolkit REQUIRED)

add_executable(vr-salient
    src/main.cu
    src/vulkanheadless.cpp
)

option(SLIDERS "Enable slider controls to change model settings on-the-fly." OFF)

set_target_properties(vr-salient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(vr-salient PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda >
  $<$<BOOL:${SLIDERS}>: -DSLIDERS >
)

# Copy static content: SPIR-V shader code and depth camera settings
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/assets/camera.settings.json" REALSENSE_CAMERA_SETTINGS)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/assets/triangle.vert.spv" VERTEX_SHADER_SOURCE_BYTES HEX)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/assets/triangle.frag.spv" FRAGMENT_SHADER_SOURCE_BYTES HEX)
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


target_include_directories(vr-salient
  PRIVATE
    include
    "${CMAKE_BINARY_DIR}/include"
)

target_link_libraries(vr-salient
  PRIVATE
    real-salient
    opencv_highgui
    OpenVR::OpenVR
    RealSense2::RealSense2
    Vulkan::Vulkan
    CUDA::cudart_static
)

add_custom_command ( TARGET vr-salient POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:opencv_highgui> $<TARGET_FILE:OpenVR::OpenVR> $<TARGET_FILE:RealSense2::RealSense2>
    $<TARGET_FILE_DIR:vr-salient>
)
