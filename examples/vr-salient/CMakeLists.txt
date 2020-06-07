
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Display window
find_package(OpenCV REQUIRED COMPONENTS core)
# VR tracking devices
find_package(OpenVR REQUIRED)
# Depth camera
find_package(RealSense2 REQUIRED)


add_executable(vr-salient src/main.cu)


set_target_properties(vr-salient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(vr-salient PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda >
  $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wpedantic -Wextra -fexceptions >
)

target_include_directories(vr-salient PRIVATE include)

target_link_libraries(vr-salient
  PRIVATE
    real-salient
    opencv_core
	OpenVR::OpenVR
	RealSense2::RealSense2
)

add_custom_command ( TARGET vr-salient POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:opencv_core> $<TARGET_FILE:OpenVR::OpenVR> $<TARGET_FILE:RealSense2::RealSense2>
	$<TARGET_FILE_DIR:vr-salient>
)