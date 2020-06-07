cmake_minimum_required(VERSION 3.17)
project(real-salient LANGUAGES CXX CUDA VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 11)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(real-salient INTERFACE)
target_include_directories(real-salient INTERFACE include/)
target_compile_options(real-salient INTERFACE
  $<$<CONFIG:DEBUG>: -DCUDAERRORCHECKS > # enable cudaErrorCheck(), which syncs GPU and shows the last error
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math >
)

option(BUILD_EXAMPLES "Build example executables.")

if(BUILD_EXAMPLES)
	add_subdirectory (examples/vr-salient)
endif()