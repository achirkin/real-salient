cmake_minimum_required(VERSION 3.1.0)
project(real-salient LANGUAGES CXX CUDA VERSION 0.1.0)


# required for the view window only
find_package(OpenCV REQUIRED)

add_executable(real-salient
  src/main.cu
  include/salient.cuh
  include/gmm.cuh
  include/crf.cuh
  include/util.hpp)
target_include_directories(real-salient PRIVATE include)
target_compile_options(real-salient PRIVATE
  $<$<CONFIG:DEBUG>: -DCUDAERRORCHECKS > # enable cudaErrorCheck(), which syncs GPU and shows the last error
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math --extended-lambda >
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native -Wall -Wpedantic -Wextra -fexceptions >
)
set (CMAKE_CXX_STANDARD 14)
#target_compile_features(real-salient PUBLIC cxx_std_14)
set_target_properties(real-salient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(real-salient realsense2 openvr_api ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

