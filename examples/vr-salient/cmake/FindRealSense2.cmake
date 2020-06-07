
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_realsense2_platform x64)
else ()
  set(_realsense2_platform x86)
endif ()

find_path(RealSense2_INCLUDE_DIR
  NAMES
    librealsense2/rs.h
    librealsense2/rs.hpp
  HINTS
    ${RealSense2_DIR}
  PATH_SUFFIXES
    include
  DOC "Intel RealSense 2 include directory")
mark_as_advanced(RealSense2_INCLUDE_DIR)

find_file(RealSense2_LIBRARY_SHARED
  NAMES "realsense2${CMAKE_SHARED_LIBRARY_SUFFIX}"
  HINTS
    ${RealSense2_DIR}
  PATH_SUFFIXES
    "bin/${_realsense2_platform}"
  DOC "Intel RealSense 2 shared library (the one in bin folder, e.g. realsense2.dll or realsense2.so)")
mark_as_advanced(RealSense2_LIBRARY_SHARED)

find_library(RealSense2_LIBRARY
  NAMES realsense2
  HINTS
    ${RealSense2_DIR}
  PATH_SUFFIXES
    "bin/${_realsense2_platform}"
    "lib/${_realsense2_platform}"
  DOC "Intel RealSense 2 library")
mark_as_advanced(RealSense2_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RealSense2
  REQUIRED_VARS RealSense2_LIBRARY RealSense2_INCLUDE_DIR)

if (RealSense2_FOUND)
  set(RealSense2_INCLUDE_DIRS "${RealSense2_INCLUDE_DIR}")
  set(RealSense2_LIBRARIES "${RealSense2_LIBRARY}")
  if (NOT TARGET RealSense2::RealSense2)
    if (RealSense2_LIBRARY_SHARED)
      add_library(RealSense2::RealSense2 SHARED IMPORTED)
      set_target_properties(RealSense2::RealSense2 PROPERTIES
        IMPORTED_IMPLIB "${RealSense2_LIBRARY}"
        IMPORTED_LOCATION "${RealSense2_LIBRARY_SHARED}"
        INTERFACE_INCLUDE_DIRECTORIES "${RealSense2_INCLUDE_DIR}"
      )
    else ()
      add_library(RealSense2::RealSense2 UNKNOWN IMPORTED)
      set_target_properties(RealSense2::RealSense2 PROPERTIES
        IMPORTED_LOCATION "${RealSense2_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RealSense2_INCLUDE_DIR}"
      )
    endif ()
  endif ()
endif ()


unset(_realsense2_platform)
