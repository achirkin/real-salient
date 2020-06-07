#[[
This file is a modified version of the original file at

url: https://gitlab.kitware.com/vtk/vtk
sha: 8b1e455f0824d046e09f73857e24dbeca8d367a2
license: BSD3
The copyright text follows:

  Program:   Visualization Toolkit
  Module:    Copyright.txt

Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
   of any contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

]]

# Note that OpenVR lacks a useful install tree. This should work if
# `OpenVR_DIR` is set to the source directory of OpenVR.

# TODO: fails for universal builds
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_openvr_bitness 64)
else ()
  set(_openvr_bitness 32)
endif ()

set(_openvr_platform_base)
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(_openvr_platform_base osx)
  # SteamVR only supports 32-bit on OS X
  set(OpenVR_PLATFORM osx32)
else ()
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(_openvr_platform_base linux)
  elseif (WIN32)
    set(_openvr_platform_base win)
  endif ()
  set(OpenVR_PLATFORM ${_openvr_platform_base}${_openvr_bitness})
endif ()

find_path(OpenVR_INCLUDE_DIR
  NAMES
    openvr.h
  HINTS
    ${OpenVR_DIR}
  PATH_SUFFIXES
    headers
    public/headers
    steam
    public/steam
  DOC "OpenVR include directory")
mark_as_advanced(OpenVR_INCLUDE_DIR)

find_file(OpenVR_LIBRARY_SHARED
  NAMES "openvr_api${CMAKE_SHARED_LIBRARY_SUFFIX}"
  HINTS
    ${OpenVR_DIR}
  PATH_SUFFIXES
    "bin/${OpenVR_PLATFORM}"
  DOC "OpenVR API shared library (the one in bin folder, e.g. openvr_api.dll or openvr_api.so)")
mark_as_advanced(OpenVR_LIBRARY_SHARED)

find_library(OpenVR_LIBRARY
  NAMES openvr_api
  HINTS
    ${OpenVR_DIR}
  PATH_SUFFIXES
    "lib/${OpenVR_PLATFORM}"
  DOC "OpenVR API library")
mark_as_advanced(OpenVR_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVR
  REQUIRED_VARS OpenVR_LIBRARY OpenVR_INCLUDE_DIR)

if (OpenVR_FOUND)
  set(OpenVR_INCLUDE_DIRS "${OpenVR_INCLUDE_DIR}")
  set(OpenVR_LIBRARIES "${OpenVR_LIBRARY}")
  if (NOT TARGET OpenVR::OpenVR)
    if (OpenVR_LIBRARY_SHARED)
      add_library(OpenVR::OpenVR SHARED IMPORTED)
      set_target_properties(OpenVR::OpenVR PROPERTIES
        IMPORTED_IMPLIB "${OpenVR_LIBRARY}"
        IMPORTED_LOCATION "${OpenVR_LIBRARY_SHARED}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenVR_INCLUDE_DIR}"
      )
    else ()
      add_library(OpenVR::OpenVR UNKNOWN IMPORTED)
      set_target_properties(OpenVR::OpenVR PROPERTIES
        IMPORTED_LOCATION "${OpenVR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenVR_INCLUDE_DIR}"
      )
    endif ()
  endif ()
endif ()


unset(_openvr_bitness)
unset(_openvr_platform_base)
