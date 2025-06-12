cmake_minimum_required(VERSION 3.23)
project(nccl_example LANGUAGES CXX CUDA)


include(cmake/CPM.cmake)
include(cmake/NVTX.cmake)

option(NVTX_OFF "Disable NVTX profiling" ON)
if(NVTX_OFF)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable_prof)
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable_perf)
endif()


# Set C++ standard and disable in-source builds
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
find_package(MPI REQUIRED)

# Set CUDA architectures (you can customize as needed)
set(CMAKE_CUDA_ARCHITECTURES 80;86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit REQUIRED)
# Find NCCL
if(NCCL_HOME)
  find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${NCCL_HOME}/include
    REQUIRED
  )
  find_library(NCCL_LIB
    NAMES nccl
    PATHS ${NCCL_HOME}/lib
    REQUIRED
  )
  set(BUILD_NCCL ON)
else()
  message(WARNING "Please set -DNCCL_HOME to compile NCCL code")
endif()

if (NVSHMEM_HOME)
  find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    PATHS ${NVSHMEM_HOME}/include
    REQUIRED
  )
  find_library(NVSHMEM_LIB
    NAMES nvshmem nvshmem_host nvshmem_device
    PATHS ${NVSHMEM_HOME}/lib
    REQUIRED
  )
  set(BUILD_NVSHMEM ON)
else()
  message(WARNING "Please set -DNVSHMEM_HOME to compile NVSHMEM code")
endif()

include_directories(headers)

add_subdirectory(impl)
