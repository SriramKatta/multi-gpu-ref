cmake_minimum_required(VERSION 3.23)
project(nccl_example LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)

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

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(MPI REQUIRED)


add_subdirectory(impl)