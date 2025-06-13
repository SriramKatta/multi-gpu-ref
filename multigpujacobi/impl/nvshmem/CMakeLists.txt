set(EXE_NAME jacobi_nvshmem)
set(SRC_FILE jacobi.cu)

add_executable(${EXE_NAME} ${SRC_FILE})
target_include_directories(${EXE_NAME} PRIVATE  ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(${EXE_NAME} PRIVATE
    nvshmem
    cuda
    nvidia-ml
    MPI::MPI_CXX
    nvtx3-c
)


if(NVTX_OFF)
  target_compile_definitions( ${EXE_NAME} PRIVATE DISABLE_NVTX)
endif(NVTX_OFF)