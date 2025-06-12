set(EXE_NAME jacobi_NCCL_overlap)
set(SRC_FILE jacobi.cu)

add_executable(${EXE_NAME} ${SRC_FILE})
target_include_directories(${EXE_NAME} PRIVATE  ${NCCL_INCLUDE_DIR})
target_compile_definitions(${EXE_NAME} PRIVATE USE_NCCL)
target_link_libraries(${EXE_NAME} PRIVATE
    nccl
    cudart
    nvidia-ml
    MPI::MPI_CXX
    nvtx3-c
)


if(NVTX_OFF)
  target_compile_definitions( ${EXE_NAME} PRIVATE DISABLE_NVTX)
endif(NVTX_OFF)