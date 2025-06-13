

set(SRC_FILE main.cu)

set(EXE_NAME nvshmemsimpestream)
add_executable(${EXE_NAME} ${SRC_FILE})
target_include_directories(${EXE_NAME} PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(${EXE_NAME} PRIVATE
    nvshmem
    cuda
    nvidia-ml
    MPI::MPI_CXX
)


set(EXE_NAME_NO_COMP nvshmemsimpestream_nocomp)
add_executable(${EXE_NAME_NO_COMP} ${SRC_FILE})
target_compile_definitions(${EXE_NAME_NO_COMP} PRIVATE NO_COMP)
target_include_directories(${EXE_NAME_NO_COMP} PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(${EXE_NAME_NO_COMP} PRIVATE
    nvshmem
    cuda
    nvidia-ml
    MPI::MPI_CXX
)
