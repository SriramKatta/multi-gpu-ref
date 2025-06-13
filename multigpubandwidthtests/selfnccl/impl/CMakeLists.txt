# Set source file
set(SRC_FILE main.cu)

# Executable without NO_COMP
set(EXE_NAME ncclsimplestream)
add_executable(${EXE_NAME} ${SRC_FILE})
target_include_directories(${EXE_NAME} PRIVATE NCCL_INCLUDE_DIR)
target_link_libraries(${EXE_NAME} PRIVATE
    nccl
    cudart
    nvidia-ml
    MPI::MPI_CXX
)

# Executable with -DNO_COMP flag
set(EXE_NAME_NO_COMP ncclsimplestream_nocomp)
add_executable(${EXE_NAME_NO_COMP} ${SRC_FILE})
target_compile_definitions(${EXE_NAME_NO_COMP} PRIVATE NO_COMP)
target_include_directories(${EXE_NAME_NO_COMP} PRIVATE NCCL_INCLUDE_DIR)
target_link_libraries(${EXE_NAME_NO_COMP} PRIVATE
    nccl
    cudart
    nvidia-ml
    MPI::MPI_CXX
)
