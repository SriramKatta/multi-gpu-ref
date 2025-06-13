# Set source file
set(SRC_FILE main.cu)

# Executable without NO_COMP
set(EXE_NAME cawaresimplestream)
add_executable(${EXE_NAME} ${SRC_FILE})
target_link_libraries(${EXE_NAME} PRIVATE
    MPI::MPI_CXX
)

# Executable with -DNO_COMP flag
set(EXE_NAME_NO_COMP cawaresimplestream_nocomp)
add_executable(${EXE_NAME_NO_COMP} ${SRC_FILE})
target_compile_definitions(${EXE_NAME_NO_COMP} PRIVATE NO_COMP)
target_link_libraries(${EXE_NAME_NO_COMP} PRIVATE
    MPI::MPI_CXX
)
