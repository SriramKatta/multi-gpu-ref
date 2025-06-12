set(EXE_NAME jacobi_caware)
set(SRC_FILE jacobi.cu)

add_executable(${EXE_NAME} ${SRC_FILE})
target_link_libraries(${EXE_NAME} PRIVATE
  MPI::MPI_CXX
  nvtx3-c
)

if(NVTX_OFF)
  target_compile_definitions(${EXE_NAME} PUBLIC DISABLE_NVTX)
endif()
