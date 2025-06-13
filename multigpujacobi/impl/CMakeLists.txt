add_subdirectory(singlegpu)
add_subdirectory(caware)
add_subdirectory(caware_overlap)

if(BUILD_NCCL)
    add_subdirectory(nccl_overlap)
    add_subdirectory(nccl_overlap_graph)
endif()


if(BUILD_NVSHMEM)
    add_subdirectory(nvshmem)
endif()
