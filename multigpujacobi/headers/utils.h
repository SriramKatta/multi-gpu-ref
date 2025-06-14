#pragma once

constexpr int number_of_warmups = 10;
constexpr int maxIt = 10000;

#ifdef USE_NCCL
#include <nccl.h>
#endif

#define CUDA_CALL(stmt)                                       \
  do                                                           \
  {                                                            \
    cudaError_t result = (stmt);                               \
    if (cudaSuccess != result)                                 \
    {                                                          \
      fprintf(stderr, "[%s:%d] CUDA failed with %s \n",        \
              __FILE__, __LINE__, cudaGetErrorString(result)); \
      exit(-1);                                                \
    }                                                          \
  } while (0)

#define MPI_CALL(call)                                                          \
  {                                                                             \
    int mpi_status = call;                                                      \
    if (MPI_SUCCESS != mpi_status)                                              \
    {                                                                           \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
      int mpi_error_string_length = 0;                                          \
      MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
      if (NULL != mpi_error_string)                                             \
        fprintf(stderr,                                                         \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "          \
                "with %s "                                                      \
                "(%d).\n",                                                      \
                #call, __LINE__, __FILE__, mpi_error_string, mpi_status);       \
      else                                                                      \
        fprintf(stderr,                                                         \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "          \
                "with %d.\n",                                                   \
                #call, __LINE__, __FILE__, mpi_status);                         \
      exit(mpi_status);                                                         \
    }                                                                           \
  }

#ifdef USE_NCCL
#define NCCL_CALL(stmt)                                       \
  do                                                           \
  {                                                            \
    ncclResult_t result = (stmt);                              \
    if (ncclSuccess != result)                                 \
    {                                                          \
      fprintf(stderr, "[%s:%d] NCCL error: %s\n",              \
              __FILE__, __LINE__, ncclGetErrorString(result)); \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (0)
#endif

inline int rowsinrank(int rank, int nranks, int N)
{
    return N / nranks + ((N % nranks > rank) ? 1 : 0);
}

inline int startrow(int rank, int nranks, int N)
{
    int remainder = N % nranks;
    int base_rows = N / nranks;

    // Each rank before `rank` gets base_rows, and an extra one if its index is less than remainder
    return rank * base_rows + ((rank < remainder) ? rank : remainder);
}
