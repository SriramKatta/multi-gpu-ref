#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <math.h>

#define number_of_messages 500
#define number_of_warmups 10

#define CUDA_CHECK(stmt)                                       \
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

#define NCCL_CHECK(stmt)                                       \
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

__global__ void simple_shift(float *destination, float *source, size_t N)
{
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
  {
#ifndef NO_COMP
    source[idx] += 1.0f;
#endif
    nvshmem_float_p(&destination[idx], source[idx], peer);
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm comm = MPI_COMM_WORLD;

  // all nvshmem from here
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int mype_node;
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CUDA_CHECK(cudaSetDevice(mype_node));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Benchmark parameters
  size_t base_val = 2;
  size_t max_pow = 32;
  size_t stride_pow = 1;
  int num_median_runs = 5;

  size_t max_size_bytes = (size_t)pow(base_val, max_pow);
  size_t max_num_elements = max_size_bytes / sizeof(float);

  float *host_buff;
  float *device_buff1, *device_buff2;

  CUDA_CHECK(cudaMallocHost(&host_buff, sizeof(float) * max_num_elements));
  device_buff1 = (float *)nvshmem_malloc(sizeof(float) * max_num_elements);
  device_buff2 = (float *)nvshmem_malloc(sizeof(float) * max_num_elements);

  for (size_t i = 0; i < max_num_elements; ++i)
    host_buff[i] = (float)rand() / (float)RAND_MAX;

  CUDA_CHECK(cudaMemcpy(device_buff1, host_buff, sizeof(float) * max_num_elements, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_buff2, host_buff, sizeof(float) * max_num_elements, cudaMemcpyHostToDevice));

  for (size_t pow = 3; pow <= max_pow; pow += stride_pow)
  {
    size_t messagesizebytes = (size_t)powf(base_val, pow);
    size_t num_elements = messagesizebytes / sizeof(float);
    dim3 thpblk(512);
    dim3 numblks((num_elements + thpblk.x - 1) / thpblk.x);

    // if (rank == 0)
    //   printf("\n[%s] Testing message size: %zu^%zu = %zu bytes\n", __TIME__, base_val, pow, messagesizebytes);
    double avgbw = 0.0;
    for (int median_loop = 1; median_loop <= num_median_runs; ++median_loop)
    {
      MPI_Barrier(MPI_COMM_WORLD);

      for (int w = 0; w < number_of_warmups; ++w)
      {
        simple_shift<<<numblks, thpblk, 0, stream>>>(device_buff1, device_buff2, max_num_elements);
        nvshmemx_barrier_all_on_stream(stream);
        std::swap(device_buff1, device_buff2);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
      double start = MPI_Wtime();

      for (size_t i = 0; i < number_of_messages; i++)
      {
        simple_shift<<<numblks, thpblk, 0, stream>>>(device_buff1, device_buff2, max_num_elements);
        nvshmemx_barrier_all_on_stream(stream);
        std::swap(device_buff1, device_buff2);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      double local_dur = MPI_Wtime() - start;
      double max_dur = 0.0;

      MPI_Reduce(&local_dur, &max_dur, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank == 0)
      {
        double time_per_msg = (max_dur) / (number_of_messages);
        double bandwidth = (2.0 * messagesizebytes / time_per_msg) / 1e9; // GB/s
        avgbw += bandwidth;
        if (median_loop == num_median_runs)
        {
          avgbw /= static_cast<double>(num_median_runs);
          printf(
              "NP %3d | Message_Size %12zu | avgbw %7.3f GB/s\n",
              nranks,
              messagesizebytes,
              avgbw);
        }
      }
    }
  }

  CUDA_CHECK(cudaFreeHost(host_buff));
  nvshmem_free(device_buff1);
  nvshmem_free(device_buff2);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
