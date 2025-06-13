#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <math.h>

#define number_of_messages 500
#define number_of_warmups 20

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

__global__ void simple_shift(float *destination, float *source, size_t N)
{
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
  {
    source[idx] += 1.0f;
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  CUDA_CHECK(cudaSetDevice(rank));

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
  CUDA_CHECK(cudaMalloc(&device_buff1, sizeof(float) * max_num_elements));
  CUDA_CHECK(cudaMalloc(&device_buff2, sizeof(float) * max_num_elements));

  for (size_t i = 0; i < max_num_elements; ++i)
    host_buff[i] = (float)rand() / (float)RAND_MAX;

  CUDA_CHECK(cudaMemcpy(device_buff1, host_buff, sizeof(float) * max_num_elements, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_buff2, host_buff, sizeof(float) * max_num_elements, cudaMemcpyHostToDevice));

  int next = (rank + 1) % nranks;
  int prev = (rank + nranks - 1) % nranks;
  MPI_Request requests[2];

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

      // Warmup
      for (int w = 0; w < number_of_warmups; ++w)
      {
#ifndef NO_COMP
        simple_shift<<<numblks, thpblk, 0, stream>>>(device_buff1, device_buff2, num_elements);
        CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        MPI_Isend(device_buff2, num_elements, MPI_FLOAT, next, 99, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(device_buff1, num_elements, MPI_FLOAT, prev, 99, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        std::swap(device_buff1, device_buff2);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
      double start = MPI_Wtime();

      for (size_t i = 0; i < number_of_messages; i++)
      {
#ifndef NO_COMP
        simple_shift<<<numblks, thpblk, 0, stream>>>(device_buff1, device_buff2, num_elements);
        CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        MPI_Isend(device_buff2, num_elements, MPI_FLOAT, next, 99, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(device_buff1, num_elements, MPI_FLOAT, prev, 99, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        std::swap(device_buff1, device_buff2);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      double local_dur = MPI_Wtime() - start;
      double max_dur = 0.0;

      MPI_Reduce(&local_dur, &max_dur, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank == 0)
      {
        double time_per_msg = (max_dur) / (number_of_messages);
        double bandwidth = (2.0 * messagesizebytes / time_per_msg) / 1e9; // GB/s // need 2.0 since we send and recive data in thmeasured time
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
  CUDA_CHECK(cudaFree(device_buff1));
  CUDA_CHECK(cudaFree(device_buff2));
  CUDA_CHECK(cudaStreamDestroy(stream));
  MPI_Finalize();
  return 0;
}
