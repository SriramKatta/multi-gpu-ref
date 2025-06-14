#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <math.h>
#include <nvtx3/nvtx3.hpp>

#include "utils.h"

using Time = std::chrono::high_resolution_clock;
using mintime = std::nano;

constexpr int number_of_warmups = 10;
constexpr int maxIt = 1000;

using real = double;
#define MPI_REAL_TYPE MPI_DOUBLE

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int N, const int my_ny);
void launch_initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                  const real pi, const int offset, const int N, const int my_ny);
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iy_start, const int iy_end, const int N);
void launch_jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                          const int iy_start, const int iy_end, const int N, cudaStream_t stream);

int main(int argc, char *argv[])
{

  CUDA_CALL(cudaSetDevice(0));

  // just to simplyfy the program
  int N = 1024;
  if (argc > 1)
  {
    N = atoi(argv[1]);
    if (N % 1024 != 0)
    {
      printf("size should be a multiple of 1024\n");
      exit(EXIT_SUCCESS);
    }
  }

  // setting up data for each GPU
  real *a;
  real *a_new;
  CUDA_CALL(cudaMalloc(&a, N * (N + 2) * sizeof(real)));
  CUDA_CALL(cudaMalloc(&a_new, N * (N + 2) * sizeof(real)));
  CUDA_CALL(cudaMemset(a, 0, N * (N + 2) * sizeof(real)));
  CUDA_CALL(cudaMemset(a_new, 0, N * (N + 2) * sizeof(real)));

  int bot = 0;
  int top = N - 1;
  int dombot = bot + 1;
  int domtop = top - 1;

  for (int i = 0; i < number_of_warmups; ++i)
  {
    launch_jacobi_kernel(a, a_new, bot + 1, top, N, {0});
    std::swap(a, a_new);
  }
  CUDA_CALL(cudaDeviceSynchronize());

  launch_initialize_boundaries(a, a_new, M_PI, 1, N, N + 2);
  CUDA_CALL(cudaDeviceSynchronize());

  cudaStream_t inner_stream;
  cudaStream_t edge_stream;
  int lowprio = 0, highprio = 0;
  CUDA_CALL(cudaDeviceGetStreamPriorityRange(&lowprio, &highprio));
  CUDA_CALL(cudaStreamCreateWithPriority(&inner_stream, cudaStreamDefault, lowprio));
  CUDA_CALL(cudaStreamCreateWithPriority(&edge_stream, cudaStreamDefault, highprio));
  cudaEvent_t edge_done;
  CUDA_CALL(cudaEventCreateWithFlags(&edge_done, cudaEventDisableTiming));

  CUDA_CALL(cudaDeviceSynchronize());

  auto start = Time::now();

  for (size_t it = 0; it < maxIt; ++it)
  {
    nvtx3::scoped_range loop{"Jacobi_Step"};

    nvtxRangePushA("inner");
    launch_jacobi_kernel(a_new, a, dombot + 1, domtop, N, inner_stream);
    nvtxRangePop();

    nvtxRangePushA("edges_BC");
    launch_jacobi_kernel(a_new, a, dombot, dombot + 1, N, edge_stream);
    launch_jacobi_kernel(a_new, a, domtop, domtop + 1, N, edge_stream);
    CUDA_CALL(cudaMemcpyAsync(a_new + (top * N), a_new + ((bot + 1) * N), N * sizeof(real), cudaMemcpyDeviceToDevice, edge_stream));
    CUDA_CALL(cudaMemcpyAsync(a_new + (bot * N), a_new + ((top - 1) * N), N * sizeof(real), cudaMemcpyDeviceToDevice, edge_stream));
    nvtxRangePop();
    CUDA_CALL(cudaEventRecord(edge_done, edge_stream));
    CUDA_CALL(cudaStreamWaitEvent(inner_stream, edge_done, 0));

    std::swap(a, a_new);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  auto end = Time::now();
  double maxdur = std::chrono::duration<double, mintime>(end - start).count() / mintime::den / maxIt;

  printf("NP %3d | LUPs %12d | perf %7.3f MLUPS/s\n", 1, (N * N), static_cast<double>(N * N) / maxdur / 1e6);

  // freeing everything
  CUDA_CALL(cudaFree(a));
  CUDA_CALL(cudaFree(a_new));
  CUDA_CALL(cudaEventDestroy(edge_done));
  CUDA_CALL(cudaStreamDestroy(inner_stream));
  return 0;
}

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int N, const int my_ny)
{
  int iy_start = blockIdx.x * blockDim.x + threadIdx.x;
  int iy_stride = blockDim.x * gridDim.x;
  for (int iy = iy_start; iy < my_ny; iy += iy_stride)
  {
    const real y0 = sin(2.0 * pi * (offset + iy) / (N - 1));
    a[iy * N + 0] = y0;
    a[iy * N + (N - 1)] = y0;
    a_new[iy * N + 0] = y0;
    a_new[iy * N + (N - 1)] = y0;
  }
}

void launch_initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                  const real pi, const int offset, const int N, const int my_ny)
{
  initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, N, my_ny);
  CUDA_CALL(cudaGetLastError());
}

void launch_jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                          const int iy_start, const int iy_end,
                          const int N, cudaStream_t stream)
{
  constexpr int dim_block_x = 32;
  constexpr int dim_block_y = 32;
  dim3 thread_dim(dim_block_x, dim_block_x);
  dim3 block_dim((N + dim_block_x - 1) / dim_block_x,
                 ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y);
  jacobi_kernel<dim_block_x, dim_block_y><<<block_dim, thread_dim, 0, stream>>>(
      a_new, a, iy_start, iy_end, N);
  CUDA_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iy_start, const int iy_end, const int N)
{
  int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (iy < iy_end && ix < (N - 1))
  {
    const real new_val = 0.25 * (a[iy * N + ix + 1] + a[iy * N + ix - 1] +
                                 a[(iy + 1) * N + ix] + a[(iy - 1) * N + ix]);
    a_new[iy * N + ix] = new_val;
  }
}
