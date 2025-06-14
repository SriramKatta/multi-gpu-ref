#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <math.h>
#include <nvtx3/nvtx3.hpp>

#include "utils.h"

// TODO: commented out since my personal machine doesn't have cuda aware mpi :(
#ifndef SKIP_CUDA_AWARENESS_CHECK
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
#error "The used MPI Implementation does not have CUDA-aware support or CUDA-aware \
support can't be determined. Define SKIP_CUDA_AWARENESS_CHECK to skip this check."
#endif
#endif

constexpr int number_of_warmups = 10;
constexpr int maxIt = 1000;

using real = double;

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int N, const int my_ny);
void launch_initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                  const real pi, const int offset, const int N, const int my_ny);
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iy_start, const int iy_end, const int N, int top_pe, int bot_pe);
void launch_jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                          const int iy_start, const int iy_end, const int N, cudaStream_t stream);

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  int rank = 0;
  int nranks = 1;
  int num_devices = 0;
  // assumption is that we are only on single node but easily extendable using MPI_Comm_split_type(.... MPI_COMM_TYPE_SHARED...
  {
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  }
  MPI_Comm comm = MPI_COMM_WORLD;

  // all nvshmem from here
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int mype_node;
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CUDA_CALL(cudaSetDevice(mype_node));

  // just to simplyfy the program
  int N = 1024;
  if (argc > 1)
  {
    N = atoi(argv[1]);
    if (N % 1024 != 0)
    {
      if (rank == 0)
        printf("size should be a multiple of 1024\n");
      MPI_CALL(MPI_Finalize());
      exit(EXIT_SUCCESS);
    }
  }

  // computing local chunk size
  int chunk_size = rowsinrank(rank, nranks, N);

  // setting up data for each GPU
  real *a = (real *)nvshmem_malloc(sizeof(real) * N * (chunk_size + 2));
  real *a_new = (real *)nvshmem_malloc(sizeof(real) * N * (chunk_size + 2));
  // clang-format off
  CUDA_CALL(cudaMemset(a   , 0, N * (chunk_size + 2) * sizeof(real)));
  CUDA_CALL(cudaMemset(a_new,0, N * (chunk_size + 2) * sizeof(real)));
  // clang-format on

  // Calculate local domain boundaries
  int iy_start_global = startrow(rank, nranks, N);

  int iy_start = 1;
  int iy_end = iy_start + chunk_size;

  launch_initialize_boundaries(a, a_new, M_PI, iy_start_global - 1, N, (chunk_size + 2));
  CUDA_CALL(cudaDeviceSynchronize());

  cudaStream_t compute_stream;
  CUDA_CALL(cudaStreamCreate(&compute_stream));

  const int top_pe = (rank + 1) % nranks;
  const int bot_pe = (rank + nranks - 1) % nranks;

  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  CUDA_CALL(cudaDeviceSynchronize());

  double start = MPI_Wtime();
  for (size_t it = 0; it < maxIt; ++it)
  {
    nvtx3::scoped_range loop{"Jacobi_Step"};

    launch_jacobi_kernel(a_new, a, iy_start, iy_end, N, compute_stream);

    std::swap(a, a_new);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  double dur = (MPI_Wtime() - start) / maxIt;
  double maxdur = 0.0;
  MPI_CALL(MPI_Reduce(&dur, &maxdur, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

  if (rank == 0)
  {
    printf("NP %3d | LUPs %12d | perf %7.3f MLUPS/s\n", nranks, (N * N), static_cast<double>(N * N) / maxdur / 1e6);
  }

  // freeing everything
  nvshmem_free(a);
  nvshmem_free(a_new);
  CUDA_CALL(cudaStreamDestroy(compute_stream));
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int N, const int my_ny)
{
  for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x)
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
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int top_pe = (mype + 1) % npes;
  int bot_pe = (mype + npes - 1) % npes;
  constexpr int dim_block_x = 32;
  constexpr int dim_block_y = 32;
  dim3 thread_dim(dim_block_x, dim_block_x);
  dim3 block_dim((N + dim_block_x - 1) / dim_block_x,
                 ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y);
  jacobi_kernel<dim_block_x, dim_block_y><<<block_dim, thread_dim, 0, stream>>>(
      a_new, a, iy_start, iy_end, N, top_pe, bot_pe);
  CUDA_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iy_start, const int iy_end, const int N, int top_pe, int bot_pe)
{
  int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (iy < iy_end && ix < (N - 1))
  {
    const real new_val = 0.25 * (a[iy * N + ix + 1] + a[iy * N + ix - 1] +
                                 a[(iy + 1) * N + ix] + a[(iy - 1) * N + ix]);
    a_new[iy * N + ix] = new_val;
    // Halo exchange
    if (iy == iy_start )
    {
      nvshmem_double_p(a_new + (iy * (N-1) + ix), new_val, bot_pe);
    }
    if (iy == iy_end - 1 )
    {
      nvshmem_double_p(a_new  + ix, new_val, top_pe);
    }
  }
}
