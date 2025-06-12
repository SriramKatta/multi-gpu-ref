#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <math.h>
#include <nvtx3/nvtx3.hpp>
#include <nccl.h>

#include "utils.h"

// TODO: comment out on my personal machine since it doesn't have cuda aware mpi :(
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
#define NCCL_REAL_TYPE ncclDouble

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int N, const int my_ny);
void launch_initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                  const real pi, const int offset, const int N, const int my_ny);
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iy_start, const int iy_end, const int N);
void launch_jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                          const int iy_start, const int iy_end, const int N, cudaStream_t stream);
void Halo_exchange(real *a, real *a_new, int N, const int top, int iy_end, const int bottom, int iy_start, ncclComm_t, cudaStream_t);

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

  // important part of cuda aware mpi
  {
    CUDA_CALL(cudaGetDeviceCount(&num_devices));
    CUDA_CALL(cudaSetDevice(rank % num_devices));
    CUDA_CALL(cudaFree(0));
  }

  // pass the communication hadling to NCCL
  ncclComm_t ncclcomm;
  ncclUniqueId id;
  if (rank == 0)
    NCCL_CALL(ncclGetUniqueId(&id));
  MPI_CALL(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_CALL(ncclCommInitRank(&ncclcomm, nranks, id, rank));

  // just to besafe
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

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
  real *a;
  real *a_new;
  // clang-format off
  CUDA_CALL(cudaMalloc(&a     , N * (chunk_size + 2) * sizeof(real)));
  CUDA_CALL(cudaMemset(a   , 0, N * (chunk_size + 2) * sizeof(real)));
  CUDA_CALL(cudaMalloc(&a_new , N * (chunk_size + 2) * sizeof(real)));
  CUDA_CALL(cudaMemset(a_new,0, N * (chunk_size + 2) * sizeof(real)));
  // clang-format on

  // Calculate local domain boundaries
  int iy_start_global = startrow(rank, nranks, N);

  int iy_start = 1;
  int iy_end = iy_start + chunk_size;

  launch_initialize_boundaries(a, a_new, M_PI, iy_start_global - 1, N, (chunk_size + 2));
  CUDA_CALL(cudaDeviceSynchronize());

  int highpriority = 0, lowpriority = 0;
  cudaStream_t inner_stream;
  cudaStream_t edge_stream;
  CUDA_CALL(cudaDeviceGetStreamPriorityRange(&lowpriority, &highpriority));
  CUDA_CALL(cudaStreamCreateWithPriority(&inner_stream, cudaStreamDefault, lowpriority));
  CUDA_CALL(cudaStreamCreateWithPriority(&edge_stream, cudaStreamDefault, highpriority));

  const int top_pe = (rank + 1) % nranks;
  const int bot_pe = (rank + nranks - 1) % nranks;

  nvtxRangePushA("NCCL_Warmup");
  for (size_t i = 0; i < number_of_warmups; i++)
  {
    Halo_exchange(a_new, a, N, top_pe, iy_end, bot_pe, iy_start, ncclcomm, edge_stream);
    std::swap(a, a_new);
  }
  nvtxRangePop();
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  CUDA_CALL(cudaDeviceSynchronize());

  double start = MPI_Wtime();
  for (size_t it = 0; it < maxIt; ++it)
  {
    nvtx3::scoped_range loop{"Jacobi_Step"};

    nvtxRangePushA("Apply_stencil");
    launch_jacobi_kernel(a_new, a, iy_start + 1, iy_end - 1, N, inner_stream);
    launch_jacobi_kernel(a_new, a, iy_start, iy_start + 1, N, edge_stream);
    launch_jacobi_kernel(a_new, a, iy_end - 1, iy_end, N, edge_stream);
    nvtxRangePop();

    nvtxRangePushA("HALO_Exchange");
    Halo_exchange(a_new, a, N, top_pe, iy_end, bot_pe, iy_start, ncclcomm, edge_stream);
    nvtxRangePop();

    CUDA_CALL(cudaStreamSynchronize(inner_stream));
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
  CUDA_CALL(cudaFree(a));
  CUDA_CALL(cudaFree(a_new));
  CUDA_CALL(cudaStreamDestroy(inner_stream));
  CUDA_CALL(cudaStreamDestroy(edge_stream));
  NCCL_CALL(ncclCommDestroy(ncclcomm));
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

void Halo_exchange(real *a_new, real *a, int N, const int top, int iy_end, const int bottom, int iy_start, ncclComm_t nccl_comm, cudaStream_t edge_stream)
{
  NCCL_CALL(ncclGroupStart());
  // clang-format off
  // launch_jacobi_kernel(a_new, a, iy_start, iy_start + 1, N, top_stream); // for ref
  // launch_jacobi_kernel(a_new, a, iy_end - 1, iy_end, N, bottom_stream); // for ref
  NCCL_CALL(ncclRecv(a_new                   , N, NCCL_REAL_TYPE, top   , nccl_comm, edge_stream));
  NCCL_CALL(ncclSend(a_new + (iy_end - 1) * N, N, NCCL_REAL_TYPE, bottom, nccl_comm, edge_stream));
  NCCL_CALL(ncclRecv(a_new + (iy_end * N)    , N, NCCL_REAL_TYPE, bottom, nccl_comm, edge_stream));
  NCCL_CALL(ncclSend(a_new + (iy_start * N)  , N, NCCL_REAL_TYPE, top   , nccl_comm, edge_stream));
  // clang-format on
  NCCL_CALL(ncclGroupEnd());
  CUDA_CALL(cudaStreamSynchronize(edge_stream));
}
