#include "gemm.h"

namespace {

constexpr size_t A10_BLOCK_M = 64;
constexpr size_t A10_BLOCK_N = 64;
constexpr size_t A10_BLOCK_K = 16;
constexpr size_t A10_THREAD_M = 16;
constexpr size_t A10_THREAD_N = 1;
constexpr size_t A10_LOCAL_M = A10_BLOCK_M / A10_THREAD_M;
constexpr size_t A10_LOCAL_N = A10_BLOCK_N / A10_THREAD_N;
constexpr size_t A10_TILE_A_STRIDE = A10_BLOCK_K;
constexpr size_t A10_TILE_B_STRIDE = A10_BLOCK_N;

} // namespace

void kernel_gemm_sycl_a10(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B, buffer<DATA_TYPE, 2> &buf_C, queue &Q)
{
  const range<2> local_range{A10_LOCAL_M, A10_LOCAL_N};
  const range<2> global_range{
      ((MATRIX_SIZE + A10_BLOCK_M - 1) / A10_BLOCK_M) * A10_LOCAL_M,
      ((MATRIX_SIZE + A10_BLOCK_N - 1) / A10_BLOCK_N) * A10_LOCAL_N};

  Q.submit([&](handler &h) {
    auto A = buf_A.get_access<access::mode::read>(h);
    auto B = buf_B.get_access<access::mode::read>(h);
    auto C = buf_C.get_access<access::mode::read_write>(h);
    local_accessor<DATA_TYPE, 1> tile_A(range<1>(A10_BLOCK_M * A10_TILE_A_STRIDE), h);
    local_accessor<DATA_TYPE, 1> tile_B(range<1>(A10_BLOCK_K * A10_TILE_B_STRIDE), h);

    h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) [[sycl::reqd_work_group_size(A10_LOCAL_M, A10_LOCAL_N)]] {
      const size_t local_row = item.get_local_id(0);
      const size_t local_col = item.get_local_id(1);
      const size_t group_row = item.get_group(0);
      const size_t group_col = item.get_group(1);

      const size_t row_base = group_row * A10_BLOCK_M + local_row * A10_THREAD_M;
      const size_t col_base = group_col * A10_BLOCK_N + local_col * A10_THREAD_N;
      const size_t linear_tid = local_row * A10_LOCAL_N + local_col;
      const size_t local_size = A10_LOCAL_M * A10_LOCAL_N;

      DATA_TYPE accum[A10_THREAD_M][A10_THREAD_N];
      for (size_t i = 0; i < A10_THREAD_M; i++)
        for (size_t j = 0; j < A10_THREAD_N; j++)
          accum[i][j] = 0.0;

      for (size_t k0 = 0; k0 < MATRIX_SIZE; k0 += A10_BLOCK_K) {
        for (size_t idx = linear_tid; idx < A10_BLOCK_M * A10_BLOCK_K; idx += local_size) {
          const size_t tile_row = idx / A10_BLOCK_K;
          const size_t tile_col = idx % A10_BLOCK_K;
          tile_A[tile_row * A10_TILE_A_STRIDE + tile_col] = A[group_row * A10_BLOCK_M + tile_row][k0 + tile_col];
        }

        for (size_t idx = linear_tid; idx < A10_BLOCK_K * A10_BLOCK_N; idx += local_size) {
          const size_t tile_row = idx / A10_BLOCK_N;
          const size_t tile_col = idx % A10_BLOCK_N;
          tile_B[tile_row * A10_TILE_B_STRIDE + tile_col] = B[k0 + tile_row][group_col * A10_BLOCK_N + tile_col];
        }

        item.barrier(access::fence_space::local_space);

        for (size_t kk = 0; kk < A10_BLOCK_K; kk++) {
          DATA_TYPE reg_a[A10_THREAD_M];
          DATA_TYPE reg_b[A10_THREAD_N];

          #pragma unroll
          for (size_t i = 0; i < A10_THREAD_M; i++)
            reg_a[i] = tile_A[(local_row * A10_THREAD_M + i) * A10_TILE_A_STRIDE + kk];

          #pragma unroll
          for (size_t j = 0; j < A10_THREAD_N; j++)
            reg_b[j] = tile_B[kk * A10_TILE_B_STRIDE + local_col * A10_THREAD_N + j];

          #pragma unroll
          for (size_t i = 0; i < A10_THREAD_M; i++) {
            #pragma unroll
            for (size_t j = 0; j < A10_THREAD_N; j++)
              accum[i][j] = sycl::fma(reg_a[i], reg_b[j], accum[i][j]);
          }
        }

        item.barrier(access::fence_space::local_space);
      }

      #pragma unroll
      for (size_t i = 0; i < A10_THREAD_M; i++) {
        #pragma unroll
        for (size_t j = 0; j < A10_THREAD_N; j++) {
          const size_t row = row_base + i;
          const size_t col = col_base + j;
          C[row][col] = sycl::fma(ALPHA, accum[i][j], BETA * C[row][col]);
        }
      }
    });
  }).wait();
}

void gemm_sycl_a10(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    kernel_gemm_sycl_a10(buf_A, buf_B, buf_C, Q);
  }
}

void bench_gemm_sycl_a10(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_gemm_sycl_a10(buf_A, buf_B, buf_C, Q);

    TIMEIT({
      kernel_gemm_sycl_a10(buf_A, buf_B, buf_C, Q);
    }, BENCH_REPS, "\n", "sycl-a10");
  }
}
