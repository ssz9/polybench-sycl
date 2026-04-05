#include "gemm.h"

namespace {

constexpr size_t DCU_BLOCK_M = 64;
constexpr size_t DCU_BLOCK_N = 64;
constexpr size_t DCU_BLOCK_K = 16;
constexpr size_t DCU_THREAD_M = 4;
constexpr size_t DCU_THREAD_N = 4;
constexpr size_t DCU_LOCAL_M = DCU_BLOCK_M / DCU_THREAD_M;
constexpr size_t DCU_LOCAL_N = DCU_BLOCK_N / DCU_THREAD_N;
constexpr size_t DCU_LOCAL_SIZE = DCU_LOCAL_M * DCU_LOCAL_N;

void kernel_gemm_sycl_dcu(buffer<DATA_TYPE, 1> &buf_A, buffer<DATA_TYPE, 1> &buf_B,
                          buffer<DATA_TYPE, 1> &buf_C, queue &Q)
{
  const range<2> local_range{DCU_LOCAL_M, DCU_LOCAL_N};
  const range<2> global_range{
      (MATRIX_SIZE / DCU_BLOCK_M) * DCU_LOCAL_M,
      (MATRIX_SIZE / DCU_BLOCK_N) * DCU_LOCAL_N};

  Q.submit([&](handler &h) {
    auto A = buf_A.get_access<access::mode::read>(h);
    auto B = buf_B.get_access<access::mode::read>(h);
    auto C = buf_C.get_access<access::mode::read_write>(h);
    local_accessor<DATA_TYPE, 2> tile_A(range<2>(DCU_BLOCK_M, DCU_BLOCK_K), h);
    local_accessor<DATA_TYPE, 2> tile_B(range<2>(DCU_BLOCK_K, DCU_BLOCK_N), h);

    h.parallel_for(
        nd_range<2>(global_range, local_range),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(DCU_LOCAL_M, DCU_LOCAL_N)]] {
          const size_t local_row = item.get_local_id(0);
          const size_t local_col = item.get_local_id(1);
          const size_t group_row = item.get_group(0);
          const size_t group_col = item.get_group(1);
          const size_t linear_tid = local_row * DCU_LOCAL_N + local_col;

          const size_t row_base = group_row * DCU_BLOCK_M + local_row * DCU_THREAD_M;
          const size_t col_base = group_col * DCU_BLOCK_N + local_col * DCU_THREAD_N;

          DATA_TYPE accum[DCU_THREAD_M][DCU_THREAD_N];
          #pragma unroll
          for (size_t i = 0; i < DCU_THREAD_M; i++) {
            #pragma unroll
            for (size_t j = 0; j < DCU_THREAD_N; j++)
              accum[i][j] = 0;
          }

          for (size_t k0 = 0; k0 < MATRIX_SIZE; k0 += DCU_BLOCK_K) {
            for (size_t idx = linear_tid; idx < DCU_BLOCK_M * DCU_BLOCK_K; idx += DCU_LOCAL_SIZE) {
              const size_t tile_row = idx / DCU_BLOCK_K;
              const size_t tile_col = idx % DCU_BLOCK_K;
              tile_A[tile_row][tile_col] = A[(group_row * DCU_BLOCK_M + tile_row) * MATRIX_SIZE + (k0 + tile_col)];
            }

            for (size_t idx = linear_tid; idx < DCU_BLOCK_K * DCU_BLOCK_N; idx += DCU_LOCAL_SIZE) {
              const size_t tile_row = idx / DCU_BLOCK_N;
              const size_t tile_col = idx % DCU_BLOCK_N;
              tile_B[tile_row][tile_col] = B[(k0 + tile_row) * MATRIX_SIZE + (group_col * DCU_BLOCK_N + tile_col)];
            }

            item.barrier(access::fence_space::local_space);

            #pragma unroll
            for (size_t kk = 0; kk < DCU_BLOCK_K; kk++) {
              DATA_TYPE reg_a[DCU_THREAD_M];
              DATA_TYPE reg_b[DCU_THREAD_N];

              #pragma unroll
              for (size_t i = 0; i < DCU_THREAD_M; i++)
                reg_a[i] = tile_A[local_row * DCU_THREAD_M + i][kk];

              #pragma unroll
              for (size_t j = 0; j < DCU_THREAD_N; j++)
                reg_b[j] = tile_B[kk][local_col * DCU_THREAD_N + j];

              #pragma unroll
              for (size_t i = 0; i < DCU_THREAD_M; i++) {
                #pragma unroll
                for (size_t j = 0; j < DCU_THREAD_N; j++)
                  accum[i][j] = sycl::fma(reg_a[i], reg_b[j], accum[i][j]);
              }
            }

            item.barrier(access::fence_space::local_space);
          }

          #pragma unroll
          for (size_t i = 0; i < DCU_THREAD_M; i++) {
            const size_t row = row_base + i;
            #pragma unroll
            for (size_t j = 0; j < DCU_THREAD_N; j++) {
              const size_t col = col_base + j;
              const size_t idx = row * MATRIX_SIZE + col;
              C[idx] = sycl::fma(ALPHA, accum[i][j], BETA * C[idx]);
            }
          }
        });
  }).wait();
}

} // namespace

void gemm_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 1> buf_A(A, range<1>(MATRIX_SIZE * MATRIX_SIZE));
    buffer<DATA_TYPE, 1> buf_B(B, range<1>(MATRIX_SIZE * MATRIX_SIZE));
    buffer<DATA_TYPE, 1> buf_C(C, range<1>(MATRIX_SIZE * MATRIX_SIZE));
    kernel_gemm_sycl_dcu(buf_A, buf_B, buf_C, Q);
  }
}

void bench_gemm_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 1> buf_A(A, range<1>(MATRIX_SIZE * MATRIX_SIZE));
    buffer<DATA_TYPE, 1> buf_B(B, range<1>(MATRIX_SIZE * MATRIX_SIZE));
    buffer<DATA_TYPE, 1> buf_C(C, range<1>(MATRIX_SIZE * MATRIX_SIZE));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_gemm_sycl_dcu(buf_A, buf_B, buf_C, Q);

    TIMEIT({
      kernel_gemm_sycl_dcu(buf_A, buf_B, buf_C, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
