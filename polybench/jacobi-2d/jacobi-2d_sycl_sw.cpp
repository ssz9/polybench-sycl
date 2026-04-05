#include "jacobi-2d.h"
#include "sw_sycl_utils.h"

static void kernel_jacobi_2d_sycl_sw(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B, queue &Q)
{
  for (int t = 0; t < TSTEPS; ++t) {
    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::read>(h);
      auto B = buf_B.get_access<access::mode::write>(h);
      h.parallel_for<class Jacobi2DSwStepBKernel>(sw_cpe_range(), [=](id<1> worker) {
        const size_t worker_id = sw_worker_id(worker);
        const size_t row_begin = sw_block_begin(MATRIX_SIZE_H - 2, worker_id) + 1;
        const size_t row_end = sw_block_end(MATRIX_SIZE_H - 2, worker_id) + 1;
        for (size_t i = row_begin; i < row_end; ++i)
          for (int j = 1; j < MATRIX_SIZE_W - 1; ++j)
            B[i][j] = 0.201 * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
      });
    }).wait();

    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::write>(h);
      auto B = buf_B.get_access<access::mode::read>(h);
      h.parallel_for<class Jacobi2DSwStepAKernel>(sw_cpe_range(), [=](id<1> worker) {
        const size_t worker_id = sw_worker_id(worker);
        const size_t row_begin = sw_block_begin(MATRIX_SIZE_H - 2, worker_id) + 1;
        const size_t row_end = sw_block_end(MATRIX_SIZE_H - 2, worker_id) + 1;
        for (size_t i = row_begin; i < row_end; ++i)
          for (int j = 1; j < MATRIX_SIZE_W - 1; ++j)
            A[i][j] = B[i][j];
      });
    }).wait();
  }
}

void jacobi_2d_sycl_sw(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    kernel_jacobi_2d_sycl_sw(buf_A, buf_B, Q);
  }
}

void bench_jacobi_2d_sycl_sw(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_jacobi_2d_sycl_sw(buf_A, buf_B, Q);

    TIMEIT({
      kernel_jacobi_2d_sycl_sw(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
