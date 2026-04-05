#include "jacobi-2d.h"

void kernel_jacobi_2d_sycl(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B, queue &Q)
{
  for (int t = 0; t < TSTEPS; t++) {
    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::read>(h);
      auto B = buf_B.get_access<access::mode::write>(h);
      h.parallel_for<class Jacobi2DStepBKernel>(range<2>(MATRIX_SIZE_H - 2, MATRIX_SIZE_W - 2), [=](item<2> item) {
        int i = item[0] + 1;
        int j = item[1] + 1;
        B[i][j] = 0.201 * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
      });
    }).wait();

    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::write>(h);
      auto B = buf_B.get_access<access::mode::read>(h);
      h.parallel_for<class Jacobi2DStepAKernel>(range<2>(MATRIX_SIZE_H - 2, MATRIX_SIZE_W - 2), [=](item<2> item) {
        int i = item[0] + 1;
        int j = item[1] + 1;
        A[i][j] = B[i][j];
      });
    }).wait();
  }
}

void jacobi_2d_sycl(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    kernel_jacobi_2d_sycl(buf_A, buf_B, Q);
  }
}

void bench_jacobi_2d_sycl(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_jacobi_2d_sycl(buf_A, buf_B, Q);

    TIMEIT({
      kernel_jacobi_2d_sycl(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-naive");
  }
}
