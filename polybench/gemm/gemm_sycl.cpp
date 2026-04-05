#include "gemm.h"

void kernel_gemm_sycl(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B,
                      buffer<DATA_TYPE, 2> &buf_C, queue &Q)
{
  Q.submit([&](handler &h) {
    auto A = buf_A.get_access<access::mode::read>(h);
    auto B = buf_B.get_access<access::mode::read>(h);
    auto C = buf_C.get_access<access::mode::read_write>(h);
    h.parallel_for<class GemmSyclKernel>(range<2>(MATRIX_SIZE, MATRIX_SIZE), [=](item<2> item) {
      size_t i = item[0];
      size_t j = item[1];
      C[item] *= BETA;
      for (size_t k = 0; k < MATRIX_SIZE; k++)
        C[item] += ALPHA * A[{i, k}] * B[{k, j}];
    });
  }).wait();
}

void gemm_sycl(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    kernel_gemm_sycl(buf_A, buf_B, buf_C, Q);
  }
}

void bench_gemm_sycl(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_gemm_sycl(buf_A, buf_B, buf_C, Q);

    TIMEIT({
      kernel_gemm_sycl(buf_A, buf_B, buf_C, Q);
    }, BENCH_REPS, "\n", "sycl-naive");
  }
}
