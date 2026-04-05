#include "gemm.h"

void kernel_gemm_serial(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  for (size_t i = 0; i < MATRIX_SIZE; i++) {
    for (size_t j = 0; j < MATRIX_SIZE; j++) {
      C[IDX(i, j)] *= BETA;
      for (size_t k = 0; k < MATRIX_SIZE; k++)
        C[IDX(i, j)] += ALPHA * A[IDX(i, k)] * B[IDX(k, j)];
    }
  }
}

void gemm_serial(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  kernel_gemm_serial(A, B, C);
}

void bench_gemm_serial(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  for (int i = 0; i < WARMUP_REPS; i++)
    kernel_gemm_serial(A, B, C);

  TIMEIT({
    kernel_gemm_serial(A, B, C);
  }, BENCH_REPS, "\n", "serial");
}
