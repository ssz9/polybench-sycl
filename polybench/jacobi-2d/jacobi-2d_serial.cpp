#include "jacobi-2d.h"

void kernel_jacobi_2d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int t = 0; t < TSTEPS; t++) {
    for (int i = 1; i < MATRIX_SIZE_H - 1; i++)
      for (int j = 1; j < MATRIX_SIZE_W - 1; j++)
        B[IDX(i, j)] = 0.201 * (A[IDX(i, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)] +
                                A[IDX(i + 1, j)] + A[IDX(i - 1, j)]);

    for (int i = 1; i < MATRIX_SIZE_H - 1; i++)
      for (int j = 1; j < MATRIX_SIZE_W - 1; j++)
        A[IDX(i, j)] = B[IDX(i, j)];
  }
}

void jacobi_2d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  kernel_jacobi_2d_serial(A, B);
}

void bench_jacobi_2d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int i = 0; i < WARMUP_REPS; i++)
    kernel_jacobi_2d_serial(A, B);

  TIMEIT({
    kernel_jacobi_2d_serial(A, B);
  }, BENCH_REPS, "\n", "serial");
}
