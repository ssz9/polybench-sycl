#include "heat-3d.h"

void kernel_heat_3d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int t = 1; t <= TSTEPS; t++) {
    for (int i = 1; i < HEAT_3D_N - 1; i++)
      for (int j = 1; j < HEAT_3D_N - 1; j++)
        for (int k = 1; k < HEAT_3D_N - 1; k++)
          B[INDEX(i, j, k)] = SCALAR_VAL(0.125) * (A[INDEX(i + 1, j, k)] - SCALAR_VAL(2.0) * A[INDEX(i, j, k)] + A[INDEX(i - 1, j, k)]) +
                              SCALAR_VAL(0.125) * (A[INDEX(i, j + 1, k)] - SCALAR_VAL(2.0) * A[INDEX(i, j, k)] + A[INDEX(i, j - 1, k)]) +
                              SCALAR_VAL(0.125) * (A[INDEX(i, j, k + 1)] - SCALAR_VAL(2.0) * A[INDEX(i, j, k)] + A[INDEX(i, j, k - 1)]) +
                              A[INDEX(i, j, k)];

    for (int i = 1; i < HEAT_3D_N - 1; i++)
      for (int j = 1; j < HEAT_3D_N - 1; j++)
        for (int k = 1; k < HEAT_3D_N - 1; k++)
          A[INDEX(i, j, k)] = SCALAR_VAL(0.125) * (B[INDEX(i + 1, j, k)] - SCALAR_VAL(2.0) * B[INDEX(i, j, k)] + B[INDEX(i - 1, j, k)]) +
                              SCALAR_VAL(0.125) * (B[INDEX(i, j + 1, k)] - SCALAR_VAL(2.0) * B[INDEX(i, j, k)] + B[INDEX(i, j - 1, k)]) +
                              SCALAR_VAL(0.125) * (B[INDEX(i, j, k + 1)] - SCALAR_VAL(2.0) * B[INDEX(i, j, k)] + B[INDEX(i, j, k - 1)]) +
                              B[INDEX(i, j, k)];
  }
}

void heat_3d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  kernel_heat_3d_serial(A, B);
}

void bench_heat_3d_serial(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int i = 0; i < 1; i++)
    kernel_heat_3d_serial(A, B);

  TIMEIT({
    kernel_heat_3d_serial(A, B);
  }, 1, "\n", "serial");
}
