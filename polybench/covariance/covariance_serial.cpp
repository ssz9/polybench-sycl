#include "covariance.h"

void kernel_covariance_serial(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  DATA_TYPE float_n = 1.2;
  int i, j, j1, j2;

  for (j = 0; j < _PB_M; j++) {
    mean[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      mean[j] += data[IDX_DATA(i, j)];
    mean[j] /= float_n;
  }

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[IDX_DATA(i, j)] -= mean[j];

  for (j1 = 0; j1 < _PB_M; j1++)
    for (j2 = j1; j2 < _PB_M; j2++) {
      symmat[IDX_SYMM(j1, j2)] = 0.0;
      for (i = 0; i < _PB_N; i++)
        symmat[IDX_SYMM(j1, j2)] += data[IDX_DATA(i, j1)] * data[IDX_DATA(i, j2)];
      symmat[IDX_SYMM(j2, j1)] = symmat[IDX_SYMM(j1, j2)];
    }
}

void covariance_serial(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  kernel_covariance_serial(data, symmat, mean);
}

void bench_covariance_serial(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  for (int i = 0; i < 1; i++)
    kernel_covariance_serial(data, symmat, mean);

  TIMEIT({
    kernel_covariance_serial(data, symmat, mean);
  }, 1, "\n", "serial");
}
