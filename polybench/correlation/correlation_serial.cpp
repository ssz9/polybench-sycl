
#include "correlation.h"

void kernel_correlation_serial(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  int i, j, k;
  DATA_TYPE eps = 0.1;
  DATA_TYPE float_n = (DATA_TYPE)_PB_N;

  for (j = 0; j < _PB_M; j++) {
    mean[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      mean[j] += data[IDX_DATA(i, j)];
    mean[j] /= float_n;
  }
  for (j = 0; j < _PB_M; j++) {
    stddev[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      stddev[j] += (data[IDX_DATA(i, j)] - mean[j]) * (data[IDX_DATA(i, j)] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = SQRT_FUN(stddev[j]);
    /* The following in an inelegant but usual way to handle near-zero std. dev. values, which below would cause a zero- divide. */
    stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  }
  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++) {
      data[IDX_DATA(i, j)] -= mean[j];
      data[IDX_DATA(i, j)] /= SQRT_FUN(float_n) * stddev[j];
    }
  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M; i++) {
    corr[IDX_CORR(i,i)] = 1.0;
    for (j = i + 1; j < _PB_M; j++) {
      corr[IDX_CORR(i,j)] = 0.0;
      for (k = 0; k < _PB_N; k++)
        corr[IDX_CORR(i,j)] += (data[IDX_DATA(k, i)] * data[IDX_DATA(k, j)]);
      corr[IDX_CORR(j,i)] = corr[IDX_CORR(i,j)];
    }
  }
}

void correlation_serial(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  kernel_correlation_serial(data, corr, mean, stddev);
}

void bench_correlation_serial(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  // warmup
  for(int i=0; i<WARMUP_REPS; i++) {
    kernel_correlation_serial(data, corr, mean, stddev);
  }
  // timing
  TIMEIT({
    kernel_correlation_serial(data, corr, mean, stddev);
  }, 3, "\n", "serial");
}
