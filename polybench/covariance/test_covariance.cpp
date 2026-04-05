#include "covariance.h"

void init_array(DATA_TYPE *data)
{
  for (int i = 0; i < _PB_N; i++)
    for (int j = 0; j < _PB_M; j++)
      data[IDX_DATA(i, j)] = ((DATA_TYPE)i * j) / _PB_M;
}

extern void covariance_serial(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean);
extern void bench_covariance_serial(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean);
extern void covariance_sycl(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean);
extern void bench_covariance_sycl(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean);

bool check_covariance()
{
  DATA_TYPE *data = (DATA_TYPE *)malloc(_PB_N * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *symmat_gold = (DATA_TYPE *)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean_gold = (DATA_TYPE *)malloc(_PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *symmat = (DATA_TYPE *)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean = (DATA_TYPE *)malloc(_PB_M * sizeof(DATA_TYPE));

  init_array(data);
  memset_zero(symmat_gold, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean_gold, _PB_M * sizeof(DATA_TYPE));
  covariance_serial(data, symmat_gold, mean_gold);

  init_array(data);
  memset_zero(symmat, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean, _PB_M * sizeof(DATA_TYPE));
  covariance_sycl(data, symmat, mean);

  bool symmat_ok = compare_array(symmat_gold, symmat, _PB_M * _PB_M);
  bool mean_ok = compare_array(mean_gold, mean, _PB_M);
  std::printf("compare symmat (sycl-naive): %s\n", symmat_ok ? "PASS" : "FAIL");
  std::printf("compare mean (%s): %s\n", "sycl-naive", mean_ok ? "PASS" : "FAIL");

  free(data);
  free(symmat_gold);
  free(mean_gold);
  free(symmat);
  free(mean);
  return symmat_ok && mean_ok;
}

void bench_covariance()
{
  DATA_TYPE *data = (DATA_TYPE *)malloc(_PB_N * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *symmat = (DATA_TYPE *)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean = (DATA_TYPE *)malloc(_PB_M * sizeof(DATA_TYPE));

  init_array(data);
  memset_zero(symmat, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean, _PB_M * sizeof(DATA_TYPE));
  bench_covariance_serial(data, symmat, mean);

  init_array(data);
  memset_zero(symmat, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean, _PB_M * sizeof(DATA_TYPE));
  bench_covariance_sycl(data, symmat, mean);

  free(data);
  free(symmat);
  free(mean);
}

int main()
{
  bool ok = check_covariance();
  bench_covariance();
  return ok ? 0 : 1;
}
