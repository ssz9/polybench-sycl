#include "header.h"


void init_array(DATA_TYPE *data)
{
  for (int i = 0; i < _PB_N; i++)
    for (int j = 0; j < _PB_M; j++)
      data[IDX_DATA(i,j)] = (DATA_TYPE)(i*j)/_PB_M + i;
}
void print_corr(int m, DATA_TYPE *corr)
{
  assert(m <= _PB_M);
  //print up left m*m 
  for (int i = 0; i < m; i++){
    for (int j = 0; j < m; j++) printf("%-10.4lf ", corr[IDX_CORR(i, j)]);
    printf("\n");
  }
  printf(".\n .\n  .\n");
  //print down right m*m
  for (int i = _PB_M-m; i < _PB_M; i++){
    printf("  ");
    for (int j = _PB_M-m; j < _PB_M; j++) printf("%-10.4lf ", corr[IDX_CORR(i, j)]);
    printf("\n");
  }
  printf("\n");
}
void print_data(int n, int m, DATA_TYPE *data)
{
  //print up left n*m
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++) printf("%-10.4lf ", data[IDX_DATA(i, j)]);
    printf("\n");
  }
  printf(".\n .\n  .\n");
  //print down right n*m
  for (int i = _PB_N-n; i < _PB_N; i++){
    printf("  ");
    for (int j = _PB_M-m; j < _PB_M; j++) printf("%-10.4lf ", data[IDX_DATA(i, j)]);
    printf("\n");
  }
}

extern void correlation_serial(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev);
extern void bench_correlation_serial(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev);
extern void correlation_sycl(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev);
extern void bench_correlation_sycl(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev);

bool check_correlation()
{
  DATA_TYPE *data = (DATA_TYPE*)malloc(_PB_N * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *corr_gold = (DATA_TYPE*)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean_gold = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *stddev_gold = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *corr = (DATA_TYPE*)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *stddev = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));

  init_array(data);
  memset_zero(corr_gold, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean_gold, _PB_M * sizeof(DATA_TYPE));
  memset_zero(stddev_gold, _PB_M * sizeof(DATA_TYPE));
  correlation_serial(data, corr_gold, mean_gold, stddev_gold);
  // print_corr(10, corr_gold);

  std::string impl_label = "";

  // naive sycl impl
  impl_label = "sycl-naive";
  init_array(data);
  memset_zero(corr, _PB_M * _PB_M * sizeof(DATA_TYPE));
  memset_zero(mean, _PB_M * sizeof(DATA_TYPE));
  memset_zero(stddev, _PB_M * sizeof(DATA_TYPE));
  correlation_sycl(data, corr, mean, stddev);
  // print_corr(10, corr); 
  bool corr_ok = compare_array(corr_gold, corr, _PB_M * _PB_M); std::printf("compare corr (%s): %s\n", impl_label.c_str(), corr_ok ? "PASS" : "FAIL");
  bool mean_ok = compare_array(mean_gold, mean, _PB_M); std::printf("compare mean (%s): %s\n", impl_label.c_str(), mean_ok ? "PASS" : "FAIL");
  bool stddev_ok = compare_array(stddev_gold, stddev, _PB_M); std::printf("compare stddev (%s): %s\n", impl_label.c_str(), stddev_ok ? "PASS" : "FAIL");

  // // xxx impl
  // impl_label = "xxx";
  // init_array(data);
  // memset_zero(corr, _PB_M * _PB_M * sizeof(DATA_TYPE));
  // memset_zero(mean, _PB_M * sizeof(DATA_TYPE));
  // memset_zero(stddev, _PB_M * sizeof(DATA_TYPE));
  // correlation_sycl(data, corr, mean, stddev);
  // // print_corr(10, corr); 
  // bool corr_ok = compare_array(corr_gold, corr, _PB_M * _PB_M); std::printf("compare corr (%s): %s\n", impl_label.c_str(), corr_ok ? "PASS" : "FAIL");
  // bool mean_ok = compare_array(mean_gold, mean, _PB_M); std::printf("compare mean (%s): %s\n", impl_label.c_str(), mean_ok ? "PASS" : "FAIL");
  // bool stddev_ok = compare_array(stddev_gold, stddev, _PB_M); std::printf("compare stddev (%s): %s\n", impl_label.c_str(), stddev_ok ? "PASS" : "FAIL");
  

  free(data);
  free(corr);
  free(mean);
  free(stddev);
  free(corr_gold);
  free(mean_gold);
  free(stddev_gold);
  return corr_ok && mean_ok && stddev_ok;
}

void bench_correlation()
{
  DATA_TYPE *data = (DATA_TYPE*)malloc(_PB_N * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *corr = (DATA_TYPE*)malloc(_PB_M * _PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *mean = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));
  DATA_TYPE *stddev = (DATA_TYPE*)malloc(_PB_M * sizeof(DATA_TYPE));

  bench_correlation_serial(data, corr, mean, stddev);
  bench_correlation_sycl(data, corr, mean, stddev);

  free(data);
  free(corr);
  free(mean);
  free(stddev);
}

int main(){
  bool ok = check_correlation();
  bench_correlation();
  return ok ? 0 : 1;
}
