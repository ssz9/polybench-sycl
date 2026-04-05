#include "heat-3d.h"

void init_array(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int i = 0; i < HEAT_3D_N; i++)
    for (int j = 0; j < HEAT_3D_N; j++)
      for (int k = 0; k < HEAT_3D_N; k++)
        A[INDEX(i, j, k)] = B[INDEX(i, j, k)] = (DATA_TYPE)(i + j + (HEAT_3D_N - k)) * 10 / HEAT_3D_N;
}

extern void heat_3d_serial(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_heat_3d_serial(DATA_TYPE *A, DATA_TYPE *B);
extern void heat_3d_sycl(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_heat_3d_sycl(DATA_TYPE *A, DATA_TYPE *B);

bool check_heat_3d()
{
  size_t size = (size_t)HEAT_3D_N * HEAT_3D_N * HEAT_3D_N;
  DATA_TYPE *A_gold = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B_gold = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);

  init_array(A_gold, B_gold);
  heat_3d_serial(A_gold, B_gold);

  init_array(A, B);
  heat_3d_sycl(A, B);

  bool a_ok = compare_array(A_gold, A, size);
  bool b_ok = compare_array(B_gold, B, size);
  std::printf("compare A (sycl-naive): %s\n", a_ok ? "PASS" : "FAIL");
  std::printf("compare B (sycl-naive): %s\n", b_ok ? "PASS" : "FAIL");

  free(A_gold);
  free(B_gold);
  free(A);
  free(B);
  return a_ok && b_ok;
}

void bench_heat_3d()
{
  size_t size = (size_t)HEAT_3D_N * HEAT_3D_N * HEAT_3D_N;
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);

  init_array(A, B);
  bench_heat_3d_serial(A, B);

  init_array(A, B);
  bench_heat_3d_sycl(A, B);

  free(A);
  free(B);
}

int main()
{
  bool ok = check_heat_3d();
  bench_heat_3d();
  return ok ? 0 : 1;
}
