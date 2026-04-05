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
#ifdef PLF_SW
extern void heat_3d_sycl_sw(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_heat_3d_sycl_sw(DATA_TYPE *A, DATA_TYPE *B);
#endif
#ifdef PLF_A10
extern void heat_3d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_heat_3d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B);
#endif
#ifdef PLF_DCU
extern void heat_3d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_heat_3d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B);
#endif

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

#ifdef PLF_SW
  init_array(A, B);
  heat_3d_sycl_sw(A, B);
  bool a_sw_ok = compare_array(A_gold, A, size);
  bool b_sw_ok = compare_array(B_gold, B, size);
  std::printf("compare A (sycl-sw): %s\n", a_sw_ok ? "PASS" : "FAIL");
  std::printf("compare B (sycl-sw): %s\n", b_sw_ok ? "PASS" : "FAIL");
  a_ok = a_ok && a_sw_ok;
  b_ok = b_ok && b_sw_ok;
#endif

#ifdef PLF_A10
  init_array(A, B);
  heat_3d_sycl_a10(A, B);

  bool a_a10_ok = compare_array(A_gold, A, size);
  bool b_a10_ok = compare_array(B_gold, B, size);
  std::printf("compare A (sycl-a10): %s\n", a_a10_ok ? "PASS" : "FAIL");
  std::printf("compare B (sycl-a10): %s\n", b_a10_ok ? "PASS" : "FAIL");
  a_ok = a_ok && a_a10_ok;
  b_ok = b_ok && b_a10_ok;
#endif
#ifdef PLF_DCU
  init_array(A, B);
  heat_3d_sycl_dcu(A, B);

  bool a_dcu_ok = compare_array(A_gold, A, size);
  bool b_dcu_ok = compare_array(B_gold, B, size);
  std::printf("compare A (sycl-dcu): %s\n", a_dcu_ok ? "PASS" : "FAIL");
  std::printf("compare B (sycl-dcu): %s\n", b_dcu_ok ? "PASS" : "FAIL");
  a_ok = a_ok && a_dcu_ok;
  b_ok = b_ok && b_dcu_ok;
#endif

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
#ifdef PLF_SW
  init_array(A, B);
  bench_heat_3d_sycl_sw(A, B);
#endif
#ifdef PLF_A10
  init_array(A, B);
  bench_heat_3d_sycl_a10(A, B);
#endif
#ifdef PLF_DCU
  init_array(A, B);
  bench_heat_3d_sycl_dcu(A, B);
#endif

  free(A);
  free(B);
}

int main()
{
  bool ok = check_heat_3d();
  bench_heat_3d();
  return ok ? 0 : 1;
}
