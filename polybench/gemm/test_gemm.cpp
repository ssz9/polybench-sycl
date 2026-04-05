#include "gemm.h"

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  for (size_t i = 0; i < MATRIX_SIZE; i++)
    for (size_t j = 0; j < MATRIX_SIZE; j++) {
      A[IDX(i, j)] = ((DATA_TYPE)i * j) / MATRIX_SIZE;
      B[IDX(i, j)] = ((DATA_TYPE)i * j + 1) / MATRIX_SIZE;
      C[IDX(i, j)] = ((DATA_TYPE)i * j + 2) / MATRIX_SIZE;
    }
}

extern void gemm_serial(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void bench_gemm_serial(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void gemm_sycl(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void bench_gemm_sycl(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
#ifdef PLF_SW
extern void gemm_sycl_sw(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void bench_gemm_sycl_sw(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
#endif
#ifdef PLF_A10
extern void gemm_sycl_a10(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void bench_gemm_sycl_a10(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
#endif
#ifdef PLF_DCU
extern void gemm_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
extern void bench_gemm_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C);
#endif

bool check_gemm()
{
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);
  DATA_TYPE *C_gold = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);
  DATA_TYPE *C = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);

  init_array(A, B, C_gold);
  gemm_serial(A, B, C_gold);

  init_array(A, B, C);
  gemm_sycl(A, B, C);

  bool c_ok = compare_array(C_gold, C, MATRIX_SIZE * MATRIX_SIZE);
  std::printf("compare C (sycl-naive): %s\n", c_ok ? "PASS" : "FAIL");

#ifdef PLF_SW
  init_array(A, B, C);
  gemm_sycl_sw(A, B, C);
  bool c_sw_ok = compare_array(C_gold, C, MATRIX_SIZE * MATRIX_SIZE);
  std::printf("compare C (sycl-sw): %s\n", c_sw_ok ? "PASS" : "FAIL");
  c_ok = c_ok && c_sw_ok;
#endif

  bool c_a10_ok = true;
  bool c_dcu_ok = true;
#ifdef PLF_A10
  init_array(A, B, C);
  gemm_sycl_a10(A, B, C);

  c_a10_ok = compare_array(C_gold, C, MATRIX_SIZE * MATRIX_SIZE);
  std::printf("compare C (sycl-a10): %s\n", c_a10_ok ? "PASS" : "FAIL");
#endif
#ifdef PLF_DCU
  init_array(A, B, C);
  gemm_sycl_dcu(A, B, C);

  c_dcu_ok = compare_array(C_gold, C, MATRIX_SIZE * MATRIX_SIZE);
  std::printf("compare C (sycl-dcu): %s\n", c_dcu_ok ? "PASS" : "FAIL");
#endif

  free(A);
  free(B);
  free(C_gold);
  free(C);
  return c_ok && c_a10_ok && c_dcu_ok;
}

void bench_gemm()
{
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);
  DATA_TYPE *C = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * MATRIX_SIZE * MATRIX_SIZE);

  init_array(A, B, C);
  bench_gemm_serial(A, B, C);

  init_array(A, B, C);
  bench_gemm_sycl(A, B, C);

#ifdef PLF_SW
  init_array(A, B, C);
  bench_gemm_sycl_sw(A, B, C);
#endif

#ifdef PLF_A10
  init_array(A, B, C);
  bench_gemm_sycl_a10(A, B, C);
#endif
#ifdef PLF_DCU
  init_array(A, B, C);
  bench_gemm_sycl_dcu(A, B, C);
#endif

  free(A);
  free(B);
  free(C);
}

int main()
{
  bool ok = check_gemm();
  bench_gemm();
  return ok ? 0 : 1;
}
