#include "jacobi-2d.h"

void init_array(DATA_TYPE *A, DATA_TYPE *B)
{
  for (int i = 0; i < MATRIX_SIZE_H; i++)
    for (int j = 0; j < MATRIX_SIZE_W; j++)
      A[IDX(i, j)] = B[IDX(i, j)] = ((DATA_TYPE)i * (j + 2) + 2) / MATRIX_SIZE_W;
}

extern void jacobi_2d_serial(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_jacobi_2d_serial(DATA_TYPE *A, DATA_TYPE *B);
extern void jacobi_2d_sycl(DATA_TYPE *A, DATA_TYPE *B);
extern void bench_jacobi_2d_sycl(DATA_TYPE *A, DATA_TYPE *B);

bool check_jacobi_2d()
{
  size_t size = (size_t)MATRIX_SIZE_H * MATRIX_SIZE_W;
  DATA_TYPE *A_gold = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B_gold = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);

  init_array(A_gold, B_gold);
  jacobi_2d_serial(A_gold, B_gold);

  init_array(A, B);
  jacobi_2d_sycl(A, B);

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

void bench_jacobi_2d()
{
  size_t size = (size_t)MATRIX_SIZE_H * MATRIX_SIZE_W;
  DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
  DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);

  init_array(A, B);
  bench_jacobi_2d_serial(A, B);

  init_array(A, B);
  bench_jacobi_2d_sycl(A, B);

  free(A);
  free(B);
}

int main()
{
  bool ok = check_jacobi_2d();
  bench_jacobi_2d();
  return ok ? 0 : 1;
}
