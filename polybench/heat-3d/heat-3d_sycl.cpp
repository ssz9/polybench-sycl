#include "heat-3d.h"

void kernel_heat_3d_sycl(buffer<DATA_TYPE, 3> &buf_A, buffer<DATA_TYPE, 3> &buf_B, queue &Q)
{
  for (int t = 1; t <= TSTEPS; t++) {
    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::read>(h);
      auto B = buf_B.get_access<access::mode::read_write>(h);
      h.parallel_for<class Heat3DStepBKernel>(range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N), [=](item<3> item) {
        size_t i = item[0];
        size_t j = item[1];
        size_t k = item[2];
        if (i >= 1 && i < HEAT_3D_N - 1 && j >= 1 && j < HEAT_3D_N - 1 && k >= 1 && k < HEAT_3D_N - 1)
          B[{i, j, k}] = SCALAR_VAL(0.125) * (A[{i + 1, j, k}] - SCALAR_VAL(2.0) * A[{i, j, k}] + A[{i - 1, j, k}]) +
                         SCALAR_VAL(0.125) * (A[{i, j + 1, k}] - SCALAR_VAL(2.0) * A[{i, j, k}] + A[{i, j - 1, k}]) +
                         SCALAR_VAL(0.125) * (A[{i, j, k + 1}] - SCALAR_VAL(2.0) * A[{i, j, k}] + A[{i, j, k - 1}]) +
                         A[{i, j, k}];
      });
    }).wait();

    Q.submit([&](handler &h) {
      auto A = buf_A.get_access<access::mode::read_write>(h);
      auto B = buf_B.get_access<access::mode::read>(h);
      h.parallel_for<class Heat3DStepAKernel>(range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N), [=](item<3> item) {
        size_t i = item[0];
        size_t j = item[1];
        size_t k = item[2];
        if (i >= 1 && i < HEAT_3D_N - 1 && j >= 1 && j < HEAT_3D_N - 1 && k >= 1 && k < HEAT_3D_N - 1)
          A[{i, j, k}] = SCALAR_VAL(0.125) * (B[{i + 1, j, k}] - SCALAR_VAL(2.0) * B[{i, j, k}] + B[{i - 1, j, k}]) +
                         SCALAR_VAL(0.125) * (B[{i, j + 1, k}] - SCALAR_VAL(2.0) * B[{i, j, k}] + B[{i, j - 1, k}]) +
                         SCALAR_VAL(0.125) * (B[{i, j, k + 1}] - SCALAR_VAL(2.0) * B[{i, j, k}] + B[{i, j, k - 1}]) +
                         B[{i, j, k}];
      });
    }).wait();
  }
}

void heat_3d_sycl(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    kernel_heat_3d_sycl(buf_A, buf_B, Q);
  }
}

void bench_heat_3d_sycl(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_heat_3d_sycl(buf_A, buf_B, Q);

    TIMEIT({
      kernel_heat_3d_sycl(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-naive");
  }
}
