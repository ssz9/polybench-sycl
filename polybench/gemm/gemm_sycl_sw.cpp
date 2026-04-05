#include "gemm.h"
#include "sw_sycl_utils.h"

static void kernel_gemm_sycl_sw(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B,
                                buffer<DATA_TYPE, 2> &buf_C, queue &Q)
{
  Q.submit([&](handler &h) {
    auto A = buf_A.get_access<access::mode::read>(h);
    auto B = buf_B.get_access<access::mode::read>(h);
    auto C = buf_C.get_access<access::mode::read_write>(h);
    h.parallel_for<class GemmSwKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t row_begin = sw_block_begin(MATRIX_SIZE, worker_id);
      const size_t row_end = sw_block_end(MATRIX_SIZE, worker_id);
      for (size_t i = row_begin; i < row_end; ++i) {
        for (size_t j = 0; j < MATRIX_SIZE; ++j) {
          DATA_TYPE cij = C[i][j] * BETA;
          for (size_t k = 0; k < MATRIX_SIZE; ++k)
            cij += ALPHA * A[i][k] * B[k][j];
          C[i][j] = cij;
        }
      }
    });
  }).wait();
}

void gemm_sycl_sw(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    kernel_gemm_sycl_sw(buf_A, buf_B, buf_C, Q);
  }
}

void bench_gemm_sycl_sw(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE, MATRIX_SIZE));
    buffer<DATA_TYPE, 2> buf_C(C, range<2>(MATRIX_SIZE, MATRIX_SIZE));

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_gemm_sycl_sw(buf_A, buf_B, buf_C, Q);

    TIMEIT({
      kernel_gemm_sycl_sw(buf_A, buf_B, buf_C, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
