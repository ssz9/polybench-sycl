#include "covariance.h"
#include "sw_sycl_utils.h"

static void kernel_covariance_sycl_sw(buffer<DATA_TYPE, 2> &data_buf, buffer<DATA_TYPE, 2> &symmat_buf,
                                      buffer<DATA_TYPE, 1> &mean_buf, queue &Q)
{
  const DATA_TYPE float_n = 1.2;

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::write>(h);
    h.parallel_for<class CovarianceSwMeanKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t col_begin = sw_block_begin(_PB_M, worker_id);
      const size_t col_end = sw_block_end(_PB_M, worker_id);
      for (size_t j = col_begin; j < col_end; ++j) {
        DATA_TYPE acc = 0.0;
        for (int i = 0; i < _PB_N; ++i)
          acc += data[i][j];
        mean[j] = acc / float_n;
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read_write>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    h.parallel_for<class CovarianceSwCenterKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t row_begin = sw_block_begin(_PB_N, worker_id);
      const size_t row_end = sw_block_end(_PB_N, worker_id);
      for (size_t i = row_begin; i < row_end; ++i)
        for (int j = 0; j < _PB_M; ++j)
          data[i][j] -= mean[j];
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto symmat = symmat_buf.get_access<access::mode::write>(h);
    h.parallel_for<class CovarianceSwSymmatKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t row_begin = sw_block_begin(_PB_M, worker_id);
      const size_t row_end = sw_block_end(_PB_M, worker_id);
      for (size_t j1 = row_begin; j1 < row_end; ++j1) {
        for (size_t j2 = j1; j2 < _PB_M; ++j2) {
          DATA_TYPE acc = 0.0;
          for (int i = 0; i < _PB_N; ++i)
            acc += data[i][j1] * data[i][j2];
          symmat[j1][j2] = acc;
          symmat[j2][j1] = acc;
        }
      }
    });
  }).wait();
}

void covariance_sycl_sw(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    kernel_covariance_sycl_sw(data_buf, symmat_buf, mean_buf, Q);
  }
}

void bench_covariance_sycl_sw(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_covariance_sycl_sw(data_buf, symmat_buf, mean_buf, Q);

    TIMEIT({
      kernel_covariance_sycl_sw(data_buf, symmat_buf, mean_buf, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
