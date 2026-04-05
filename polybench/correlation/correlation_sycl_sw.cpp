#include "correlation.h"
#include "sw_sycl_utils.h"

static void kernel_correlation_sycl_sw(buffer<DATA_TYPE, 2> &data_buf, buffer<DATA_TYPE, 2> &corr_buf,
                                       buffer<DATA_TYPE, 1> &mean_buf, buffer<DATA_TYPE, 1> &stddev_buf, queue &Q)
{
  const DATA_TYPE eps = 0.1;
  const DATA_TYPE float_n = (DATA_TYPE)_PB_N;

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::write>(h);
    h.parallel_for<class CorrelationSwMeanKernel>(sw_cpe_range(), [=](id<1> worker) {
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
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    auto stddev = stddev_buf.get_access<access::mode::write>(h);
    h.parallel_for<class CorrelationSwStddevKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t col_begin = sw_block_begin(_PB_M, worker_id);
      const size_t col_end = sw_block_end(_PB_M, worker_id);
      for (size_t j = col_begin; j < col_end; ++j) {
        DATA_TYPE acc = 0.0;
        for (int i = 0; i < _PB_N; ++i) {
          const DATA_TYPE diff = data[i][j] - mean[j];
          acc += diff * diff;
        }
        DATA_TYPE value = SQRT_FUN(acc / float_n);
        stddev[j] = value <= eps ? DATA_TYPE(1.0) : value;
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read_write>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    auto stddev = stddev_buf.get_access<access::mode::read>(h);
    h.parallel_for<class CorrelationSwNormalizeKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t row_begin = sw_block_begin(_PB_N, worker_id);
      const size_t row_end = sw_block_end(_PB_N, worker_id);
      const DATA_TYPE scale = SQRT_FUN(float_n);
      for (size_t i = row_begin; i < row_end; ++i)
        for (int j = 0; j < _PB_M; ++j)
          data[i][j] = (data[i][j] - mean[j]) / (scale * stddev[j]);
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto corr = corr_buf.get_access<access::mode::write>(h);
    h.parallel_for<class CorrelationSwCorrKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t row_begin = sw_block_begin(_PB_M, worker_id);
      const size_t row_end = sw_block_end(_PB_M, worker_id);
      for (size_t i = row_begin; i < row_end; ++i) {
        corr[i][i] = 1.0;
        for (size_t j = i + 1; j < _PB_M; ++j) {
          DATA_TYPE acc = 0.0;
          for (int k = 0; k < _PB_N; ++k)
            acc += data[k][i] * data[k][j];
          corr[i][j] = acc;
          corr[j][i] = acc;
        }
      }
    });
  }).wait();
}

void correlation_sycl_sw(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));
    kernel_correlation_sycl_sw(data_buf, corr_buf, mean_buf, stddev_buf, Q);
  }
}

void bench_correlation_sycl_sw(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_correlation_sycl_sw(data_buf, corr_buf, mean_buf, stddev_buf, Q);

    TIMEIT({
      kernel_correlation_sycl_sw(data_buf, corr_buf, mean_buf, stddev_buf, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
