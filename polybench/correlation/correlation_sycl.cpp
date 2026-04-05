#include "header.h"


void kernel_correlation_sycl(buffer<DATA_TYPE,2> &data_buf, buffer<DATA_TYPE,2> &corr_buf, buffer<DATA_TYPE,1> &mean_buf, buffer<DATA_TYPE,1> &stddev_buf, queue &Q)
{
  DATA_TYPE eps = 0.1;
  DATA_TYPE float_n = (DATA_TYPE)_PB_N;
  Q.submit([&](handler &h){
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::write>(h);
    h.parallel_for({_PB_M}, [=](id<1> j){
      mean[j] = 0.0;
      for (int i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    });
  }).wait();

  Q.submit([&](handler &h){
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    auto stddev = stddev_buf.get_access<access::mode::write>(h);
    h.parallel_for({_PB_M}, [=](id<1> j){
      stddev[j] = 0.0;
      for (int i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle near-zero std. dev. values, which below would cause a zero- divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    });
  }).wait();

  Q.submit([&](handler &h){
    auto data = data_buf.get_access<access::mode::read_write>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    auto stddev = stddev_buf.get_access<access::mode::read>(h);
    h.parallel_for({_PB_N, _PB_M}, [=](id<2> idx){
      int i = idx[0];
      int j = idx[1];
      data[i][j] -= mean[j];
      data[i][j] /= SQRT_FUN(float_n) * stddev[j];
    });
  }).wait();
 
  Q.submit([&](handler &h){
    auto data = data_buf.get_access<access::mode::read>(h);
    auto corr = corr_buf.get_access<access::mode::write>(h);
    h.parallel_for({_PB_M, _PB_M}, [=](id<2> idx){
      int i = idx[0];
      int j = idx[1];
      if (i == j) {
        corr[i][j] = 1.0;
      } else if (i < j) {
        corr[i][j] = 0.0;
        for (int k = 0; k < _PB_N; k++)
          corr[i][j] += (data[k][i] * data[k][j]);
        corr[j][i] = corr[i][j];
      }
    });
  }).wait();
}

void correlation_sycl(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));
    kernel_correlation_sycl(data_buf, corr_buf, mean_buf, stddev_buf, Q);
  }
}

void bench_correlation_sycl(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));

    // warmup
    for (int i = 0; i < WARMUP_REPS; i++) {
      kernel_correlation_sycl(data_buf, corr_buf, mean_buf, stddev_buf, Q);
    }
    // timing
    TIMEIT({
      kernel_correlation_sycl(data_buf, corr_buf, mean_buf, stddev_buf, Q);
    }, 10, "\n", "sycl-naive");
  }
}
