#include "covariance.h"

void kernel_covariance_sycl(buffer<DATA_TYPE, 2> &data_buf, buffer<DATA_TYPE, 2> &symmat_buf,
                            buffer<DATA_TYPE, 1> &mean_buf, queue &Q)
{
  DATA_TYPE float_n = 1.2;

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto mean = mean_buf.get_access<access::mode::write>(h);
    h.parallel_for(range<1>(_PB_M), [=](id<1> j) {
      mean[j] = 0.0;
      for (int i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read_write>(h);
    auto mean = mean_buf.get_access<access::mode::read>(h);
    h.parallel_for(range<2>(_PB_N, _PB_M), [=](id<2> idx) {
      data[idx] -= mean[idx[1]];
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto data = data_buf.get_access<access::mode::read>(h);
    auto symmat = symmat_buf.get_access<access::mode::write>(h);
    h.parallel_for(range<2>(_PB_M, _PB_M), [=](id<2> idx) {
      int j1 = idx[0];
      int j2 = idx[1];
      if (j1 <= j2) {
        symmat[j1][j2] = 0.0;
        for (int i = 0; i < _PB_N; i++)
          symmat[j1][j2] += data[i][j1] * data[i][j2];
        symmat[j2][j1] = symmat[j1][j2];
      }
    });
  }).wait();
}

void covariance_sycl(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    kernel_covariance_sycl(data_buf, symmat_buf, mean_buf, Q);
  }
}

void bench_covariance_sycl(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_covariance_sycl(data_buf, symmat_buf, mean_buf, Q);

    TIMEIT({
      kernel_covariance_sycl(data_buf, symmat_buf, mean_buf, Q);
    }, BENCH_REPS, "\n", "sycl-naive");
  }
}
