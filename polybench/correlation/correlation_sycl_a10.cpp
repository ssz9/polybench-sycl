#include "correlation.h"

namespace {

constexpr size_t REDUCE_WG = 256;
constexpr size_t TILE_M = 16;
constexpr size_t TILE_N = 16;
constexpr size_t TILE_K = 16;

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

} // namespace

void kernel_correlation_sycl_a10(buffer<DATA_TYPE, 2> &data_buf, buffer<DATA_TYPE, 2> &corr_buf,
                                 buffer<DATA_TYPE, 1> &mean_buf, buffer<DATA_TYPE, 1> &stddev_buf,
                                 queue &Q)
{
  constexpr DATA_TYPE eps = DATA_TYPE{0.1f};
  const DATA_TYPE float_n = static_cast<DATA_TYPE>(_PB_N);
  const DATA_TYPE norm = SQRT_FUN(float_n);

  {
    const range<2> local_range{1, REDUCE_WG};
    const range<2> global_range{_PB_M, REDUCE_WG};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read>(h);
      auto mean = mean_buf.get_access<access::mode::discard_write>(h);
      local_accessor<DATA_TYPE, 1> partial(range<1>(REDUCE_WG), h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t j = item.get_group(0);
        const size_t lid = item.get_local_id(1);

        DATA_TYPE sum = DATA_TYPE{0};
        for (size_t i = lid; i < _PB_N; i += REDUCE_WG)
          sum += data[i][j];

        partial[lid] = sum;
        item.barrier(access::fence_space::local_space);

        for (size_t offset = REDUCE_WG / 2; offset > 0; offset >>= 1) {
          if (lid < offset)
            partial[lid] += partial[lid + offset];
          item.barrier(access::fence_space::local_space);
        }

        if (lid == 0)
          mean[j] = partial[0] / float_n;
      });
    }).wait();
  }

  {
    const range<2> local_range{1, REDUCE_WG};
    const range<2> global_range{_PB_M, REDUCE_WG};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read>(h);
      auto mean = mean_buf.get_access<access::mode::read>(h);
      auto stddev = stddev_buf.get_access<access::mode::discard_write>(h);
      local_accessor<DATA_TYPE, 1> partial(range<1>(REDUCE_WG), h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t j = item.get_group(0);
        const size_t lid = item.get_local_id(1);

        DATA_TYPE sum = DATA_TYPE{0};
        for (size_t i = lid; i < _PB_N; i += REDUCE_WG) {
          const DATA_TYPE diff = data[i][j] - mean[j];
          sum += diff * diff;
        }

        partial[lid] = sum;
        item.barrier(access::fence_space::local_space);

        for (size_t offset = REDUCE_WG / 2; offset > 0; offset >>= 1) {
          if (lid < offset)
            partial[lid] += partial[lid + offset];
          item.barrier(access::fence_space::local_space);
        }

        if (lid == 0) {
          DATA_TYPE value = SQRT_FUN(partial[0] / float_n);
          stddev[j] = value <= eps ? DATA_TYPE{1} : value;
        }
      });
    }).wait();
  }

  {
    const range<2> local_range{TILE_N, TILE_M};
    const range<2> global_range{round_up(_PB_N, TILE_N), round_up(_PB_M, TILE_M)};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read_write>(h);
      auto mean = mean_buf.get_access<access::mode::read>(h);
      auto stddev = stddev_buf.get_access<access::mode::read>(h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);
        if (i < _PB_N && j < _PB_M) {
          DATA_TYPE value = data[i][j] - mean[j];
          data[i][j] = value / (norm * stddev[j]);
        }
      });
    }).wait();
  }

  {
    const range<2> local_range{TILE_M, TILE_N};
    const range<2> global_range{round_up(_PB_M, TILE_M), round_up(_PB_M, TILE_N)};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read>(h);
      auto corr = corr_buf.get_access<access::mode::discard_write>(h);
      local_accessor<DATA_TYPE, 2> tile_i(range<2>(TILE_K, TILE_M), h);
      local_accessor<DATA_TYPE, 2> tile_j(range<2>(TILE_K, TILE_N), h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t li = item.get_local_id(0);
        const size_t lj = item.get_local_id(1);
        const size_t col_i = item.get_group(0) * TILE_M + li;
        const size_t col_j = item.get_group(1) * TILE_N + lj;

        DATA_TYPE sum = DATA_TYPE{0};
        for (size_t k0 = 0; k0 < _PB_N; k0 += TILE_K) {
          const size_t row_i = k0 + lj;
          const size_t row_j = k0 + li;

          tile_i[lj][li] = (col_i < _PB_M && row_i < _PB_N) ? data[row_i][col_i] : DATA_TYPE{0};
          tile_j[li][lj] = (col_j < _PB_M && row_j < _PB_N) ? data[row_j][col_j] : DATA_TYPE{0};
          item.barrier(access::fence_space::local_space);

          #pragma unroll
          for (size_t kk = 0; kk < TILE_K; kk++)
            sum += tile_i[kk][li] * tile_j[kk][lj];

          item.barrier(access::fence_space::local_space);
        }

        if (col_i < _PB_M && col_j < _PB_M)
          corr[col_i][col_j] = (col_i == col_j) ? DATA_TYPE{1} : sum;
      });
    }).wait();
  }
}

void correlation_sycl_a10(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));
    kernel_correlation_sycl_a10(data_buf, corr_buf, mean_buf, stddev_buf, Q);
  }
}

void bench_correlation_sycl_a10(DATA_TYPE *data, DATA_TYPE *corr, DATA_TYPE *mean, DATA_TYPE *stddev)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> corr_buf(corr, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    buffer<DATA_TYPE, 1> stddev_buf(stddev, range<1>(_PB_M));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_correlation_sycl_a10(data_buf, corr_buf, mean_buf, stddev_buf, Q);

    TIMEIT({
      kernel_correlation_sycl_a10(data_buf, corr_buf, mean_buf, stddev_buf, Q);
    }, BENCH_REPS, "\n", "sycl-a10");
  }
}
