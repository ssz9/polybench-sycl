#include "covariance.h"

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

void kernel_covariance_sycl_dcu(buffer<DATA_TYPE, 2> &data_buf, buffer<DATA_TYPE, 2> &symmat_buf,
                                buffer<DATA_TYPE, 1> &mean_buf, queue &Q)
{
  const DATA_TYPE float_n = DATA_TYPE{1.2f};

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
    const range<2> local_range{16, 16};
    const range<2> global_range{round_up(_PB_N, 16), round_up(_PB_M, 16)};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read_write>(h);
      auto mean = mean_buf.get_access<access::mode::read>(h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);
        if (i < _PB_N && j < _PB_M)
          data[i][j] -= mean[j];
      });
    }).wait();
  }

  {
    const range<2> local_range{TILE_M, TILE_N};
    const range<2> global_range{round_up(_PB_M, TILE_M), round_up(_PB_M, TILE_N)};
    Q.submit([&](handler &h) {
      auto data = data_buf.get_access<access::mode::read>(h);
      auto symmat = symmat_buf.get_access<access::mode::discard_write>(h);
      local_accessor<DATA_TYPE, 2> tile_i(range<2>(TILE_K, TILE_M), h);
      local_accessor<DATA_TYPE, 2> tile_j(range<2>(TILE_K, TILE_N), h);

      h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
        const size_t li = item.get_local_id(0);
        const size_t lj = item.get_local_id(1);
        const size_t j1 = item.get_group(0) * TILE_M + li;
        const size_t j2 = item.get_group(1) * TILE_N + lj;

        DATA_TYPE sum = DATA_TYPE{0};
        for (size_t k0 = 0; k0 < _PB_N; k0 += TILE_K) {
          const size_t row_i = k0 + lj;
          const size_t row_j = k0 + li;

          tile_i[lj][li] = (j1 < _PB_M && row_i < _PB_N) ? data[row_i][j1] : DATA_TYPE{0};
          tile_j[li][lj] = (j2 < _PB_M && row_j < _PB_N) ? data[row_j][j2] : DATA_TYPE{0};
          item.barrier(access::fence_space::local_space);

          #pragma unroll
          for (size_t kk = 0; kk < TILE_K; kk++)
            sum += tile_i[kk][li] * tile_j[kk][lj];

          item.barrier(access::fence_space::local_space);
        }

        if (j1 < _PB_M && j2 < _PB_M)
          symmat[j1][j2] = sum;
      });
    }).wait();
  }
}

void covariance_sycl_dcu(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));
    kernel_covariance_sycl_dcu(data_buf, symmat_buf, mean_buf, Q);
  }
}

void bench_covariance_sycl_dcu(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 2> data_buf(data, range<2>(_PB_N, _PB_M));
    buffer<DATA_TYPE, 2> symmat_buf(symmat, range<2>(_PB_M, _PB_M));
    buffer<DATA_TYPE, 1> mean_buf(mean, range<1>(_PB_M));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_covariance_sycl_dcu(data_buf, symmat_buf, mean_buf, Q);

    TIMEIT({
      kernel_covariance_sycl_dcu(data_buf, symmat_buf, mean_buf, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
