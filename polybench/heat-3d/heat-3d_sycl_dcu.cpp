#include "heat-3d.h"

namespace {

constexpr size_t TILE_X = 4;
constexpr size_t TILE_Y = 4;
constexpr size_t TILE_Z = 8;

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

void launch_heat_step(buffer<DATA_TYPE, 3> &src_buf, buffer<DATA_TYPE, 3> &dst_buf, queue &Q)
{
  const range<3> local_range{TILE_X, TILE_Y, TILE_Z};
  const range<3> global_range{
      round_up(HEAT_3D_N - 2, TILE_X),
      round_up(HEAT_3D_N - 2, TILE_Y),
      round_up(HEAT_3D_N - 2, TILE_Z)};

  Q.submit([&](handler &h) {
    auto src = src_buf.get_access<access::mode::read>(h);
    auto dst = dst_buf.get_access<access::mode::write>(h);

    h.parallel_for(nd_range<3>(global_range, local_range), [=](nd_item<3> item) {
      const size_t ii = item.get_global_id(0);
      const size_t jj = item.get_global_id(1);
      const size_t kk = item.get_global_id(2);
      if (ii < HEAT_3D_N - 2 && jj < HEAT_3D_N - 2 && kk < HEAT_3D_N - 2) {
        const size_t i = ii + 1;
        const size_t j = jj + 1;
        const size_t k = kk + 1;
        const DATA_TYPE center = src[i][j][k];
        dst[i][j][k] =
            SCALAR_VAL(0.125) * (src[i + 1][j][k] - SCALAR_VAL(2.0) * center + src[i - 1][j][k]) +
            SCALAR_VAL(0.125) * (src[i][j + 1][k] - SCALAR_VAL(2.0) * center + src[i][j - 1][k]) +
            SCALAR_VAL(0.125) * (src[i][j][k + 1] - SCALAR_VAL(2.0) * center + src[i][j][k - 1]) +
            center;
      }
    });
  }).wait();
}

} // namespace

void kernel_heat_3d_sycl_dcu(buffer<DATA_TYPE, 3> &buf_A, buffer<DATA_TYPE, 3> &buf_B, queue &Q)
{
  for (int t = 1; t <= TSTEPS; t++) {
    launch_heat_step(buf_A, buf_B, Q);
    launch_heat_step(buf_B, buf_A, Q);
  }
}

void heat_3d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    kernel_heat_3d_sycl_dcu(buf_A, buf_B, Q);
  }
}

void bench_heat_3d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_heat_3d_sycl_dcu(buf_A, buf_B, Q);

    TIMEIT({
      kernel_heat_3d_sycl_dcu(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
