#include "heat-3d.h"

namespace {

#ifndef HEAT3D_A10_TILE_X
#define HEAT3D_A10_TILE_X 1
#endif

#ifndef HEAT3D_A10_TILE_Y
#define HEAT3D_A10_TILE_Y 16
#endif

#ifndef HEAT3D_A10_TILE_Z
#define HEAT3D_A10_TILE_Z 16
#endif

constexpr size_t TILE_X = HEAT3D_A10_TILE_X;
constexpr size_t TILE_Y = HEAT3D_A10_TILE_Y;
constexpr size_t TILE_Z = HEAT3D_A10_TILE_Z;

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
    local_accessor<DATA_TYPE, 3> tile(range<3>(TILE_X + 2, TILE_Y + 2, TILE_Z + 2), h);

    h.parallel_for(nd_range<3>(global_range, local_range), [=](nd_item<3> item) {
      const size_t li = item.get_local_id(0);
      const size_t lj = item.get_local_id(1);
      const size_t lk = item.get_local_id(2);
      const size_t group_i0 = item.get_group(0) * TILE_X + 1;
      const size_t group_j0 = item.get_group(1) * TILE_Y + 1;
      const size_t group_k0 = item.get_group(2) * TILE_Z + 1;
      const size_t i = item.get_global_id(0) + 1;
      const size_t j = item.get_global_id(1) + 1;
      const size_t k = item.get_global_id(2) + 1;
      const size_t active_x = (group_i0 < HEAT_3D_N - 1) ? sycl::min<size_t>(TILE_X, HEAT_3D_N - 1 - group_i0) : 0;
      const size_t active_y = (group_j0 < HEAT_3D_N - 1) ? sycl::min<size_t>(TILE_Y, HEAT_3D_N - 1 - group_j0) : 0;
      const size_t active_z = (group_k0 < HEAT_3D_N - 1) ? sycl::min<size_t>(TILE_Z, HEAT_3D_N - 1 - group_k0) : 0;

      const bool inside = i < HEAT_3D_N - 1 && j < HEAT_3D_N - 1 && k < HEAT_3D_N - 1;

      tile[li + 1][lj + 1][lk + 1] = inside ? src[i][j][k] : DATA_TYPE{0};
      if (inside && li == 0)
        tile[0][lj + 1][lk + 1] = src[i - 1][j][k];
      if (inside && active_x > 0 && li == active_x - 1)
        tile[active_x + 1][lj + 1][lk + 1] = (i + 1 < HEAT_3D_N) ? src[i + 1][j][k] : DATA_TYPE{0};
      if (inside && lj == 0)
        tile[li + 1][0][lk + 1] = src[i][j - 1][k];
      if (inside && active_y > 0 && lj == active_y - 1)
        tile[li + 1][active_y + 1][lk + 1] = (j + 1 < HEAT_3D_N) ? src[i][j + 1][k] : DATA_TYPE{0};
      if (inside && lk == 0)
        tile[li + 1][lj + 1][0] = src[i][j][k - 1];
      if (inside && active_z > 0 && lk == active_z - 1)
        tile[li + 1][lj + 1][active_z + 1] = (k + 1 < HEAT_3D_N) ? src[i][j][k + 1] : DATA_TYPE{0};

      item.barrier(access::fence_space::local_space);

      if (inside) {
        const DATA_TYPE center = tile[li + 1][lj + 1][lk + 1];
        dst[i][j][k] =
            SCALAR_VAL(0.125) * (tile[li + 2][lj + 1][lk + 1] - SCALAR_VAL(2.0) * center + tile[li][lj + 1][lk + 1]) +
            SCALAR_VAL(0.125) * (tile[li + 1][lj + 2][lk + 1] - SCALAR_VAL(2.0) * center + tile[li + 1][lj][lk + 1]) +
            SCALAR_VAL(0.125) * (tile[li + 1][lj + 1][lk + 2] - SCALAR_VAL(2.0) * center + tile[li + 1][lj + 1][lk]) +
            center;
      }
    });
  }).wait();
}

} // namespace

void kernel_heat_3d_sycl_a10(buffer<DATA_TYPE, 3> &buf_A, buffer<DATA_TYPE, 3> &buf_B, queue &Q)
{
  for (int t = 1; t <= TSTEPS; t++) {
    launch_heat_step(buf_A, buf_B, Q);
    launch_heat_step(buf_B, buf_A, Q);
  }
}

void heat_3d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    kernel_heat_3d_sycl_a10(buf_A, buf_B, Q);
  }
}

void bench_heat_3d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 3> buf_A(A, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));
    buffer<DATA_TYPE, 3> buf_B(B, range<3>(HEAT_3D_N, HEAT_3D_N, HEAT_3D_N));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_heat_3d_sycl_a10(buf_A, buf_B, Q);

    TIMEIT({
      kernel_heat_3d_sycl_a10(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-a10");
  }
}
