#include "jacobi-2d.h"

namespace {

#ifndef JACOBI_A10_TILE_H
#define JACOBI_A10_TILE_H 2
#endif

#ifndef JACOBI_A10_TILE_W
#define JACOBI_A10_TILE_W 64
#endif

#ifndef JACOBI_A10_COPY_WG
#define JACOBI_A10_COPY_WG 512
#endif

constexpr size_t TILE_H = JACOBI_A10_TILE_H;
constexpr size_t TILE_W = JACOBI_A10_TILE_W;
constexpr size_t COPY_WG = JACOBI_A10_COPY_WG;

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

void launch_jacobi_step(buffer<DATA_TYPE, 2> &src_buf, buffer<DATA_TYPE, 2> &dst_buf, queue &Q)
{
  const range<2> local_range{TILE_H, TILE_W};
  const range<2> global_range{
      round_up(MATRIX_SIZE_H - 2, TILE_H),
      round_up(MATRIX_SIZE_W - 2, TILE_W)};

  Q.submit([&](handler &h) {
    auto src = src_buf.get_access<access::mode::read>(h);
    auto dst = dst_buf.get_access<access::mode::write>(h);
    local_accessor<DATA_TYPE, 2> tile(range<2>(TILE_H + 2, TILE_W + 2), h);

    h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
      const size_t li = item.get_local_id(0);
      const size_t lj = item.get_local_id(1);
      const size_t group_i0 = item.get_group(0) * TILE_H + 1;
      const size_t group_j0 = item.get_group(1) * TILE_W + 1;
      const size_t i = item.get_global_id(0) + 1;
      const size_t j = item.get_global_id(1) + 1;
      const size_t active_h = (group_i0 < MATRIX_SIZE_H - 1) ? sycl::min<size_t>(TILE_H, MATRIX_SIZE_H - 1 - group_i0) : 0;
      const size_t active_w = (group_j0 < MATRIX_SIZE_W - 1) ? sycl::min<size_t>(TILE_W, MATRIX_SIZE_W - 1 - group_j0) : 0;

      const bool inside = i < MATRIX_SIZE_H - 1 && j < MATRIX_SIZE_W - 1;

      tile[li + 1][lj + 1] = inside ? src[i][j] : DATA_TYPE{0};
      if (inside && li == 0)
        tile[0][lj + 1] = src[i - 1][j];
      if (inside && active_h > 0 && li == active_h - 1)
        tile[active_h + 1][lj + 1] = (i + 1 < MATRIX_SIZE_H) ? src[i + 1][j] : DATA_TYPE{0};
      if (inside && lj == 0)
        tile[li + 1][0] = src[i][j - 1];
      if (inside && active_w > 0 && lj == active_w - 1)
        tile[li + 1][active_w + 1] = (j + 1 < MATRIX_SIZE_W) ? src[i][j + 1] : DATA_TYPE{0};

      item.barrier(access::fence_space::local_space);

      if (inside) {
        dst[i][j] = DATA_TYPE{0.201f} *
                    (tile[li + 1][lj + 1] + tile[li + 1][lj] + tile[li + 1][lj + 2] +
                     tile[li + 2][lj + 1] + tile[li][lj + 1]);
      }
    });
  }).wait();
}

void launch_copy_step(buffer<DATA_TYPE, 2> &dst_buf, buffer<DATA_TYPE, 2> &src_buf, queue &Q)
{
  const range<2> local_range{1, COPY_WG};
  const range<2> global_range{
      MATRIX_SIZE_H - 2,
      round_up(MATRIX_SIZE_W - 2, COPY_WG)};

  Q.submit([&](handler &h) {
    auto dst = dst_buf.get_access<access::mode::write>(h);
    auto src = src_buf.get_access<access::mode::read>(h);
    h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
      const size_t i = item.get_global_id(0) + 1;
      const size_t j = item.get_global_id(1) + 1;
      if (i < MATRIX_SIZE_H - 1 && j < MATRIX_SIZE_W - 1)
        dst[i][j] = src[i][j];
    });
  }).wait();
}

} // namespace

void kernel_jacobi_2d_sycl_a10(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B, queue &Q)
{
  for (int t = 0; t < TSTEPS; t++) {
    launch_jacobi_step(buf_A, buf_B, Q);
    launch_copy_step(buf_A, buf_B, Q);
  }
}

void jacobi_2d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    kernel_jacobi_2d_sycl_a10(buf_A, buf_B, Q);
  }
}

void bench_jacobi_2d_sycl_a10(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q;
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_jacobi_2d_sycl_a10(buf_A, buf_B, Q);

    TIMEIT({
      kernel_jacobi_2d_sycl_a10(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-a10");
  }
}
