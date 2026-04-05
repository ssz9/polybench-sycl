#include "jacobi-2d.h"

namespace {

constexpr size_t JACOBI_TILE_H = 8;
constexpr size_t JACOBI_TILE_W = 32;

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

void launch_jacobi_step(buffer<DATA_TYPE, 2> &src_buf, buffer<DATA_TYPE, 2> &dst_buf, queue &Q)
{
  const range<2> local_range{JACOBI_TILE_H, JACOBI_TILE_W};
  const range<2> global_range{round_up(MATRIX_SIZE_H - 2, JACOBI_TILE_H), round_up(MATRIX_SIZE_W - 2, JACOBI_TILE_W)};

  Q.submit([&](handler &h) {
    auto src = src_buf.get_access<access::mode::read>(h);
    auto dst = dst_buf.get_access<access::mode::write>(h);
    h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
      const size_t ii = item.get_global_id(0);
      const size_t jj = item.get_global_id(1);
      if (ii < MATRIX_SIZE_H - 2 && jj < MATRIX_SIZE_W - 2) {
        const size_t i = ii + 1;
        const size_t j = jj + 1;
        dst[i][j] = DATA_TYPE{0.201f} *
                    (src[i][j] + src[i][j - 1] + src[i][j + 1] + src[i + 1][j] + src[i - 1][j]);
      }
    });
  }).wait();
}

void launch_copy_step(buffer<DATA_TYPE, 2> &dst_buf, buffer<DATA_TYPE, 2> &src_buf, queue &Q)
{
  const range<2> local_range{JACOBI_TILE_H, JACOBI_TILE_W};
  const range<2> global_range{round_up(MATRIX_SIZE_H - 2, JACOBI_TILE_H), round_up(MATRIX_SIZE_W - 2, JACOBI_TILE_W)};

  Q.submit([&](handler &h) {
    auto dst = dst_buf.get_access<access::mode::write>(h);
    auto src = src_buf.get_access<access::mode::read>(h);
    h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
      const size_t ii = item.get_global_id(0);
      const size_t jj = item.get_global_id(1);
      if (ii < MATRIX_SIZE_H - 2 && jj < MATRIX_SIZE_W - 2) {
        const size_t i = ii + 1;
        const size_t j = jj + 1;
        dst[i][j] = src[i][j];
      }
    });
  }).wait();
}

} // namespace

void kernel_jacobi_2d_sycl_dcu(buffer<DATA_TYPE, 2> &buf_A, buffer<DATA_TYPE, 2> &buf_B, queue &Q)
{
  for (int t = 0; t < TSTEPS; t++) {
    launch_jacobi_step(buf_A, buf_B, Q);
    launch_copy_step(buf_A, buf_B, Q);
  }
}

void jacobi_2d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    kernel_jacobi_2d_sycl_dcu(buf_A, buf_B, Q);
  }
}

void bench_jacobi_2d_sycl_dcu(DATA_TYPE *A, DATA_TYPE *B)
{
  queue Q{gpu_selector_v};
  {
    buffer<DATA_TYPE, 2> buf_A(A, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));
    buffer<DATA_TYPE, 2> buf_B(B, range<2>(MATRIX_SIZE_H, MATRIX_SIZE_W));

    for (int i = 0; i < WARMUP_REPS; i++)
      kernel_jacobi_2d_sycl_dcu(buf_A, buf_B, Q);

    TIMEIT({
      kernel_jacobi_2d_sycl_dcu(buf_A, buf_B, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
