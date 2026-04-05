#include "fastwavesuv.h"
#include "sw_sycl_utils.h"

static void kernel_fastwavesuv_sycl_sw(
    buffer<DATA_TYPE, 3> &uout_buf, buffer<DATA_TYPE, 3> &vout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 3> &utens_buf, buffer<DATA_TYPE, 3> &vtens_buf, buffer<DATA_TYPE, 3> &wgtfac_buf, buffer<DATA_TYPE, 3> &ppuv_buf,
    buffer<DATA_TYPE, 3> &hhl_buf, buffer<DATA_TYPE, 3> &rho_buf, buffer<DATA_TYPE, 1> &fx_buf, buffer<DATA_TYPE, 3> &ppgk_buf,
    buffer<DATA_TYPE, 3> &ppgc_buf, buffer<DATA_TYPE, 3> &ppgu_buf, buffer<DATA_TYPE, 3> &ppgv_buf, DATA_TYPE edadlat, DATA_TYPE dt, queue &Q)
{
  Q.submit([&](handler &h) {
    auto wgtfac = wgtfac_buf.get_access<access::mode::read>(h);
    auto ppuv = ppuv_buf.get_access<access::mode::read>(h);
    auto ppgk = ppgk_buf.get_access<access::mode::write>(h);
    h.parallel_for<class FastwavesuvSwPpgkKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)(DOMAIN_SIZE + 1) * (DOMAIN_SIZE + 1), worker_id);
      const size_t end = sw_block_end((size_t)(DOMAIN_SIZE + 1) * (DOMAIN_SIZE + 1), worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / (DOMAIN_SIZE + 1);
        const size_t j = ij % (DOMAIN_SIZE + 1);
        for (size_t k = 0; k < DOMAIN_HEIGHT + 1; ++k)
          ppgk[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH] =
              wgtfac[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH] * ppuv[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH] +
              (DATA_TYPE(1.0) - wgtfac[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH]) * ppuv[i + HALO_WIDTH][j + HALO_WIDTH][k - 1 + HALO_WIDTH];
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto ppgk = ppgk_buf.get_access<access::mode::read>(h);
    auto ppgc = ppgc_buf.get_access<access::mode::write>(h);
    h.parallel_for<class FastwavesuvSwPpgcKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)(DOMAIN_SIZE + 1) * (DOMAIN_SIZE + 1), worker_id);
      const size_t end = sw_block_end((size_t)(DOMAIN_SIZE + 1) * (DOMAIN_SIZE + 1), worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / (DOMAIN_SIZE + 1);
        const size_t j = ij % (DOMAIN_SIZE + 1);
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k)
          ppgc[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH] =
              ppgk[i + HALO_WIDTH][j + HALO_WIDTH][k + 1 + HALO_WIDTH] - ppgk[i + HALO_WIDTH][j + HALO_WIDTH][k + HALO_WIDTH];
      }
    });
  }).wait();

  Q.submit([&](handler &h) {
    auto ppuv = ppuv_buf.get_access<access::mode::read>(h);
    auto ppgc = ppgc_buf.get_access<access::mode::read>(h);
    auto hhl = hhl_buf.get_access<access::mode::read>(h);
    auto ppgu = ppgu_buf.get_access<access::mode::write>(h);
    auto ppgv = ppgv_buf.get_access<access::mode::write>(h);
    auto fx = fx_buf.get_access<access::mode::read>(h);
    auto rho = rho_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto utens = utens_buf.get_access<access::mode::read>(h);
    auto vtens = vtens_buf.get_access<access::mode::read>(h);
    auto uout = uout_buf.get_access<access::mode::write>(h);
    auto vout = vout_buf.get_access<access::mode::write>(h);
    h.parallel_for<class FastwavesuvSwUpdateKernel>(sw_cpe_range(), [=](id<1> worker) {
      const size_t worker_id = sw_worker_id(worker);
      const size_t begin = sw_block_begin((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      const size_t end = sw_block_end((size_t)DOMAIN_SIZE * DOMAIN_SIZE, worker_id);
      for (size_t ij = begin; ij < end; ++ij) {
        const size_t i = ij / DOMAIN_SIZE;
        const size_t j = ij % DOMAIN_SIZE;
        const size_t ii = i + HALO_WIDTH;
        const size_t jj = j + HALO_WIDTH;
        for (size_t k = 0; k < DOMAIN_HEIGHT; ++k) {
          const size_t kk = k + HALO_WIDTH;
          ppgu[ii][jj][kk] =
              (ppuv[ii + 1][jj][kk] - ppuv[ii][jj][kk]) +
              (ppgc[ii + 1][jj][kk] + ppgc[ii][jj][kk]) * DATA_TYPE(0.5) *
                  ((hhl[ii][jj][kk + 1] + hhl[ii][jj][kk]) - (hhl[ii + 1][jj][kk + 1] + hhl[ii + 1][jj][kk])) /
                  ((hhl[ii][jj][kk + 1] - hhl[ii][jj][kk]) + (hhl[ii + 1][jj][kk + 1] - hhl[ii + 1][jj][kk]));
          ppgv[ii][jj][kk] =
              (ppuv[ii][jj + 1][kk] - ppuv[ii][jj][kk]) +
              (ppgc[ii][jj + 1][kk] + ppgc[ii][jj][kk]) * DATA_TYPE(0.5) *
                  ((hhl[ii][jj][kk + 1] + hhl[ii][jj][kk]) - (hhl[ii][jj + 1][kk + 1] + hhl[ii][jj + 1][kk])) /
                  ((hhl[ii][jj][kk + 1] - hhl[ii][jj][kk]) + (hhl[ii][jj + 1][kk + 1] - hhl[ii][jj + 1][kk]));
          uout[ii][jj][kk] = uin[ii][jj][kk] + dt * (utens[ii][jj][kk] - ppgu[ii][jj][kk] * (DATA_TYPE(2.0) * fx[jj] / (rho[ii + 1][jj][kk] + rho[ii][jj][kk])));
          vout[ii][jj][kk] = vin[ii][jj][kk] + dt * (vtens[ii][jj][kk] - ppgv[ii][jj][kk] * (DATA_TYPE(2.0) * edadlat / (rho[ii][jj + 1][kk] + rho[ii][jj][kk])));
        }
      }
    });
  }).wait();
}

void fastwavesuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt)
{
  queue Q;
  {
    range<3> r3(DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_HEIGHT + 2 * HALO_WIDTH);
    range<1> r1(DOMAIN_SIZE + 2 * HALO_WIDTH);
    buffer<DATA_TYPE, 3> uout_buf(uout, r3);
    buffer<DATA_TYPE, 3> vout_buf(vout, r3);
    buffer<DATA_TYPE, 3> uin_buf(uin, r3);
    buffer<DATA_TYPE, 3> vin_buf(vin, r3);
    buffer<DATA_TYPE, 3> utens_buf(utens, r3);
    buffer<DATA_TYPE, 3> vtens_buf(vtens, r3);
    buffer<DATA_TYPE, 3> wgtfac_buf(wgtfac, r3);
    buffer<DATA_TYPE, 3> ppuv_buf(ppuv, r3);
    buffer<DATA_TYPE, 3> hhl_buf(hhl, r3);
    buffer<DATA_TYPE, 3> rho_buf(rho, r3);
    buffer<DATA_TYPE, 1> fx_buf(fx, r1);
    buffer<DATA_TYPE, 3> ppgk_buf(ppgk, r3);
    buffer<DATA_TYPE, 3> ppgc_buf(ppgc, r3);
    buffer<DATA_TYPE, 3> ppgu_buf(ppgu, r3);
    buffer<DATA_TYPE, 3> ppgv_buf(ppgv, r3);
    kernel_fastwavesuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                               ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);
  }
}

void bench_fastwavesuv_sycl_sw(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt)
{
  queue Q;
  {
    range<3> r3(DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_SIZE + 2 * HALO_WIDTH, DOMAIN_HEIGHT + 2 * HALO_WIDTH);
    range<1> r1(DOMAIN_SIZE + 2 * HALO_WIDTH);
    buffer<DATA_TYPE, 3> uout_buf(uout, r3);
    buffer<DATA_TYPE, 3> vout_buf(vout, r3);
    buffer<DATA_TYPE, 3> uin_buf(uin, r3);
    buffer<DATA_TYPE, 3> vin_buf(vin, r3);
    buffer<DATA_TYPE, 3> utens_buf(utens, r3);
    buffer<DATA_TYPE, 3> vtens_buf(vtens, r3);
    buffer<DATA_TYPE, 3> wgtfac_buf(wgtfac, r3);
    buffer<DATA_TYPE, 3> ppuv_buf(ppuv, r3);
    buffer<DATA_TYPE, 3> hhl_buf(hhl, r3);
    buffer<DATA_TYPE, 3> rho_buf(rho, r3);
    buffer<DATA_TYPE, 1> fx_buf(fx, r1);
    buffer<DATA_TYPE, 3> ppgk_buf(ppgk, r3);
    buffer<DATA_TYPE, 3> ppgc_buf(ppgc, r3);
    buffer<DATA_TYPE, 3> ppgu_buf(ppgu, r3);
    buffer<DATA_TYPE, 3> ppgv_buf(ppgv, r3);

    for (int i = 0; i < WARMUP_REPS; ++i)
      kernel_fastwavesuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                                 ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);

    TIMEIT({
      kernel_fastwavesuv_sycl_sw(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                                 ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);
    }, BENCH_REPS, "\n", "sycl-sw");
  }
}
