#include "fastwavesuv.h"

namespace {

constexpr size_t WG_SIZE = 256;
constexpr size_t CELLS = size_t(DOMAIN_SIZE) * size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT);

size_t round_up(size_t value, size_t factor)
{
  return ((value + factor - 1) / factor) * factor;
}

template <typename WgtAcc, typename PpuvAcc>
inline DATA_TYPE interp_ppgk(const WgtAcc &wgtfac, const PpuvAcc &ppuv, size_t i, size_t j, size_t k)
{
  return wgtfac[IDXSY(i, j, k)] * ppuv[IDXSY(i, j, k)] +
         (DATA_TYPE(1.0) - wgtfac[IDXSY(i, j, k)]) * ppuv[IDXSY(i, j, k - 1)];
}

inline void decode_index(size_t linear, size_t &i, size_t &j, size_t &k)
{
  i = linear / (size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT));
  const size_t rem = linear - i * size_t(DOMAIN_SIZE) * size_t(DOMAIN_HEIGHT);
  j = rem / size_t(DOMAIN_HEIGHT);
  k = rem - j * size_t(DOMAIN_HEIGHT);
}

void kernel_fastwavesuv_sycl_dcu(
    buffer<DATA_TYPE, 3> &uout_buf, buffer<DATA_TYPE, 3> &vout_buf, buffer<DATA_TYPE, 3> &uin_buf, buffer<DATA_TYPE, 3> &vin_buf,
    buffer<DATA_TYPE, 3> &utens_buf, buffer<DATA_TYPE, 3> &vtens_buf, buffer<DATA_TYPE, 3> &wgtfac_buf, buffer<DATA_TYPE, 3> &ppuv_buf,
    buffer<DATA_TYPE, 3> &hhl_buf, buffer<DATA_TYPE, 3> &rho_buf, buffer<DATA_TYPE, 1> &fx_buf, buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &,
    buffer<DATA_TYPE, 3> &, buffer<DATA_TYPE, 3> &, DATA_TYPE edadlat, DATA_TYPE dt, queue &Q)
{
  const range<1> local_range{WG_SIZE};
  const range<1> global_range{round_up(CELLS, WG_SIZE)};

  Q.submit([&](handler &h) {
    auto wgtfac = wgtfac_buf.get_access<access::mode::read>(h);
    auto ppuv = ppuv_buf.get_access<access::mode::read>(h);
    auto hhl = hhl_buf.get_access<access::mode::read>(h);
    auto rho = rho_buf.get_access<access::mode::read>(h);
    auto fx = fx_buf.get_access<access::mode::read>(h);
    auto uin = uin_buf.get_access<access::mode::read>(h);
    auto vin = vin_buf.get_access<access::mode::read>(h);
    auto utens = utens_buf.get_access<access::mode::read>(h);
    auto vtens = vtens_buf.get_access<access::mode::read>(h);
    auto uout = uout_buf.get_access<access::mode::write>(h);
    auto vout = vout_buf.get_access<access::mode::write>(h);

    h.parallel_for(nd_range<1>(global_range, local_range), [=](nd_item<1> item) {
      const size_t linear = item.get_global_id(0);
      if (linear < CELLS) {
        size_t i, j, k;
        decode_index(linear, i, j, k);

        const DATA_TYPE ppgk_c = interp_ppgk(wgtfac, ppuv, i, j, k);
        const DATA_TYPE ppgk_n = interp_ppgk(wgtfac, ppuv, i, j, k + 1);
        const DATA_TYPE ppgc_c = ppgk_n - ppgk_c;
        const DATA_TYPE ppgc_ip1 = interp_ppgk(wgtfac, ppuv, i + 1, j, k + 1) - interp_ppgk(wgtfac, ppuv, i + 1, j, k);
        const DATA_TYPE ppgc_jp1 = interp_ppgk(wgtfac, ppuv, i, j + 1, k + 1) - interp_ppgk(wgtfac, ppuv, i, j + 1, k);

        const DATA_TYPE ppgu_v = (ppuv[IDXSY(i + 1, j, k)] - ppuv[IDXSY(i, j, k)]) +
                                 (ppgc_ip1 + ppgc_c) * DATA_TYPE(0.5) *
                                     ((hhl[IDXSY(i, j, k + 1)] + hhl[IDXSY(i, j, k)]) - (hhl[IDXSY(i + 1, j, k + 1)] + hhl[IDXSY(i + 1, j, k)])) /
                                     ((hhl[IDXSY(i, j, k + 1)] - hhl[IDXSY(i, j, k)]) + (hhl[IDXSY(i + 1, j, k + 1)] - hhl[IDXSY(i + 1, j, k)]));
        const DATA_TYPE ppgv_v = (ppuv[IDXSY(i, j + 1, k)] - ppuv[IDXSY(i, j, k)]) +
                                 (ppgc_jp1 + ppgc_c) * DATA_TYPE(0.5) *
                                     ((hhl[IDXSY(i, j, k + 1)] + hhl[IDXSY(i, j, k)]) - (hhl[IDXSY(i, j + 1, k + 1)] + hhl[IDXSY(i, j + 1, k)])) /
                                     ((hhl[IDXSY(i, j, k + 1)] - hhl[IDXSY(i, j, k)]) + (hhl[IDXSY(i, j + 1, k + 1)] - hhl[IDXSY(i, j + 1, k)]));

        uout[IDXSY(i, j, k)] = uin[IDXSY(i, j, k)] +
                               dt * (utens[IDXSY(i, j, k)] - ppgu_v * (DATA_TYPE(2.0) * fx[IDX_1D(j)] / (rho[IDXSY(i + 1, j, k)] + rho[IDXSY(i, j, k)])));
        vout[IDXSY(i, j, k)] = vin[IDXSY(i, j, k)] +
                               dt * (vtens[IDXSY(i, j, k)] - ppgv_v * (DATA_TYPE(2.0) * edadlat / (rho[IDXSY(i, j + 1, k)] + rho[IDXSY(i, j, k)])));
      }
    });
  }).wait();
}

} // namespace

void fastwavesuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt)
{
  queue Q{gpu_selector_v};
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
    kernel_fastwavesuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                                ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);
  }
}

void bench_fastwavesuv_sycl_dcu(
    DATA_TYPE *uout, DATA_TYPE *vout, DATA_TYPE *uin, DATA_TYPE *vin, DATA_TYPE *utens, DATA_TYPE *vtens, DATA_TYPE *wgtfac, DATA_TYPE *ppuv,
    DATA_TYPE *hhl, DATA_TYPE *rho, DATA_TYPE *fx, DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, DATA_TYPE edadlat, DATA_TYPE dt)
{
  queue Q{gpu_selector_v};
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
      kernel_fastwavesuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                                  ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);

    TIMEIT({
      kernel_fastwavesuv_sycl_dcu(uout_buf, vout_buf, uin_buf, vin_buf, utens_buf, vtens_buf, wgtfac_buf, ppuv_buf, hhl_buf, rho_buf, fx_buf,
                                  ppgk_buf, ppgc_buf, ppgu_buf, ppgv_buf, edadlat, dt, Q);
    }, BENCH_REPS, "\n", "sycl-dcu");
  }
}
