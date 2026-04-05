#include "fastwavesuv.h"

void kernel_fastwavesuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *utens, const DATA_TYPE *vtens,
    const DATA_TYPE *wgtfac, const DATA_TYPE *ppuv, const DATA_TYPE *hhl, const DATA_TYPE *rho, const DATA_TYPE *fx,
    DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, const DATA_TYPE edadlat, const DATA_TYPE dt)
{
  for (ssize_t i = 0; i < DOMAIN_SIZE + 1; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE + 1; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT + 1; ++k)
        ppgk[IDX(i, j, k)] = wgtfac[IDX(i, j, k)] * ppuv[IDX(i, j, k)] +
                             (DATA_TYPE(1.0) - wgtfac[IDX(i, j, k)]) * ppuv[IDX(i, j, k - 1)];

  for (ssize_t i = 0; i < DOMAIN_SIZE + 1; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE + 1; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        ppgc[IDX(i, j, k)] = ppgk[IDX(i, j, k + 1)] - ppgk[IDX(i, j, k)];

  for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        ppgu[IDX(i, j, k)] = (ppuv[IDX(i + 1, j, k)] - ppuv[IDX(i, j, k)]) +
                             (ppgc[IDX(i + 1, j, k)] + ppgc[IDX(i, j, k)]) * DATA_TYPE(0.5) *
                                 ((hhl[IDX(i, j, k + 1)] + hhl[IDX(i, j, k)]) - (hhl[IDX(i + 1, j, k + 1)] + hhl[IDX(i + 1, j, k)])) /
                                 ((hhl[IDX(i, j, k + 1)] - hhl[IDX(i, j, k)]) + (hhl[IDX(i + 1, j, k + 1)] - hhl[IDX(i + 1, j, k)]));

  for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        ppgv[IDX(i, j, k)] = (ppuv[IDX(i, j + 1, k)] - ppuv[IDX(i, j, k)]) +
                             (ppgc[IDX(i, j + 1, k)] + ppgc[IDX(i, j, k)]) * DATA_TYPE(0.5) *
                                 ((hhl[IDX(i, j, k + 1)] + hhl[IDX(i, j, k)]) - (hhl[IDX(i, j + 1, k + 1)] + hhl[IDX(i, j + 1, k)])) /
                                 ((hhl[IDX(i, j, k + 1)] - hhl[IDX(i, j, k)]) + (hhl[IDX(i, j + 1, k + 1)] - hhl[IDX(i, j + 1, k)]));

  for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        uout[IDX(i, j, k)] = uin[IDX(i, j, k)] +
                             dt * (utens[IDX(i, j, k)] -
                                   ppgu[IDX(i, j, k)] * (DATA_TYPE(2.0) * fx[IDX_1D(j)] / (rho[IDX(i + 1, j, k)] + rho[IDX(i, j, k)])));

  for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
    for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        vout[IDX(i, j, k)] = vin[IDX(i, j, k)] +
                             dt * (vtens[IDX(i, j, k)] -
                                   ppgv[IDX(i, j, k)] * (DATA_TYPE(2.0) * edadlat / (rho[IDX(i, j + 1, k)] + rho[IDX(i, j, k)])));
}

void fastwavesuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *utens, const DATA_TYPE *vtens,
    const DATA_TYPE *wgtfac, const DATA_TYPE *ppuv, const DATA_TYPE *hhl, const DATA_TYPE *rho, const DATA_TYPE *fx,
    DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, const DATA_TYPE edadlat, const DATA_TYPE dt)
{
  kernel_fastwavesuv_serial(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv, edadlat, dt);
}

void bench_fastwavesuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *utens, const DATA_TYPE *vtens,
    const DATA_TYPE *wgtfac, const DATA_TYPE *ppuv, const DATA_TYPE *hhl, const DATA_TYPE *rho, const DATA_TYPE *fx,
    DATA_TYPE *ppgk, DATA_TYPE *ppgc, DATA_TYPE *ppgu, DATA_TYPE *ppgv, const DATA_TYPE edadlat, const DATA_TYPE dt)
{
  for (int i = 0; i < WARMUP_REPS; ++i)
    kernel_fastwavesuv_serial(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv, edadlat, dt);

  TIMEIT({
    kernel_fastwavesuv_serial(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv, edadlat, dt);
  }, BENCH_REPS, "\n", "serial");
}
