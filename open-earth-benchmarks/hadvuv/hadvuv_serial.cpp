#include "hadvuv.h"

DATA_TYPE advection_driver(const DATA_TYPE *field, ssize_t i, ssize_t j, ssize_t k, DATA_TYPE uavg, DATA_TYPE vavg, DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  DATA_TYPE result_x = DATA_TYPE{0};
  DATA_TYPE result_y = DATA_TYPE{0};

  if (uavg > DATA_TYPE{0})
    result_x = uavg * (DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDX(i - 2, j, k)] + field[IDX(i - 1, j, k)] +
                       DATA_TYPE(-0.5) * field[IDX(i, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDX(i + 1, j, k)]);
  else
    result_x = -uavg * (DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDX(i - 1, j, k)] + DATA_TYPE(-0.5) * field[IDX(i, j, k)] +
                        field[IDX(i + 1, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDX(i + 2, j, k)]);

  if (vavg > DATA_TYPE{0})
    result_y = vavg * (DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDX(i, j - 2, k)] + field[IDX(i, j - 1, k)] +
                       DATA_TYPE(-0.5) * field[IDX(i, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDX(i, j + 1, k)]);
  else
    result_y = -vavg * (DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDX(i, j - 1, k)] + DATA_TYPE(-0.5) * field[IDX(i, j, k)] +
                        field[IDX(i, j + 1, k)] + DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDX(i, j + 2, k)]);

  return eddlat * result_x + eddlon * result_y;
}

void kernel_hadvuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *acrlat0, const DATA_TYPE *acrlat1,
    const DATA_TYPE *tgrlatda0, const DATA_TYPE *tgrlatda1, DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos,
    DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres, DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j) {
        uatupos[IDX(i, j, k)] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (uin[IDX(i - 1, j, k)] + uin[IDX(i, j, k)] + uin[IDX(i + 1, j, k)]);
        vatupos[IDX(i, j, k)] = DATA_TYPE(0.25) * (vin[IDX(i + 1, j, k)] + vin[IDX(i + 1, j - 1, k)] + vin[IDX(i, j, k)] + vin[IDX(i, j - 1, k)]);
      }

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j) {
        uavg[IDX(i, j, k)] = acrlat0[IDX_1D(j)] * uatupos[IDX(i, j, k)];
        vavg[IDX(i, j, k)] = EARTH_RADIUS_RECIP * vatupos[IDX(i, j, k)];
      }

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
        ures[IDX(i, j, k)] = advection_driver(uin, i, j, k, uavg[IDX(i, j, k)], vavg[IDX(i, j, k)], eddlat, eddlon);

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
        uout[IDX(i, j, k)] = ures[IDX(i, j, k)] + tgrlatda0[IDX_1D(j)] * uin[IDX(i, j, k)] * vatupos[IDX(i, j, k)];

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j) {
        uatvpos[IDX(i, j, k)] = DATA_TYPE(0.25) * (uin[IDX(i - 1, j, k)] + uin[IDX(i, j, k)] + uin[IDX(i, j + 1, k)] + uin[IDX(i - 1, j + 1, k)]);
        vatvpos[IDX(i, j, k)] = (DATA_TYPE(1.0) / DATA_TYPE(3.0)) * (vin[IDX(i, j - 1, k)] + vin[IDX(i, j, k)] + vin[IDX(i, j + 1, k)]);
      }

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j) {
        uavg[IDX(i, j, k)] = acrlat1[IDX_1D(j)] * uatvpos[IDX(i, j, k)];
        vavg[IDX(i, j, k)] = EARTH_RADIUS_RECIP * vatvpos[IDX(i, j, k)];
      }

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
        vres[IDX(i, j, k)] = advection_driver(vin, i, j, k, uavg[IDX(i, j, k)], vavg[IDX(i, j, k)], eddlat, eddlon);

  for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
    for (ssize_t i = 0; i < DOMAIN_SIZE; ++i)
      for (ssize_t j = 0; j < DOMAIN_SIZE; ++j)
        vout[IDX(i, j, k)] = vres[IDX(i, j, k)] - tgrlatda1[IDX_1D(j)] * uatvpos[IDX(i, j, k)] * uatvpos[IDX(i, j, k)];
}

void hadvuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *acrlat0, const DATA_TYPE *acrlat1,
    const DATA_TYPE *tgrlatda0, const DATA_TYPE *tgrlatda1, DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos,
    DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres, DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  kernel_hadvuv_serial(uout, vout, uin, vin, acrlat0, acrlat1, tgrlatda0, tgrlatda1, uatupos, vatupos, uatvpos, vatvpos, uavg, vavg, ures, vres,
                       eddlat, eddlon);
}

void bench_hadvuv_serial(
    DATA_TYPE *uout, DATA_TYPE *vout, const DATA_TYPE *uin, const DATA_TYPE *vin, const DATA_TYPE *acrlat0, const DATA_TYPE *acrlat1,
    const DATA_TYPE *tgrlatda0, const DATA_TYPE *tgrlatda1, DATA_TYPE *uatupos, DATA_TYPE *vatupos, DATA_TYPE *uatvpos, DATA_TYPE *vatvpos,
    DATA_TYPE *uavg, DATA_TYPE *vavg, DATA_TYPE *ures, DATA_TYPE *vres, DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  for (int i = 0; i < WARMUP_REPS; ++i)
    kernel_hadvuv_serial(uout, vout, uin, vin, acrlat0, acrlat1, tgrlatda0, tgrlatda1, uatupos, vatupos, uatvpos, vatvpos, uavg, vavg, ures, vres,
                         eddlat, eddlon);

  TIMEIT({
    kernel_hadvuv_serial(uout, vout, uin, vin, acrlat0, acrlat1, tgrlatda0, tgrlatda1, uatupos, vatupos, uatvpos, vatvpos, uavg, vavg, ures, vres,
                         eddlat, eddlon);
  }, BENCH_REPS, "\n", "serial");
}
