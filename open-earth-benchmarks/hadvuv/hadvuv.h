#ifndef HADVUV_H
#define HADVUV_H

#define DATA_TYPE double
#define COMPARE_RTOL DATA_TYPE{1e-12}
#define COMPARE_ATOL DATA_TYPE{1e-12}

#define DOMAIN_SIZE 256
#define DOMAIN_HEIGHT 256

#include "open_earth_utils.h"

DATA_TYPE advection_driver(const DATA_TYPE *field, ssize_t i, ssize_t j, ssize_t k, DATA_TYPE uavg, DATA_TYPE vavg, DATA_TYPE eddlat, DATA_TYPE eddlon);

template <typename T>
inline DATA_TYPE advection_driver_sycl(const T &field, ssize_t i, ssize_t j, ssize_t k, DATA_TYPE uavg, DATA_TYPE vavg, DATA_TYPE eddlat, DATA_TYPE eddlon)
{
  DATA_TYPE result_x = DATA_TYPE{0};
  DATA_TYPE result_y = DATA_TYPE{0};

  if (uavg > DATA_TYPE{0})
    result_x = uavg * (DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDXSY(i - 2, j, k)] + field[IDXSY(i - 1, j, k)] +
                       DATA_TYPE(-0.5) * field[IDXSY(i, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDXSY(i + 1, j, k)]);
  else
    result_x = -uavg * (DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDXSY(i - 1, j, k)] + DATA_TYPE(-0.5) * field[IDXSY(i, j, k)] +
                        field[IDXSY(i + 1, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDXSY(i + 2, j, k)]);

  if (vavg > DATA_TYPE{0})
    result_y = vavg * (DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDXSY(i, j - 2, k)] + field[IDXSY(i, j - 1, k)] +
                       DATA_TYPE(-0.5) * field[IDXSY(i, j, k)] + DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDXSY(i, j + 1, k)]);
  else
    result_y = -vavg * (DATA_TYPE(-1.0) / DATA_TYPE(3.0) * field[IDXSY(i, j - 1, k)] + DATA_TYPE(-0.5) * field[IDXSY(i, j, k)] +
                        field[IDXSY(i, j + 1, k)] + DATA_TYPE(-1.0) / DATA_TYPE(6.0) * field[IDXSY(i, j + 2, k)]);

  return eddlat * result_x + eddlon * result_y;
}

#endif
