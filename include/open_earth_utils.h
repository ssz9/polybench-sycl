#ifndef OPEN_EARTH_UTILS_H
#define OPEN_EARTH_UTILS_H

#include <cmath>
#include <cstdlib>

#include "utils.h"

#ifndef HALO_WIDTH
#define HALO_WIDTH 2
#endif

#ifndef DOMAIN_SIZE
#define DOMAIN_SIZE 32
#endif

#ifndef DOMAIN_HEIGHT
#define DOMAIN_HEIGHT 32
#endif

#define IDX_1D(n) ((n) + HALO_WIDTH)
#define IDX(i, j, k) (((i) + HALO_WIDTH) * (DOMAIN_SIZE + (HALO_WIDTH * 2)) * (DOMAIN_HEIGHT + (HALO_WIDTH * 2)) + \
                      ((j) + HALO_WIDTH) * (DOMAIN_HEIGHT + (HALO_WIDTH * 2)) + ((k) + HALO_WIDTH))
#define IDXSY(i, j, k) {size_t((i) + HALO_WIDTH), size_t((j) + HALO_WIDTH), size_t((k) + HALO_WIDTH)}

#define EARTH_RADIUS ((DATA_TYPE)6371.229e3)
#define EARTH_RADIUS_RECIP ((DATA_TYPE)1.0 / EARTH_RADIUS)

inline size_t open_earth_3d_size()
{
  return (size_t)(DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_SIZE + 2 * HALO_WIDTH) * (DOMAIN_HEIGHT + 2 * HALO_WIDTH);
}

inline size_t open_earth_1d_size()
{
  return (size_t)(DOMAIN_SIZE + 2 * HALO_WIDTH);
}

inline DATA_TYPE *alloc3D(size_t domain_size, size_t domain_height)
{
  return (DATA_TYPE *)std::malloc((domain_size + 2 * HALO_WIDTH) * (domain_size + 2 * HALO_WIDTH) * (domain_height + 2 * HALO_WIDTH) * sizeof(DATA_TYPE));
}

inline DATA_TYPE *alloc1D(size_t domain_size)
{
  return (DATA_TYPE *)std::malloc((domain_size + 2 * HALO_WIDTH) * sizeof(DATA_TYPE));
}

inline void initValue(DATA_TYPE *field, const DATA_TYPE val)
{
  for (ssize_t i = -HALO_WIDTH; i < DOMAIN_SIZE + HALO_WIDTH; ++i)
    for (ssize_t j = -HALO_WIDTH; j < DOMAIN_SIZE + HALO_WIDTH; ++j)
      for (ssize_t k = -HALO_WIDTH; k < DOMAIN_HEIGHT + HALO_WIDTH; ++k)
        field[IDX(i, j, k)] = val;
}

inline void fillMath3D(DATA_TYPE a, DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f, DATA_TYPE *field)
{
  const DATA_TYPE dx = DATA_TYPE(1.0) / DATA_TYPE(DOMAIN_SIZE + 2 * HALO_WIDTH);
  const DATA_TYPE dy = DATA_TYPE(1.0) / DATA_TYPE(DOMAIN_SIZE + 2 * HALO_WIDTH);
  const DATA_TYPE pi = std::acos(DATA_TYPE(-1.0));

  for (ssize_t j = -HALO_WIDTH; j < DOMAIN_SIZE + HALO_WIDTH; ++j) {
    for (ssize_t i = -HALO_WIDTH; i < DOMAIN_SIZE + HALO_WIDTH; ++i) {
      const DATA_TYPE x = dx * DATA_TYPE(i);
      const DATA_TYPE y = dy * DATA_TYPE(j);
      for (ssize_t k = 0; k < DOMAIN_HEIGHT; ++k)
        field[IDX(i, j, k)] = DATA_TYPE(k) * DATA_TYPE(10e-3) +
                              a * (b + std::cos(pi * (x + c * y)) + std::sin(d * pi * (x + e * y))) / f;
    }
  }
}

inline void fillMath1D(DATA_TYPE a, DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f, DATA_TYPE *field)
{
  const DATA_TYPE dx = DATA_TYPE(1.0) / DATA_TYPE(DOMAIN_SIZE + 2 * HALO_WIDTH);
  const DATA_TYPE pi = std::acos(DATA_TYPE(-1.0));

  for (ssize_t i = -HALO_WIDTH; i < DOMAIN_SIZE + HALO_WIDTH; ++i) {
    const DATA_TYPE x = dx * DATA_TYPE(i);
    field[IDX_1D(i)] = a * (b + std::cos(pi * (c * x)) + std::sin(d * pi * (e * x))) / f;
  }
}

#endif
