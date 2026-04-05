#ifndef SW_SYCL_UTILS_H
#define SW_SYCL_UTILS_H

#include <cstddef>

#include "sycl/sycl.hpp"

constexpr size_t SW_CPE_COUNT = 64;

inline sycl::range<1> sw_cpe_range()
{
  return sycl::range<1>(SW_CPE_COUNT);
}

inline size_t sw_worker_id(const sycl::id<1> &worker)
{
  return worker[0];
}

inline size_t sw_block_begin(size_t total, size_t worker_id)
{
  return (total * worker_id) / SW_CPE_COUNT;
}

inline size_t sw_block_end(size_t total, size_t worker_id)
{
  return (total * (worker_id + 1)) / SW_CPE_COUNT;
}

#endif
