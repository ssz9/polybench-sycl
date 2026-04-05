#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include "sycl/sycl.hpp"

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef COMPARE_RTOL
#define COMPARE_RTOL DATA_TYPE{2e-5f}
#endif

#ifndef COMPARE_ATOL
#define COMPARE_ATOL DATA_TYPE{1e-8f}
#endif

#ifndef WARMUP_REPS
#define WARMUP_REPS 1
#endif

#ifndef BENCH_REPS
#define BENCH_REPS 1
#endif

using namespace sycl;

inline void memset_zero(void *data, size_t size)
{
  std::memset(data, 0, size);
}

template <int Dim>
inline void memset_zero_sycl(buffer<DATA_TYPE, Dim> &buf, queue &Q)
{
  Q.submit([&](handler &h) {
    auto acc = buf.template get_access<access::mode::write>(h);
    h.fill(acc, DATA_TYPE{0});
  }).wait();
}

inline bool compare_array(const DATA_TYPE *lhs, const DATA_TYPE *rhs, size_t size,
                          DATA_TYPE rtol = COMPARE_RTOL, DATA_TYPE atol = COMPARE_ATOL)
{
  for (size_t i = 0; i < size; i++) {
    DATA_TYPE diff = std::fabs(lhs[i] - rhs[i]);
    DATA_TYPE limit = atol + rtol * std::fabs(lhs[i]);
    if (diff > limit) {
      std::printf("Mismatch at %zu: lhs=%0.12lf rhs=%0.12lf diff=%0.12lf limit=%0.12lf\n",
                  i, lhs[i], rhs[i], diff, limit);
      return false;
    }
  }
  return true;
}

#define TIMEIT(code_block, reps, print_end, label) do { \
    struct timeval start, end; \
    int timeit_reps_ = (reps); \
    if (timeit_reps_ <= 0) timeit_reps_ = 1; \
    double timeit_total_ms_ = 0.0; \
    double timeit_min_ms_ = -1.0; \
    double timeit_max_ms_ = 0.0; \
    for (int timeit_i_ = 0; timeit_i_ < timeit_reps_; ++timeit_i_) { \
        gettimeofday(&start, NULL); \
        code_block; \
        gettimeofday(&end, NULL); \
        double timeit_elapsed_ms_ = (end.tv_sec - start.tv_sec) * 1000.0; \
        timeit_elapsed_ms_ += (end.tv_usec - start.tv_usec) / 1000.0; \
        timeit_total_ms_ += timeit_elapsed_ms_; \
        if (timeit_min_ms_ < 0.0 || timeit_elapsed_ms_ < timeit_min_ms_) timeit_min_ms_ = timeit_elapsed_ms_; \
        if (timeit_elapsed_ms_ > timeit_max_ms_) timeit_max_ms_ = timeit_elapsed_ms_; \
    } \
    double timeit_avg_ms_ = timeit_total_ms_ / timeit_reps_; \
    printf("[%-20s] avg: %10.4f ms, min: %10.4f ms, max: %10.4f ms", (label), timeit_avg_ms_, timeit_min_ms_, timeit_max_ms_); \
    printf(print_end); \
} while (0)

#endif
