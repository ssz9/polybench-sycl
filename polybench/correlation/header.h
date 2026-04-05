#ifndef _CORRELATION_H
#define _CORRELATION_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include "sycl/sycl.hpp"
#include "timer.h"


#ifndef _PB_M
#define _PB_M 1024
#endif
#ifndef _PB_N
#define _PB_N 1024
#endif

#define DATA_TYPE double
#define SQRT_FUN std::sqrt
#define IDX_DATA(i, j) ((i)*(_PB_M) + (j))
#define IDX_CORR(i, j) ((i)*(_PB_M) + (j))

#define WARMUP_REPS 1

using namespace sycl;


inline void memset_zero(void *data, size_t size)
{
    memset(data, 0, size);
}

template <int Dim>
void memset_zero_sycl(buffer<DATA_TYPE, Dim> &buf, queue &Q)
{
    Q.submit([&](handler &h){
        auto acc = buf.template get_access<access::mode::write>(h);
        h.fill(acc, DATA_TYPE{0});
    }).wait();
}

inline bool compare_array(const DATA_TYPE *lhs, const DATA_TYPE *rhs, size_t size, DATA_TYPE rtol = DATA_TYPE{1e-6}, DATA_TYPE atol = DATA_TYPE{1e-9})
{
    for (size_t i = 0; i < size; i++) {
        DATA_TYPE diff = std::fabs(lhs[i] - rhs[i]);
        DATA_TYPE limit = atol + rtol * std::fabs(lhs[i]);
        if (diff > limit) {
            std::printf("Mismatch at %zu: lhs=%0.12lf rhs=%0.12lf diff=%0.12lf limit=%0.12lf\n", i, lhs[i], rhs[i], diff, limit);
            return false;
        }
    }
    return true;
}


#endif // _CORRELATION_H
