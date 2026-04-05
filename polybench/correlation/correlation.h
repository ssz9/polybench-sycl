#ifndef _CORRELATION_H
#define _CORRELATION_H

#ifndef _PB_M
#define _PB_M 1024
#endif
#ifndef _PB_N
#define _PB_N 1024
#endif

#define SQRT_FUN std::sqrt
#define IDX_DATA(i, j) ((i)*(_PB_M) + (j))
#define IDX_CORR(i, j) ((i)*(_PB_M) + (j))

#include "utils.h"

#endif // _CORRELATION_H
