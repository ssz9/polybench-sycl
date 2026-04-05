#ifndef _COVARIANCE_H
#define _COVARIANCE_H

#include <cassert>

#ifndef _PB_M
#define _PB_M 1024
#endif
#ifndef _PB_N
#define _PB_N 1024
#endif

#define IDX_DATA(i, j) ((i) * (_PB_M) + (j))
#define IDX_SYMM(i, j) ((i) * (_PB_M) + (j))

#include "utils.h"

#endif
