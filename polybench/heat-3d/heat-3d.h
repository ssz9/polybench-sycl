#ifndef _HEAT_3D_H
#define _HEAT_3D_H

#include "utils.h"

#ifndef HEAT_3D_N
#define HEAT_3D_N 512
#endif

#ifndef TSTEPS
#define TSTEPS 10
#endif

#define SCALAR_VAL(x) x##f
#define INDEX(i, j, k) ((i) * HEAT_3D_N * HEAT_3D_N + (j) * HEAT_3D_N + (k))

#endif
