#ifndef _GEMM_H
#define _GEMM_H

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024
#endif

#define ALPHA 32412
#define BETA 2123
#define IDX(i, j) ((i) * MATRIX_SIZE + (j))

#include "utils.h"

#endif
