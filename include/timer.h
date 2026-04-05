#include <sys/time.h>
#include <stdio.h>


#define TIMEIT(code_block, print_end, ...) do { \
    struct timeval start, end; \
    gettimeofday(&start, NULL); \
    code_block; \
    gettimeofday(&end, NULL); \
    double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0; \
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0; \
    printf(__VA_ARGS__); \
    printf("%.4f", elapsed_time); \
    printf(print_end); \
} while (0)