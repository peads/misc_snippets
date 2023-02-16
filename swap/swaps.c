/*
 * This file is part of the misc_snippets distribution (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023 Patrick Eads.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#define DEBUG
#define TIMING_RUNS 3
#define MIN -30.0
#define MAX 30.0
#define STEP 0.00001

#include "timed_functions.h"

static const uint32_t SQUARE_SIDE = (MAX - MIN) / STEP + 1;

char *runNames[TIMING_RUNS] = {"swap :: ", "swap1 :: ", "swap2 :: "};

extern void swap2(void *x, void *y);
__asm__ (
#ifdef __APPLE_CC__
"_swap2: "
#else
"swap2: "
#endif
    "movq (%rsi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "xorq (%rdi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "movq %rax, (%rsi)\n\t"
    "ret"
);

static inline void swap1(void *x, void *y) {

    **((uintptr_t **) &x) ^= **((uintptr_t **) &y);
    **((uintptr_t **) &y) ^= **((uintptr_t **) &x);
    **((uintptr_t **) &x) ^= **((uintptr_t **) &y);
}

static inline void swap(void *x, void *y) {

    uintptr_t temp = *(uintptr_t *) x;
    **((uintptr_t **) &x) = **((uintptr_t **) &y);
    *(uintptr_t *) y = *((uintptr_t *) &temp);

}

int main(void) {

    struct timespec tstart, tend;

    static const uint64_t n = SQUARE_SIDE * SQUARE_SIDE;

    double x, y, xx, yy;

    for (x = y = MIN; x < MAX; y += STEP) {

        if ((*(uint64_t *) &x & *(uint64_t *) &y)) { // x and y not 0

            xx = x;
            yy = y;
//            printf("%sx: %f, y: %f\n", runNames[0], x, y);
            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(0, &tstart, &tend);

//            printf("%sx: %f, y: %f\n", runNames[0], x, y);
            assert(x == yy && y == xx);

//            printf("%sx: %f, y: %f\n", runNames[1], x, y);
            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap1(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);

            findDeltaTime(1, &tstart, &tend);

//            printf("%sx: %f, y: %f\n", runNames[1], x, y);
            assert(x == xx && y == yy);

//            printf("%sx: %f, y: %f\n", runNames[2], x, y);
            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap2(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(2, &tstart, &tend);
//            printf("%sx: %f, y: %f\n", runNames[2], x, y);
            assert(x == yy && y == xx);

        }
        if (y >= MAX) {
            x += STEP;
            y = MIN;
        }
    }

    printTimedRuns(runNames, TIMING_RUNS);
    printf("Iterations: %llu\n", n);
    return 0;
}
