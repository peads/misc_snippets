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
#define TIMING_RUNS 4
#define MIN -2
#define MAX 2
#define STEP 1

#include "timed_functions.h"

static const uint32_t SQUARE_SIDE = (MAX - MIN) / STEP + 1;

char *runNames[TIMING_RUNS] = {"swap1 :: ", "swap1a :: ", "swap2 :: ", "swap2a :: "};


extern void swap1(void *x, void *y);
__asm__ (
#ifdef __APPLE_CC__
"_swap1: "
#else
"swap1: "
#endif
    "movq (%rsi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "xorq (%rdi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "movq %rax, (%rsi)\n\t"
    "ret"
);

extern void swap2(void *x, void *y);

static void swap1a(void *x, void *y) {

    **((uintptr_t **) &x) ^= **((uintptr_t **) &y);
    **((uintptr_t **) &y) ^= **((uintptr_t **) &x);
    **((uintptr_t **) &x) ^= **((uintptr_t **) &y);
}

static void swap2b(void *x, void *y) {

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

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap1(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(0, &tstart, &tend);
            assert(x == yy && y == xx);

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap1a(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(1, &tstart, &tend);
            assert(x == xx && y == yy);

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap2(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(2, &tstart, &tend);
            assert(x == yy && y == xx);

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            swap2b(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(3, &tstart, &tend);
            assert(x == xx && y == yy);

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
