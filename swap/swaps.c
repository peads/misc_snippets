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

#define TIMING_RUNS 3
#define MIN -1000000.0
#define MAX 1000000.0
#define STEP 0.1
#define MAX_ERROR 1e-1

#include "timed_functions.h"

typedef void (*runnableTest)(void *x, void *y);
struct swapArgs {
    double x;
    double y;
    runnableTest fun;
};

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

    uintptr_t **i = (uintptr_t**)&x;
    uintptr_t **j = (uintptr_t**)&y;

    **i ^= **j;
    **j ^= **i;
    **i ^= **j;
}

static inline void swap(__attribute__((unused)) void *x,
                        __attribute__((unused)) void *y) {

    __asm__ __volatile__ (
        "movq (%%rsi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "xorq (%%rdi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "movq %%rax, (%%rsi)\n\t"
        ://:::"rax"
    );
}

void runTest(void* args, ...) {

    struct swapArgs *sargs = args;
    sargs->fun(&sargs->x, &sargs->y);
}

void testIteration(struct swapArgs *args, int runIndex) {

    double xx = args->x;
    double yy = args->y;

//    runTest(args);
    timeFun((timedFun) runTest, args, NULL, runIndex);
    assert(xx == args->y && yy == args->x);
}

int main(void) {

    static const uint64_t n = SQUARE_SIDE * SQUARE_SIDE;

    int i;
    struct swapArgs args;

    for (i = 0, args.x = args.y = MIN; i < n && args.x < MAX; ++i, args.y += STEP) {

        if ((*(uint64_t *)&args.x & *(uint64_t *)&args.y)) { // x and y not 0

            args.fun = swap;
            testIteration(&args, 0);

            args.fun = swap1;
            testIteration(&args, 1);

            args.fun = swap2;
            testIteration(&args, 2);
        }

        if (args.y >= MAX) {
            args.x += STEP;
            args.y = MIN;
        }
    }

    printTimedRuns(runNames, TIMING_RUNS);
    printf("Iterations: %llu\n", n);
    return 0;
}