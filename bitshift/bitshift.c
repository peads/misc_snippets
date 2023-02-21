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
#include <stdint.h>
#define TIMING_RUNS 2
#include "timed_functions.h"

char *runNames[TIMING_RUNS] = {"bsr :: ", "bsrLzcnt :: " };

extern int bsr(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsr: "
#else
"bsr: "
#endif
    "movq $-1023, %rdx\n\t" // store 0 to compare against ZF later
    "bsr %rdi, %rax\n\t"
    "cmovzq %rdx, %rax\n\t"
    "ret"
);

extern int lzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_lzcnt: "
#else
"lzcnt: "
#endif
    "movq $-1023, %rdx\n\t" // store -1023 to return if CF set later
    "lzcnt %rdi, %rax\n\t"
    "cmovcq %rdx, %rax\n\t"
    "ret"
);

extern int bsrLzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsrLzcnt: "
#else
"bsrLzcnt: "
#endif
    "lzcnt %rdi, %rdx\n\t"
    "jc zero\n\t" // jump to return -1023 if input was 0
    "movq $63, %rax\n\t"
    "subq %rdx, %rax\n\t"
    "ret\n\t"
"zero: "
    "movq $-1023, %rax\n\t"
    "ret"
);

int main(void) {

    uint64_t i;

    for (i = 0L; i < 10000000L; ++i) {
        int correct = bitScanReverse(i);
        int result = bsr(i);
        struct timespec tstart, tend;

        clock_gettime(CLOCK_MONOTONIC, &tstart);

        assert(result == correct);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(0, &tstart, &tend);

        clock_gettime(CLOCK_MONOTONIC, &tstart);

        result = bsrLzcnt(i);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(1, &tstart, &tend);

        assert(result == correct);
    }

    printTimedRuns(runNames, TIMING_RUNS);

    return 0;
}
