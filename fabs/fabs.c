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
//#define  DEBUG

#ifdef DEBUG
    #define PRINTF printf
#else
    #define PRINTF //
#endif
#define TIMING_RUNS 5
#include "timed_functions.h"

#define MIN -10000.f
#define MAX 10000.f
#define STEP 0.001f
#define MAX_ERROR 1e-9
#define ABS_MASK 0x7FFFFFFF

extern float ffabsf(float f);

extern void fpabs(float *f);
__asm__(
#ifdef __APPLE_CC__
"_fpabs: "
#else
"fpabs: "
#endif
    "andl $0x7FFFFFFF, (%rdi)\n\t"
    "retq"
);

static inline void sneakyFabs2(float *__restrict__ x, float *__restrict__ result){

    union {
        float f;
        uint32_t i;
    } unf = {*x};

    unf.i &= ABS_MASK;

    *result = unf.f;
}

static inline void sneakyFabs(float *__restrict__ x, float *__restrict__ result) {
    uint32_t absX = (*(uint32_t *)x) & ABS_MASK;
    *result = (*(float *)&absX);
}

int main(void) {
    static char *runNames[TIMING_RUNS] = {"sneakyFabs2 :: ", "ffabsf :: ","fpabs :: ", "fabs :: ", "sneakyFabs2 :: "};
    struct timespec tstart, tend;

    int n, err;
    float x,y, results[TIMING_RUNS];

    for (y = MIN; y < MAX; y += STEP, n++) {

        x = y;
        timeFun((timedFun) sneakyFabs2, &y, (void *) &(results[0]), 0);

        timeFun((timedFun) sneakyFabs, &y, (void *) &(results[4]), 4);

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        results[1] = ffabsf(x);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(1, &tstart, &tend);
        // END TIMED

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        results[2] = x;
        fpabs(&(results[2]));

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(2, &tstart, &tend);
        // END TIMED

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        results[3] = fabs(y);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(3, &tstart, &tend);
        // END TIMED
#ifdef DEBUG
        printf("x: %f :: %f %f %f %f %f\n", x, results[0], results[1], results[2], results[3], results[4]);
        if (results[0] != results[1]) err++;
#endif
        assert(fabs(results[3] - results[1]) < MAX_ERROR
            && fabs(results[3] - results[2]) < MAX_ERROR
            && fabs(results[3] - results[0]) < MAX_ERROR
            && fabs(results[3] - results[4]) < MAX_ERROR);
    }
    printTimedRuns(runNames, TIMING_RUNS);
    PRINTF("iterations: %d, %.2f%%\n", n, 100.f*err/n);
}