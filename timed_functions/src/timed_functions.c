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

#include "timed_functions.h"

static inline void timespec_diff(const struct timespec *a, const struct timespec *b,
                                 struct timespec *result) {

    result->tv_sec = a->tv_sec - b->tv_sec;
    result->tv_nsec = a->tv_nsec - b->tv_nsec;
    if (result->tv_nsec < 0) {
        --result->tv_sec;
        result->tv_nsec += 1000000000L;
    }
}

static void findDeltaTime(int idx, const struct timespec *tstart, const struct timespec *tend) {

    static uint64_t counts[TIMING_RUNS];
    struct timespec tau;

    timespec_diff(tend, tstart, &tau);

//    printf("%sCAnp1 = (xnp1 + n * CAn)/(n + 1) = (%ld + %llu * %LF)/%llu\n", runNames[idx], tau.tv_nsec, counts[idx], rollingTimeAvgs[idx], counts[idx] + 1);
    rollingTimeAvgs[idx] = ((long double) tau.tv_nsec + (long double) counts[idx] * rollingTimeAvgs[idx]) / (long double) ++counts[idx];
//    timeDiffSums[idx] += tau.tv_nsec;
    //printf("curr avg %Lf\n", rollingTimeAvgs[idx]);

    //printf("%llu: %ld.%.9ldns %LF\n", counts[idx]-1, tau.tv_sec, tau.tv_nsec, rollingTimeAvgs[idx]);
}

uint8_t timeFunD(timedFunD fun, double s, int i) {

    uint8_t result;
    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    result = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);

    return result;

}

uint8_t timeFunF(timedFunF fun, float s, int i) {

    uint8_t result;
    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    result = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);

    return result;
}

void timeFun(timedFun fun, void *s, void **result, int i) {

    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    fun(s, &(result[i]));

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);
}

void printTimedRuns(char **runNames, uint32_t length) {

    int j;
    for (j = 0; j < length; ++j) {
        const char *runName = runNames[j];
        printf("%sAverage time: %Lf ns\n", runName, rollingTimeAvgs[j]);
//        printf("%sAverage time: %Lf ns\n", runName, (long double) timeDiffSums[j] / *counts);
    }
}
