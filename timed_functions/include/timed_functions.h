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

#ifndef DICKING_ABOUT_TIMED_FUNCTIONS_H
#define DICKING_ABOUT_TIMED_FUNCTIONS_H
#ifndef TIMING_RUNS
    #define TIMING_RUNS 4
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <unistd.h>

typedef uint8_t (*timedFunF)(float n);
typedef uint8_t (*timedFunD)(double n);
typedef void (*timedFun)(void *n, void *result);

static long double rollingTimeAvgs[TIMING_RUNS];

static void findDeltaTime(int idx, const struct timespec *tstart, const struct timespec *tend);
//uint8_t timeFunD(timedFunD fun, double s, int i);
//uint8_t timeFunF(timedFunF fun, float s, int i);

void timeFun(timedFun fun, void *s, void **result, int i);

void printTimedRuns(char **runNames, uint32_t length);
#endif //DICKING_ABOUT_TIMED_FUNCTIONS_H
