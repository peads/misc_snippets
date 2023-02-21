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

#define TIMING_RUNS 8
#include "timed_functions.h"

#define DEBUG_PATTERN "%s%f -> %f "
#define MAX_ERROR 2
//#define DEBUG

static const uint8_t SIZE = TIMING_RUNS >> 1;


extern float __attribute__((aligned(16))) fsqrtf(float n __attribute__((aligned(16))));
extern float __attribute__((aligned(16))) fsqrt(float n __attribute__((aligned(16))));
extern float ffabsf(float f);


// The next two are too trivially to be implemented as to not be worth putting in the header
extern float vectorSqrtf(float n __attribute__((aligned(16))));
__asm__ (
#ifdef __clang__
"_vectorSqrtf: "
#else
"vectorSqrtf: "
#endif
#ifdef X86
    "vsqrtss %xmm0, %xmm1, %xmm0\n\t"
    "ret"
#else
    "VSQRT.F32 %0, %0" // TODO fix ARM implmentaton
#endif
);

extern double vectorSqrt(double n __attribute__((aligned(16))));
__asm__ (
#ifdef __clang__
"_vectorSqrt: "
#else
"vectorSqrt: "
#endif

#ifdef X86
    "vsqrtsd %xmm0, %xmm1, %xmm0\n\t"
    "ret"
#else
    "VSQRT.F64 %0, %0" // TODO fix ARM implmentaton
#endif
);

int main(void) {

    char *runNames[TIMING_RUNS]
            = {"sqrtApproxf :: ", "vectorSqrtf :: ", "fsqrtf :: ", "sqrtf :: ",
               "sqrtApprox :: ", "vectorSqrt :: ", "fsqrt :: ", "sqrt :: "};
    int i = 0;
    int j, k;
    float x = 1.0f;
    union unFloat sf[SIZE];
    union unDouble s[SIZE];

    for (; i < 1000001; ++i, x += 0.01) {

        sf[0] = timeFunf((timedFunF) sqrtApproxf, x, 0);
        sf[1] = timeFunf((timedFunF) vectorSqrtf, x, 1);
        sf[2] = timeFunf((timedFunF) fsqrtf, x, 2);
        sf[3] = timeFunf((timedFunF) sqrtf, x, 3);

        s[0] = timeFund((timedFunD) sqrtApprox, x, 4);
        s[1] = timeFund((timedFunD) vectorSqrt, x, 5);
        s[2] = timeFund((timedFunD) fsqrt, x, 6);
        s[3] = timeFund((timedFunD) sqrt, x, 7);

        for (j = 0; j < 2; ++j) {
            for (k = 1; k < SIZE; ++k) {
                const float deltaf = ffabsf(ffabsf(sf[3].f) - ffabsf(sf[k].f));
                const double delta = (fabs(fabs(s[3].f) - fabs(s[k].f)));
                switch (j) {
#ifndef DEBUG
                    case 0:
                        assert(deltaf < MAX_ERROR);
                        break;
                    case 1:
                        assert(delta < MAX_ERROR);
                        break;
#else

                    case 0:
                        printf(DEBUG_PATTERN"\n"/*"%X\n"*/, runNames[k], x, sf[k].f);//, sf[k].i);
                        break;
                    case 1:
                        printf(DEBUG_PATTERN"\n"/*"%lX\n"*/, runNames[k], x, s[k].f);//, s[k].i);
                        break;
#endif

                    default:exit(-1);
                }
            }
        }
    }

    printTimedRuns(runNames, TIMING_RUNS);
    return 0;
}

