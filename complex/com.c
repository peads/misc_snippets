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

#include <stdio.h>
#include <immintrin.h>
#include <time.h>
//#include <emmintrin.h>
#include <math.h>
#include <stdint.h>

#define TIMING_RUNS 4
#include <assert.h>
#include "timed_functions.h"
uint8_t DEBUG = 0;

struct complexArgs {
    int ar;
    int aj;
    int br;
    int bj;
};

union vect {
    __m256d vect;
};

static double norm(int x, int y) {

    return sqrt(x * x + y * y);
}

static void calcVectPd(void *args, double *result) {

    struct complexArgs *cargs = args;
    union vect u = {cargs->ar, cargs->aj, cargs->ar, cargs->br};
    union vect v = {cargs->br, cargs->bj, cargs->bj, cargs->aj};

    double zr;

    u.vect = _mm256_mul_pd(u.vect, v.vect);
    u.vect = _mm256_addsub_pd(u.vect, _mm256_permute_pd(u.vect, 0b0101));
    zr = u.vect[0];
    u.vect = _mm256_mul_pd(u.vect, u.vect);
    u.vect = _mm256_permute_pd(u.vect, _MM_SHUFFLE(3, 2, 1, 0));
    u.vect = _mm256_add_pd(u.vect, _mm256_permute2f128_pd(u.vect, u.vect, 1));
    u.vect = _mm256_sqrt_pd(u.vect);

    *result = asin(zr / u.vect[0]);
}

static void calcVect(void *args, double *result) {

    struct complexArgs *cargs = args;

    double zr;
    double r;

    union vect {
        __m128i vect;
    };

    union vect u1 = {cargs->ar, cargs->ar};
    union vect v1 = {cargs->br, cargs->bj};
    union vect u0 = {cargs->aj, cargs->br};
    union vect v0 = {cargs->bj, cargs->aj};

    v1.vect = _mm_mul_epi32(u1.vect, v1.vect); // => {ar*br, ar*bj}
    v0.vect = _mm_mul_epi32(u0.vect, v0.vect); // => {aj*bj, br*aj}

    u1.vect = _mm_sub_epi64(v1.vect,    // => {ar*br - aj*bj, ar*bj - br*aj}
                            v0.vect);   // *we don't care about the second entry

    u0.vect = _mm_add_epi32(v1.vect,    // => {ar*br + aj*bj, ar*bj + br*aj}
                            v0.vect);   // *we don't care about the first entry

    zr = u1.vect[0];

    u0.vect = _mm_mul_epi32(u0.vect, u0.vect);
    u1.vect = _mm_mul_epi32(u1.vect, u1.vect);
    u0.vect = _mm_add_epi32(u0.vect,
                            _mm_shuffle_epi32(u1.vect, _MM_SHUFFLE(1, 0, 1, 0)));

    r = sqrt(u0.vect[1]);
    *result = asin(zr / r);
}

static void calcAcos(void *args, double *result) {

    struct complexArgs *cargs = args;

    int vr = cargs->ar * cargs->br - cargs->aj * cargs->bj;
    int vj = cargs->aj * cargs->br + cargs->ar * cargs->bj;
    double vnorm = norm(vr, vj);

    *result = asin(vr / vnorm);
}

static void calcAtan2(void *args, double *result) {

    struct complexArgs *cargs = args;

    int vr = cargs->ar * cargs->br - cargs->aj * cargs->bj;
    int vj = cargs->aj * cargs->br + cargs->ar * cargs->bj;

    *result = atan2((double) vr, (double) vj);
}

static void mult(int ar, int aj, int br, int bj/*, int *vr, int *vj*/) {

    static long double maxAvgErr = -1.0;
    long double avgError;
    struct complexArgs cargs;
    double results[TIMING_RUNS];

    cargs.aj = aj;
    cargs.ar = ar;
    cargs.br = br;
    cargs.bj = bj;

    timeFun((timedFun) calcAtan2, &cargs, (void *) &results, 0);
    timeFun((timedFun) calcAcos, &cargs, (void *)&results, 1);
    timeFun((timedFun) calcVect, &cargs, (void *)&results, 2);
    timeFun((timedFun) calcVectPd, &cargs, (void *)&results, 3);

    long double errs[4] = {results[0]-results[1],
                           results[0]-results[2],
                           results[0]-results[3]};

    if (DEBUG) {
        avgError = errs[0];
        avgError *= errs[1];
        avgError *= errs[2];
        avgError = cbrtl(avgError);
//    avgError += errs[1];
//    avgError += errs[2];
//    avgError /= 4.0;
        maxAvgErr = avgError > maxAvgErr ? avgError : maxAvgErr;

        printf("(%d + %di) . (%d + %di) = (%d + %di) angle: %f, %f, %f, %f "
               "max avg error: %Le\n",
               ar, aj,
               br, bj,
               ar * br - aj * bj,
               aj * br + ar * bj,
               results[0], results[1], results[2], results[3],
               maxAvgErr);
    }
    assert(errs[0] < 1e-14 && errs[1] < 1e-14 && errs[2] < 1e-14);
}

int main(const int argc, const char **argv) {

    if (argc > 1) {
        DEBUG = 'v' == argv[1][0];
    }

    int i,j,k,m;
    for (i = 1; i < 32; ++i)
        for (j = 1; j < 32; ++j)
            for (k = 1; k < 32; ++k)
                for (m = 1; m < 32; ++m)
                    mult(i,j,k,m);

    mult((int16_t) 1, (int16_t) 2, (int16_t) 3, (int16_t) 4);
    mult(5, 6, 7, 8);
    mult(9, 10, 11, 12);

    char *runNames[TIMING_RUNS]
            = {"calcAtan2 :: ", "calcAcos :: ", "calcVect :: ", "caclVectPd :: "};
    printTimedRuns(runNames, TIMING_RUNS);

    return 0;
}
