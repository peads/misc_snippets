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

#define MAX_ITERATIONS 128
#define MAX_ERROR 1e-12

uint8_t DEBUG = 0;

struct complexArgs {
    double ar;
    double aj;
    double br;
    double bj;
};

union vect {
    __m256d vect;
};

const char *runNames[TIMING_RUNS]
        = {"calcAtan2 :: ", "calcAsin :: ", 
            "calcVect :: ", "caclVectPd :: "};//, "esbensen :: "};

static double norm(double x, double y) {

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

static void calcAsin(void *args, double *result) {

    struct complexArgs *cargs = args;

    double vr = cargs->ar * cargs->br - cargs->aj * cargs->bj;
    double vj = cargs->aj * cargs->br + cargs->ar * cargs->bj;
    double vnorm = norm(vr, vj);

    *result = asin(vr / vnorm);
}

static void calcAtan2(void *args, double *result) {

    struct complexArgs *cargs = args;

    double vr = cargs->ar * cargs->br - cargs->aj * cargs->bj;
    double vj = cargs->aj * cargs->br + cargs->ar * cargs->bj;

    *result = atan2((double) vr, (double) vj);
}

void esbensen(void *args, double *result)
/*
  input signal: s(t) = a*exp(-i*w*t+p)
  a = amplitude, w = angular freq, p = phase difference
  solve w
  s' = -i(w)*a*exp(-i*w*t+p)
  s'*conj(s) = -i*w*a*a
  s'*conj(s) / |s|^2 = -i*w
*/
{

    const int32_t scaled_pi = 2608;// 1<<14 / (2*pi) */
    struct complexArgs *cargs = args;

    int cj, dr, dj;
    int br = (int) cargs->br;
    int ar = (int) cargs->ar;
    int bj = (int) cargs->bj;
    int aj = (int) cargs->aj;

    dr = (br - ar) << 1;
    dj = (bj - aj) << 1;
    cj = bj * dr - br * dj; /* imag(ds*conj(s)) */

    int32_t denom = ar*ar + aj*aj + 1;
    printf("dr = %d dj = %d cj = %d dividing %d * %d / %d\n", dr, dj, cj, scaled_pi, cj, denom);
    double d = (double) cj / denom;

    *result = sin(scaled_pi * d) / 2.0;
}

struct {
    long double maxAvgErr;
    char *name;
} errorInfo;

uint64_t totalErrors = 0;
static void mult(double ar, double aj, double br, double bj/*, int *vr, int *vj*/) {

    long double avgError;
    long double localMax; 
    struct complexArgs cargs;
    double results[TIMING_RUNS];
    uint32_t i, localMaxIdx;

    cargs.aj = aj;
    cargs.ar = ar;
    cargs.br = br;
    cargs.bj = bj;

    timeFun((timedFun) calcAtan2, &cargs, (void *) &results, 0);
    timeFun((timedFun) calcAsin, &cargs, (void *)&results, 1);
    timeFun((timedFun) calcVect, &cargs, (void *)&results, 2);
    timeFun((timedFun) calcVectPd, &cargs, (void *)&results, 3);
//    timeFun((timedFun) esbensen, &cargs, (void *)&results, 4);

    long double absRes = fabs(results[0]);
    long double errs[3] = {absRes-fabs(results[1]),
                           absRes-fabs(results[2]),
                           absRes-fabs(results[3]),
                           };//results[0]-results[4]};

    avgError = errs[0];
    avgError *= errs[1];
    avgError *= errs[2];
    avgError = cbrtl(avgError);
//    avgError += errs[1];
//    avgError += errs[2];
//    avgError /= 4.0;
      //  maxAvgErr = avgError > maxAvgErr ? avgError : maxAvgErr;
    if (avgError > errorInfo.maxAvgErr) {
        errorInfo.maxAvgErr = avgError;
        for (localMaxIdx = -1, localMax = -1.0, i = 0; i < 3; ++i) {
            if (errs[i] > localMax) {
                localMax = errs[i];
                errorInfo.name = runNames[i+1];
            }
        }
    }

    if (DEBUG 
        || errs[0] != 0.0 && errs[0] >= MAX_ERROR 
        || errs[1] != 0.0 && errs[1] >= MAX_ERROR 
        || errs[2] != 0.0 && errs[2] >= MAX_ERROR) {
        printf("(%f + %fi) . (%f + %fi) = (%f + %fi) angle: %f, %f, %f, %f "
               "err1: %Le, err2: %Le, err3: %Le cause: %s\n",
               ar, aj,
               br, bj,
               ar * br - aj * bj,
               aj * br + ar * bj,
               results[0], results[1], results[2], results[3], //results[4],
               errs[0], errs[1], errs[2], errorInfo.name);
        totalErrors++;
    }

//    assert(errs[0] < MAX_ERROR && errs[1] < MAX_ERROR && errs[2] < MAX_ERROR);
}

int main(const int argc, const char **argv) {

    if (argc > 1) {
        DEBUG = 'v' == argv[1][0];
    }
    errorInfo.maxAvgErr = -1.0;
    uint32_t i,j,k,m;
    double f,n,p,q;
    for (i = 1, f = -10.0; i < MAX_ITERATIONS; ++i, f += 0.01) {
        for (j = 1, n = -10.0; j < MAX_ITERATIONS; ++j, n += 0.01) {
            for (k = 1, p = -10.0; k < MAX_ITERATIONS; ++k, p += 0.01) {
                for (m = 1, q = -10.0; m < MAX_ITERATIONS; ++m, p += 0.01) {
                    if (0.0 == f || 0.0 == n) continue;
                    //mult(i,j,k,m);
                    mult(f,n,p,q);
                }
            }
        }
    }

    mult((int16_t) 1, (int16_t) 2, (int16_t) 3, (int16_t) 4);
    mult(5, 6, 7, 8);
    mult(9, 10, 11, 12);

    printTimedRuns(runNames, TIMING_RUNS);
    printf("Worst: %s, Max avg error: %Le, total above threshold: %lu, total calculations: %lu\n", 
        errorInfo.name, errorInfo.maxAvgErr, totalErrors, MAX_ITERATIONS*MAX_ITERATIONS*MAX_ITERATIONS*MAX_ITERATIONS);
    return 0;
}
