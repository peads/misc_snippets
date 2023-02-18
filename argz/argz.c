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
//#define DEBUG
#define TIMING_RUNS 3
#include <immintrin.h>
#include "timed_functions.h"

#define MIN -10
#define MAX 10
#define STEP 1
#define MAX_ERROR 1e-9
#ifdef DEBUG
    #define ASSERT(b) assert(b)
#else
    #define ASSERT(b) //
#endif

char *runNames[TIMING_RUNS] = {"vect1 :: ", "vect2 :: ", "polar_discriminant :: "};

typedef double (*argzFun)(int a, int b, int c, int d);

struct argzArgs {
    int ar;
    int aj;
    int br;
    int bj;
    argzFun fun;
};

static double polar_discriminant(int ar, int aj, int br, int bj) {

    int cr, cj;

    cr = ar * br - aj * bj;
    cj = aj * br + ar * bj;

    return atan2((double) cj, (double) cr);
}

static double calcVectPd(int ar, int aj, int br, int bj) {

    union vect {
        __m256d vect;
    };

    union vect u = {ar, aj, ar, br};
    union vect v = {br, bj, bj, aj};
//    __m256d temp;
    double zr, zj;

    u.vect = _mm256_mul_pd(u.vect, v.vect);                  // ar*br, aj*bj, ar*bj, br*aj
//    temp = _mm256_permute_pd(u.vect, 0b0101);                    // aj*bj, ar*br, br*aj, ar*bj
    u.vect = _mm256_addsub_pd(u.vect,
                            _mm256_permute_pd(u.vect, 0b0101)); // ar*ar - aj*bj, aj*bj + ar*br, ar*bj - br*aj, br*aj + ar*bj
    zr = u.vect[0];                                                // ar*ar - aj*bj
    zj = u.vect[3];                                                // br*aj + ar*bj
    if (0.0 == zr && 0.0 == zj) {
        return NAN;
    }

    u.vect = _mm256_mul_pd(u.vect, u.vect);                  // (ar*ar - aj*bj)^2, (aj*aj + ar*br)^2, (ar*bj - br*aj)^2, (br*aj + ar*bj)^2
//    temp = _mm256_permute_pd(u.vect, _MM_SHUFFLE(3, 2, 1, 0));   // (ar*ar - aj*bj)^2, (ar*ar - aj*bj)^2, (br*aj + ar*bj)^2, (ar*bj - br*aj)^2
//    v.vect = _mm256_permute2f128_pd(u.vect, u.vect, 1);          // (ar*bj - br*aj)^2, (br*aj + ar*bj)^2, (ar*ar - aj*bj)^2, (ar*bj - br*aj)^2
//    u.vect = _mm256_add_pd(temp, v.vect);                        // u[0] + u[2], u[0] + u[3], u[3] + u[0], u[2] + u[1]
    u.vect = _mm256_permute_pd(u.vect, _MM_SHUFFLE(3, 2, 1, 0));
    u.vect = _mm256_add_pd(u.vect, _mm256_permute2f128_pd(u.vect, u.vect, 1));
    u.vect = _mm256_sqrt_pd(u.vect);                            // (u[0] + u[2])^1/2, (u[0] + u[3])^1/2, (u[3] + u[0])^1/2, (u[2] + u[1])^1/2

#ifndef DEBUG
    return zj < 0 ? -acos(zr / u.vect[0]) : acos(zr / u.vect[0]);
#else
    static uint64_t errCnt = 0;
    double result = zj < 0 ? -acos(zr / u.vect[0]) : acos(zr / u.vect[0]);                         // ((ar*ar - aj*bj)^2 + (br*aj + ar*bj)^2)^1/2

    int mulr = ar * br - aj * bj;
    int mulj = ar * bj + br * aj;
    ASSERT(mulr == zr);
    ASSERT(mulj == zj);
//    multiply(ar, aj, br, bj, &mulr, &mulj);
    double phase = atan2(mulj, mulr);

    long double delta = fabsl(fabsl(result) - fabsl(phase));
    int isWrong = delta >= MAX_ERROR;

    if (isWrong) {
        printf("%s a := (%d + %di), b := (%d + %di)\n", runNames[0], ar, aj, br, bj);
        printf("%s a := (%d + %di), b := (%d + %di)\n", runNames[0], ar, aj, br, bj);
        printf("%llu - Expected phase: %f Got phase: %f Delta: %Lf\n", errCnt++, phase, result, delta);
    }
    ASSERT(!isWrong);

    return result;
#endif
}

static double calcVect(int ar, int aj, int br, int bj) {

    double zr, zj;

    union vect128 {
        __m128d vect;
    };

    union vect128 u1 = {ar, ar};
    union vect128 v1 = {br, bj};
    union vect128 u0 = {aj, br};
    union vect128 v0 = {bj, aj};
    __m128d u1a;

    v1.vect = _mm_mul_pd(u1.vect, v1.vect); // => {ar*br, ar*bj}
    v0.vect = _mm_mul_pd(u0.vect, v0.vect); // => {aj*bj, br*aj}

    u1.vect = _mm_sub_pd(v1.vect,    // => {ar*br - aj*bj, ar*bj - br*aj}
                            v0.vect);   // *we don't care about the second entry

    u0.vect = _mm_add_pd(v1.vect,    // => {ar*br + aj*bj, ar*bj + br*aj}
                            v0.vect);   // *we don't care about the first entry

    zr = u1.vect[0];
    zj = u0.vect[1];

    u0.vect = _mm_mul_pd(u0.vect, u0.vect);
    u1.vect = _mm_mul_pd(u1.vect, u1.vect);
    u1a = _mm_set_pd(u1.vect[0], u1.vect[1]);
    u0.vect = _mm_add_pd(u0.vect, u1a);
    u0.vect = _mm_sqrt_pd(u0.vect);

#ifndef DEBUG
    return zj < 0 ? -acos(zr / u0.vect[1]) : acos(zr / u0.vect[1]);
#else
    ASSERT(mulr == zr);
    ASSERT(mulj == zj);

    int mulr = ar * br - aj * bj;
    int mulj = ar * bj + br * aj;
    double phase = atan2(mulj, mulr);
    double result = zj < 0 ? -acos(zr / u0.vect[1]) : acos(zr / u0.vect[1]);
    long double delta = fabsl(fabsl(result) - fabsl(phase));
    int isWrong = delta >= MAX_ERROR;

    ASSERT(!isWrong);
    return result;
#endif
}

void runTest(void *arg, double *result) {

    struct argzArgs *bargs = arg;

    *result = bargs->fun(bargs->ar, bargs->aj, bargs->br, bargs->bj);
}

void testIteration(struct argzArgs *args, void *results, int runIndex) {

    double mulr = args->ar * args->br - args->aj * args->bj;
    double mulj = args->ar * args->bj + args->br * args->aj;
    double phase = atan2(mulj, mulr);

    timeFun((timedFun) runTest, args, results, runIndex);

    double result = ((double *) results)[runIndex];
    long double delta = fabsl(result - phase);
    int isWrong = delta >= MAX_ERROR;

#ifdef DEBUG
    if (!runIndex && isWrong) {
        printf("%-25s a := (%d + %di), b := (%d +%di), Got: %f ",
               runNames[runIndex], args->ar, args->aj, args->br, args->bj, result);
        printf("Expected: %f Delta: %Lf\n", phase, delta);
        printf("Signedness ar: %d aj: %d br: %d bj: %d\n", args->ar, args->aj, args->br, args->bj);//, mulr, mulj);
    }
#endif
    assert(!isWrong);
}

extern asmArgz(int xr, int xj, int yr, int yj);
__asm__(
#ifdef __APPLE_CC__
"_asmArgz: "
#else
"asmArgz: "
#endif
    "movsldup xmm0, src1\n\t"
);

int main(void) {

    int m, k, j, i;
    struct argzArgs args;
    double results[TIMING_RUNS];

    args.bj = args.br = args.ar = args.aj = MIN;
    args.ar = 1;
    args.br = 3;
    args.aj = 2;
    args.fun = calcVectPd;
    args.bj = 4;

    calcVectPd(1,2,3,4);
    calcVect(1,2,3,4);

    for (i = MIN; i < MAX; ++i) {
        args.ar = i;
        for (j = MIN; j < MAX; ++j) {
            args.aj = j;
            for (k = MIN; k < MAX; ++k) {
                args.br = k;
                for (m = MIN; m < MAX; ++m) {
                    args.bj = m;

                    args.fun = calcVectPd;
                    testIteration(&args, results, 0);

                    args.fun = calcVect;
                    testIteration(&args, results, 1);

                    args.fun = polar_discriminant;
                    testIteration(&args, results, 2);
//
//                    args.fun = esbensen;
//                    testIteration(&args, results, 3);
                }
            }
        }
    }

    printTimedRuns(runNames, TIMING_RUNS);
}
