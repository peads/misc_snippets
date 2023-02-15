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

#define TIMING_RUNS 4
#include <immintrin.h>
#include "timed_functions.h"

#define MIN -2
#define MAX 2
#define STEP 1
#define MAX_ERROR 1e-1
#ifdef DEBUG
    #define ASSERT(b) assert(b)
#else
    #define ASSERT(b) //
#endif

char *runNames[TIMING_RUNS] = {"vect1 :: ", "polar_discriminant :: ", "polar_disc_fast :: ",
                               "esbensen :: " };

//typedef int (*argzFun)(int a, int b, int c, int d);
typedef double (*argzFun)(int a, int b, int c, int d);

struct argzArgs {
    int ar;
    int aj;
    int br;
    int bj;
    argzFun fun;
};

//extern int bsr(uint64_t i);
//__asm__ (
//#ifdef __APPLE_CC__
//"_bsr: "
//#else
//"bsr: "
//#endif
//    "movq $-1023, %rdx\n\t" // store 0 to compare against ZF later
//    "bsr %rdi, %rax\n\t"
//    "cmovzq %rdx, %rax\n\t"
//    "ret"
//);
//
//extern uint64_t lzcnt(uint64_t i);
//__asm__ (
//#ifdef __APPLE_CC__
//"_lzcnt: "
//    #else
//"lzcnt: "
//    #endif
//    "movq $0, %rdx\n\t" // store 0 to return if CF set later
//    "lzcnt %rdi, %rax\n\t"
//    "cmovcq %rdx, %rax\n\t"
//    "ret"
//);

//extern int bsrLzcnt(uint64_t i);
//__asm__ (
//#ifdef __APPLE_CC__
//"_bsrLzcnt: "
//    #else
//"bsrLzcnt: "
//    #endif
//    "lzcnt %rdi, %rdx\n\t"
//    "jc zero\n\t" // jump to return -1023 if input was 0
//    "movq $63, %rax\n\t"
//    "subq %rdx, %rax\n\t"
//    "ret\n\t"
//"zero:\n\t"
//    "movq $-1023, %rax\n\t"
//    "ret"
//);

static inline void multiply(int ar, int aj, int br, int bj, int *cr, int *cj) {

    *cr = ar * br - aj * bj;
    *cj = aj * br + ar * bj;
}

static double fast_atan2(double y, double x)
/* pre scaled for int16 */
{

    double yabs, angle;
//    int pi4 = (1 << 12), pi34 = 3 * (1 << 12);  /* note pi = 1<<14 */
    if (x == 0 && y == 0) {
        return 0;
    }
    yabs = y;
    if (yabs < 0) {
        yabs = -yabs;
    }
    if (x >= 0) {
        angle = M_PI_2 - (x - yabs) / (x + yabs) + 1.0;
    } else {
        angle = (y < 0.0 ? M_PI_2 : -M_PI_2) - (x + yabs) / (yabs - x) + 1.0;
    }
    if (y < 0.0) {
        return -angle;
    }
    return angle;
}

static double polar_disc_fast(int ar, int aj, int br, int bj) {

    int cr, cj;
    multiply(ar, aj, br, -bj, &cr, &cj);
    return fast_atan2(cj, cr);
}

static double polar_discriminant(int ar, int aj, int br, int bj) {

    int cr, cj;
    multiply(ar, aj, br, bj, &cr, &cj);
    return atan2((double) cj, (double) cr);
}

static double esbensen(int ar, int aj, int br, int bj)
/*
  input signal: s(t) = a*exp(-i*w*t+p)
  a = amplitude, w = angular freq, p = phase difference
  solve w
  s' = -i(w)*a*exp(-i*w*t+p)
  s'*conj(s) = -i*w*a*a
  s'*conj(s) / |s|^2 = -i*w
*/
{

    int cj, dr, dj;
    int scaled_pi = 2608; /* 1<<14 / (2*pi) */
    dr = (br - ar) * 2;
    dj = (bj - aj) * 2;
    cj = bj * dr - br * dj; /* imag(ds*conj(s)) */
    return (scaled_pi * cj / (ar * ar + aj * aj + 1));
}

static double calcVectPd(int ar, int aj, int br, int bj) {

    union vect {
        __m256d vect;
    };

    union vect u = {ar, aj, ar, br};
    union vect v = {br, bj, bj, aj};                               // => ar*br, aj*bj, ar*bj, br*aj
//    __m256d temp;
    double zr, zj;

    u.vect = _mm256_mul_pd(u.vect, v.vect);
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
    return acos(zr / u.vect[0]);
#else
    static uint64_t errCnt = 0;
    double result =  acos(zr / u.vect[0]);                          // ((ar*ar - aj*bj)^2 + (br*aj + ar*bj)^2)^1/2

    int mulr = ar * br - aj * bj;
    int mulj = ar * bj + br * aj;
    ASSERT(mulr == zr);
    ASSERT(mulj == zj);
//    multiply(ar, aj, br, bj, &mulr, &mulj);
    double phase = atan2(mulj, mulr);

    long double delta = fabsl(fabsl(result) - fabsl(phase));
    int isWrong = delta >= 1e-15L;
    if (isWrong) {
        printf("%s a := (%d + %di), b := (%d + %di)\n", runNames[3], ar, aj, br, bj);
        printf("%s a := (%d + %di), b := (%d + %di)\n", runNames[3], ar, aj, br, bj);
        printf("%llu - Expected phase: %f Got phase: %f Delta: %Lf\n", errCnt++, phase, result, delta);
    }
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

    long double delta = fabsl(fabsl(result) - fabsl(phase));
    int isWrong = delta >= 1e-15L;
    if (isWrong) {
        printf("%-25s a := (%d + %di), b := (%d +%di), phase := %f\n",
               runNames[runIndex], args->ar, args->aj, args->br, args->bj, result);
        printf("Expected phase: %f\n", phase);
    }
}

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
    testIteration(&args, results, 0);

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

                    args.fun = polar_discriminant;
                    testIteration(&args, results, 1);

                    args.fun = polar_disc_fast;
                    testIteration(&args, results, 2);

                    args.fun = esbensen;
                    testIteration(&args, results, 3);
                }
            }
        }
    }

    printTimedRuns(runNames, 4);
}
