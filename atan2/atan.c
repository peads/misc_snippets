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
#include "timed_functions.h"
#ifdef __ICC
#include <immintrin.h>
#endif

//#define DEBUG
#define MAX_ERROR 0.1f
#define MIN -1.f
#define MAX 1.f
#define STEP 0.1f

/**
 * Takes packed float representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns their argument as a float
 **/
extern float argz(__m128 a);
__asm__(
#ifdef __clang__
"_argz: "
#else
"argz: "
#endif
    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm3\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm3, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vrsqrtps %xmm1, %xmm1\n\t"             // ..., Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    "comiss %xmm0, %xmm1\n\t"
    "jp zero\n\t"

    // push
//    "vextractps $1, %xmm0, -8(%rsp) \n\t"
    "vpermilps $0x01, %xmm0, %xmm2\n\t"
    "vmovq %xmm2, -8(%rsp)\n\t"
    "flds -8(%rsp) \n\t"

    // push
//    "vextractps $2, %xmm0, -8(%rsp) \n\t"
    "vpermilps $0x02, %xmm0, %xmm2\n\t"
    "vmovq %xmm2, -8(%rsp)\n\t"
    "flds -8(%rsp) \n\t"

    "fpatan \n\t"

    // pop and return
    "fstps -8(%rsp) \n\t"
    "vmovq -8(%rsp), %xmm0 \n\t"
    "ret\n\t"

"zero: "
    "vpxor %xmm0, %xmm0, %xmm0\n\t"
    "ret"
);

/**
 * Takes packed float representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns their argument as a float
 **/
extern float argzB(__m128 a);
__asm__(

".section:\n\t"
".p2align 4\n\t"
"LC0: "
    ".quad 4791830004637892608\n\t"
"LC1: "
    ".quad 4735535009282654208\n\t"
"LC2: "
    ".quad 4765934306774482944\n\t"
"LC3: "
    ".quad 0x40490fdb\n\t"
".text\n\t"

#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif
    "vpxor %xmm3, %xmm3, %xmm3\n\t"         // store zero
    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm2\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm2, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    "vpermilps $0x01, %xmm0, %xmm2\n\t"
    "vcomiss %xmm2, %xmm3\n\t"
    "jnz showtime\n\t"                      // zj != 0, do the fun math
    "vpermilps $0x02, %xmm0, %xmm2\n\t"
    "vcomiss %xmm3, %xmm2\n\t"
    "jz zed\n\t"                            // zr == zj == 0, return 0 (should properly be undefined,
                                            // but this is inline with how library atan2f functions)
    "ja showtime\n\t"                       // zr > 0, fun math
    "vmovq LC3(%rip), %xmm0\n\t"            // zr < 0 && zj == 0, return Pi
    "ret \n\t"

"zed: "
//    "vpxor %xmm0,%xmm0,%xmm0\n\t"
    "vmovq %xmm3,%xmm0\n\t"
    "ret\n\t"

"showtime: "                                // approximating atan2 with atan(z)
                                            //   = z/(1 + (9/32) z^2) for z = (64 y)/(23 x + 41 Sqrt[x^2 + y^2])
                                            //   and the edge cases (i.e. where it is not the case that y != 0 || x > 0)
                                            //   taken care of above.
    "vrsqrtps %xmm1, %xmm1\n\t"             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "movddup LC0(%rip), %xmm2\n\t"          // 64
    "movddup LC1(%rip), %xmm3\n\t"          // 23

    "vmulps %xmm2, %xmm0, %xmm2\n\t"        // 64*zj
    "vmulps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr
    "movddup LC2(%rip), %xmm0\n\t"          // 41
    "vaddps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr + 41
    "vpermilps $0x1B, %xmm3, %xmm3\n\t"
    "vrcpps %xmm3, %xmm3\n\t"
    "vmulps %xmm3, %xmm2, %xmm0\n\t"

    "vpermilps $0x01, %xmm0, %xmm0\n\t"
    "ret\n\t"
);

static inline void multiply(__m128 z, float *zr, float *zj) {
    union vect temp = {.vect = z};
    *zr = temp.arr[0] * temp.arr[2] - temp.arr[1] * temp.arr[3];
    *zj = temp.arr[0] * temp.arr[3] + temp.arr[1] * temp.arr[2];
}

static inline void printData(__m128 z, float zr, float zj, float theta, float phi, float omega) {
    union vect vect = {.vect = z};
    printf("(%g + %gI).(%g + %gI) = (%g + %gI)"
           "\nPhase from atan2f: %f\nPhase from argz: %f\nPhase from argzB: %f\n",
           vect.arr[0], vect.arr[1], vect.arr[2], vect.arr[3],
           zr, zj, theta, phi, omega);
}

#ifdef __ICC
static inline __m128 argzC(__m128 z) {

    z = _mm_mul_ps(_mm_permute_ps(z, 0x5), _mm_permute_ps(z, 0xEB));
    z = _mm_addsub_ps(z, _mm_permute_ps(z, 0x8D));
    return _mm_atan2_ps(z, _mm_permute_ps(z, 0x1B));
}
#endif

static void baseCases() {
    
    int i;
    float zr, zj, theta, phi, omega;
    __m128 z;
    __m128 Z[10] = {{1,2,3,4},{4,3,2,1},{-5,-10,1,0},{5,-10,1,0},{0,0,0,0},
                   {FLT_EPSILON,FLT_EPSILON,FLT_EPSILON,FLT_EPSILON},
                   {1,1,1,1}, {-1,1,1,1}, {1,-1,1,1}, {1,-1,1,-1}};

    for (i = 0; i < sizeof(Z)/sizeof(*Z); ++i) {
        z = Z[i];
        multiply(z, &zr, &zj);
        theta = atan2f(zj, zr);
        phi = argz(z);
        omega = argzB(z);

        printData(z, zr, zj, theta, phi, omega);
    }
    printf("\n");
}

static inline float /*__attribute__((noinline))*/ arg(__m128 z) {

    float zr, zj;
    multiply(z, &zr, &zj);
    return atan2f(zj, zr);
}

int main(void) {

    char *runNames[TIMING_RUNS] = {"atan2f :: ", "argz :: ", "argzB :: ",
#ifdef __ICC
                                   "_mm_atan2_ps :: "};
    __m128 tmp;
#else
                                    "n/a; not using ICC :: "};
#endif

    float ar = MIN;
    float aj = MIN;
    float br = MIN;
    float bj = MIN;
    float sum = 0.f;
    float sd = 0.f;
    float mu, theta, phi, omega, delta;
    uint64_t i, n;
    uint64_t N = ceil(pow((MAX - MIN)/STEP + 1, 4.));
    float *xs = calloc(N, sizeof(float));
    struct timespec tstart, tend;
    __m128 z;
#ifdef DEBUG
    uint64_t onePercent = (uint64_t)(.01f * N);
    float zr;
    float zj;
#endif

    baseCases();

    for (i = 0; ar <= MAX; ++i) {

        union vect temp = {ar, aj, br, bj};
        z = temp.vect;
#ifdef DEBUG
        multiply(z, &zr, &zj);
#endif

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        phi = argz(z);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(1, &tstart, &tend);
        // END TIMED
        assert(!isnan(phi));

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        omega = argzB(z);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(2, &tstart, &tend);
        // END TIMED

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        theta = arg(z);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(0, &tstart, &tend);
        // END TIMED

#ifdef __ICC
        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        tmp = argzC(z);
        theta = (union vect){.vect = tmp}.arr[1];

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(3, &tstart, &tend);
        // END TIMED
#endif

        assert(!isnan(theta));

#ifdef DEBUG
        if (isnan(omega)) {
            printData(z, zr, zj, theta, phi, omega);
        }
#endif
        assert(!isnan(omega));

        delta = fabsf(theta) - fabsf(omega);
        sum += delta;
        xs[i] = delta;

        bj+=STEP;
        if (bj > MAX) {
            bj = MIN;
            br += STEP;
        }
        if (br > MAX) {
            br = MIN;
            aj += STEP;
        }
        if (aj > MAX) {
            aj = MIN;
            ar += STEP;
        }
#ifdef DEBUG
        if (!(i % onePercent)) {
            printf("%g%%\n", 100.f*i/N);
        }
#endif
    }
    printf("\n");

    mu = sum/i;
    for (n = 0; n < i; n+=2) {
        sd += powf((xs[n] - mu), 2.f) + powf((xs[n + 1] - mu), 2.f);
    }
    sd = sqrtf(sd / i);

    free(xs);

    printf("N: %llu mu: %f sd: %f\n\n", i, mu, sd);

    printTimedRuns(runNames, TIMING_RUNS);

    return 0;
}
