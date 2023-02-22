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
#include <immintrin.h>
#define TIMING_RUNS 2
#include "timed_functions.h"

#define DEBUG

#ifdef DEBUG
    #define PRINTF printf
#else
    #define PRINTF //
#endif

#define MAX_ERROR 10e-3
#define MIN -200
#define MAX 200
#define STEP 1
//#define DEBUG

union vect {
    float arr[4] __attribute__((aligned(16)));
    uint64_t l;
    __m128 vect;
};

typedef void (*atanFun)(const float y, const float x, float *__restrict__ result);

struct atanArgs {
    atanFun fun;
    float x;
    float y;
};
static const float oneOverRootFive = 0.4472135954999579392818347337462552470881236719223051448541f;
static const float twoOverRootFive = 2.f * oneOverRootFive;

extern float ffabsf(float f);
extern int isNegZero(float f);
extern float fsqrtf(float x);

float aacos(float x) {

    int negate = signum(x);
    x = fabs(x);

    float result = -0.0187293;
    result *= x;
    result += 0.0742610;
    result *= x;
    result -= 0.2121144;
    result *= x;
    result += M_PI_2;
    result *= fsqrtf(1.f - x);
    result -= 2 * negate * result;
    return negate * M_PI + result;
}

float aatan3(float x) {

    return aacos(2*x / (1+x*x));
}

extern __m128 argz(__m128 *z);
__asm__(
#ifdef __clang__
"_argz: "
#else
"argz: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"
    "and $-16, %rsp\n\t"

    "vmovaps (%rdi), %xmm0\n\t"
    "vpermilps $0x0, %xmm0, %xmm2\n\t"  // {x, x, x, x}
    "vpermilps $0xFF, %xmm0, %xmm3\n\t" // {y, y, y, y}

    "xorq %rdx, %rdx\n\t"
    "movq %xmm2, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz xynz\n\t"                      // if x != 0 continue on

    "movq %xmm3, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz xynz\n\t"                      // if y also != 0 continue on
    "pcmpeqd %xmm0, %xmm0\n\t"
    "vbroadcastss %xmm0, %xmm0\n\t"
    "jmp return\n\t"                    // otherwise, bye, felicia (return NaN)

"xynz: "
    "vmulps %xmm0, %xmm0, %xmm0\n\t"    // x*x, x*x, y*y, y*y
    "vpermilps $0x1B, %xmm0, %xmm1\n\t"
    "vaddps %xmm1, %xmm0, %xmm0\n\t"    // x*x + y*y, ...
//    "vmovaps %xmm1, %xmm4\n\t"
    "vsqrtps %xmm1, %xmm1\n\t"          // Sqrt[x*x + y*y], ...
    "vdivps %xmm1, %xmm0, %xmm0\n\t"
    "vdivps %xmm1, %xmm2, %xmm2\n\t"
    "vaddsubps %xmm2, %xmm1, %xmm0\n\t" // Sqrt[...] +- x alternately, avoids an extra instruction
    "vmovaps %xmm1, (%rdi)\n\t"         // store in return argument

    "vxorps %xmm1, %xmm1, %xmm1\n\t"
    "vcmpless %xmm1, %xmm2, %xmm1\n\t"
    "movq %xmm1, %rcx\n\t"
    "jrcxz posx\n\t"                    // if (x > 0) goto .posx

    "movq %xmm3, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz ynz\n\t"                       // if y also != 0 continue on
    "movl $0x40490fdb, %edx\n\t"
    "movq %rdx, %xmm0\n\t"
    "vbroadcastss %xmm0, %xmm0\n\t"
    "jmp return\n\t"                    // otherwise return Pi

"ynz: "
    "vdivps %xmm0, %xmm3, %xmm0\n\t"    // y / (Sqrt[...] + x), ...
    "jmp homestretch\n\t"

"posx: "
    "vdivps %xmm3, %xmm0, %xmm0\n\t"    // (Sqrt[...] - x) / y, ...
    "jmp homestretch\n\t"

"homestretch: "
/* Approx via Pi/4 */
//    "movl $0x3f490fdb, %edx\n"
//    "vmovq %rdx, %xmm2\n\t"
//    "vbroadcastss %xmm2, %xmm2\n\t"
//    "vmulps %xmm2, %xmm0, %xmm0\n\t"

/* Approximate via
 * https://www-labs.iro.umontreal.ca/~mignotte/IFT2425/Documents/EfficientApproximationArctgFunction.pdf (11) */
    "movq $0x42000000, %rdx\n\t"
    "vmovq %rdx, %xmm2\n\t"
    "vbroadcastss %xmm2, %xmm2\n\t"         // 32, 32, 32, 32 -> xmm2
    "movq $0x41100000, %rdx\n\t"
    "vmovq %rdx, %xmm3\n\t"
    "vbroadcastss %xmm3, %xmm3\n\t"         // 9, 9, 9, 9 -> xmm3
    "vmulps %xmm0, %xmm0, %xmm1\n\t "       // x*x, x*x, x*x, x*x -> xmm1
    "vmulps %xmm3, %xmm1, %xmm1\n\t"        // 9xx ,.. -> xmm1
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // 9xx + 32, ... -> xmm1
    "vmulps %xmm2, %xmm0, %xmm0\n\t"        // 32x, ... -> xmm0
    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // 32x/(9xx + 32), ... -> xmm0

/* Cheat */
//    "vextractps $1, %xmm0, %rdx\n\t" // TODO replace this with function call to one that takes a vector
//    "xorps %xmm0, %xmm0\n\t"
//    "movq %rdx, %xmm0\n\t"
//    "sub $128, %rsp\n\t"
//    "mov %rsp, %rdx\n\t"
//    "call _aatan\n\t"               // TODO replace this with vector ops implementing aatan
//    "mov %rdx, %rsp\n\t"
//    "add $128, %rsp\n\t"
//
    "movq $0x40000000, %rdx\n\t"
    "vmovq %rdx, %xmm1\n\t"
    "vbroadcastss %xmm1, %xmm1\n\t"
    "mulps %xmm1, %xmm0\n\t"

"return: "
    "movq  %rbp, %rsp\n\t"
    "popq %rbp\n\t"
    "ret\n\t"
);

static void aatan2(const float y, const float x, float *__restrict__ result) {

    if (x == 0.f && y == 0.f) {
        *result = NAN;
    }

    if (0.0f == x) {
        *result =  y > 0.0f ? M_PI_2 : -M_PI_2;
    } else {
        if (ffabsf(x) > ffabsf(y)) {
            *result =  x > 0.f ? aatan(y / x) : y >= 0.f ? aatan(y / x) + M_PI : aatan(y / x) - M_PI;
        } else {
            *result =  y > 0.f ? -aatan(x / y) + M_PI_2 : -aatan(x / y) - M_PI_2;
        }
    }
}

static inline void runTest(struct atanArgs *args, float *__restrict__ result) {

    args->fun(args->x, args->y, result);
}

static inline void arctan2Wrapper(const float x, const float y, float *__restrict__ result) {

    *result = atan2(x,y);
}

int main(void) {

    static char *runNames[TIMING_RUNS] = {"aatan2 :: ", "atan2 :: "};
    int i, j;
    struct atanArgs args;
    float results[TIMING_RUNS], delta;

    for (i = j = MAX; i < MAX; j += STEP) {

        // z_norm = (x + iy) / ||(x + iy)|| =  (x + iy) / Sqrt[x^2 + y^2]
        float norm = sqrtf(i * i + j * j);
        args.x /= norm;
        args.y /= norm;

        args.fun = aatan2;
        timeFun((timedFun) runTest, &args, (void *) &(results[0]), 0);

        args.fun = arctan2Wrapper;
        timeFun((timedFun) runTest, &args, (void *) &(results[1]), 1);

        delta = ffabsf(results[1]) - ffabsf(results[0]);
        PRINTF("(%d, %d) -> (%f, %f) :: delta: %f\n", i, j, results[0], results[1], delta);

        if (!(isnan(results[0]) || isnan(results[1]))) {

            assert(ffabsf(delta < MAX_ERROR));
        }

        if (j >= MAX) {
            j = MIN;
            i++;
        }
    }

#ifdef DEBUG
    static union vect v = {-5, -5, 10, 10};
    static union vect u = {5, 5, 10, 10};
    static union vect vbar = {-oneOverRootFive, -oneOverRootFive, twoOverRootFive, twoOverRootFive}; // => Sqrt[25 + 100] = 5*sqrt[5]
    static union vect ubar = {oneOverRootFive, oneOverRootFive, twoOverRootFive, twoOverRootFive};
    static union vect x = {0,0,0,0};
    static union vect y = {-1, -1, 0, 0};
    static union vect w;

    float r = v.arr[0];
    float k = v.arr[3];
    __m128 az;
//    float az = argz(&v.vect);
//    w = (union vect){.vect = az};
//    printf("%f %f %f %f\n", w.arr[0], w.arr[1], w.arr[2],  w.arr[3]);
////    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    r = v.arr[0];
    k = v.arr[3];
    az = argz(&vbar.vect);
//    w = (union vect){{1,2,3,4}};
//    union vect p = (union vect){5,6,7,8};
//    w.vect = _mm_shuffle_ps(w.vect, p.vect, _MM_SHUFFLE(0,1,2,3));
    w = (union vect){.vect = az};
    float f;
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);

    r = u.arr[0];
    k = u.arr[3];
    az = argz(&u.vect);
    w = (union vect){.vect = az};
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);
//    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    r = x.arr[0];
    k = x.arr[3];
    az = argz(&x.vect);
    w = (union vect){.vect = az};
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);
//    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    r = y.arr[0];
    k = y.arr[3];
    az = argz(&y.vect);
    w = (union vect){.vect = az};
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);
//    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    r = vbar.arr[0];
    k = vbar.arr[3];
    az = argz(&vbar.vect);
    w = (union vect){.vect = az};
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);
//    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    r = ubar.arr[0];
    k = ubar.arr[3];
    az = argz(&ubar.vect);
    w = (union vect){.vect = az};
    aatan2(k, r, &f);
    printf("Arg[(%f + %fi)] -> %f %f %f %f :: %f, %f\n",r, k, w.arr[0], w.arr[1], w.arr[2],  w.arr[3], atan2f(k, r), f);
//    printf("Arg[(%f + %fi)] -> %f vs %f\n", r, k, az, atan2f(k, r));

    printf("%X %X %X %X\n", _MM_SHUFFLE(2,1,2,1), _MM_SHUFFLE(1,2,1,2), _MM_SHUFFLE(3,1,2,0),
           _MM_SHUFFLE(3,2,1,0));
#endif
    printTimedRuns(runNames, TIMING_RUNS);
    
    return 0;
}
