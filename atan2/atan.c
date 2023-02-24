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

#include <string.h>
#include <immintrin.h>
#define TIMING_RUNS 2
#define DEBUG
#include "timed_functions.h"

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
    uint64_t i64[2] __attribute__((aligned(16)));
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

extern float __attribute__((aligned(16))) ffabsf(float f __attribute__((aligned(16))));
extern int __attribute__((aligned(16))) isNegZero(float f __attribute__((aligned(16))));
extern float __attribute__((aligned(16))) fsqrtf(float x __attribute__((aligned(16))));
extern void absMaxMin(float *x __attribute__((aligned(16))), float * y __attribute__((aligned(16))));

/*
 * k[ar_, aj_, br_, bj_] :=
  Piecewise[{
    {ArcCos[(ar*br - aj*bj)/(Sqrt[ar^2 + aj^2]*Sqrt[br^2 + bj^2])],
     		ar*br - aj*bj > 0 && ar*bj + br*aj > 0},

    {-ArcCos[(ar*br - aj*bj)/(Sqrt[ar^2 + aj^2]*Sqrt[br^2 + bj^2])],
     		ar*br - aj*bj > 0 && ar*bj + br*aj < 0},

    {-ArcCos[(ar*br - aj*bj)/(Sqrt[ar^2 + aj^2]*Sqrt[br^2 + bj^2])] ,
     		ar*br - aj*bj < 0 && ar*bj + br*aj < 0},

    {ArcCos[(ar*br - aj*bj)/(Sqrt[ar^2 + aj^2]*Sqrt[br^2 + bj^2])],
     		ar*br - aj*bj < 0 && ar*bj + br*aj > 0},

    (*{0,            ar*bj == 0 &&  br*aj ==0},*)
    {Pi/2,    ar*br - aj*bj == 0 && ar*bj + br*aj != 0}
    }];
 */

/**
 * takes four float representing the complex numbers (ar + iaj) * (br + ibj),
 * s.t. z = {ar, aj, br, bj}
 **/
extern __m128 argz2(float ar __attribute__((aligned(16))),
                    float aj __attribute__((aligned(16))),
                    float br __attribute__((aligned(16))),
                    float bj __attribute__((aligned(16))));
__asm__(
#ifdef __clang__
"_argz2: "
#else
"argz2: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"
    "and $-16, %rsp\n\t"

//    "vshufps $0x0, %xmm1, %xmm0, %xmm0\n\t" // ar, ar, aj, aj
//    "vshufps $0x0, %xmm3, %xmm2, %xmm1\n\t" // br, br, bj, bj
//    "vpermilps $0x27, %xmm0, %xmm0\n\t"     // aj, ar, aj, ar -> xmm0
//    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1

//    "vshufps $17, %xmm2, %xmm3, %xmm1\n\t"
//    "vpermilps $221, %xmm1, %xmm0\n\t"

    "vshufps $17, %xmm0, %xmm1, %xmm0\n\t"
    "vpermilps $221, %xmm0, %xmm0\n\t"      // aj, aj, ar, ar -> xmm0

    "vshufps $0x0, %xmm3, %xmm2, %xmm1\n\t"
    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1

    "vmulps %xmm0, %xmm1, %xmm0\n\t"        // aj*bj, ar*br, aj*br, ar*bj
    "vpermilps $0xB1, %xmm0, %xmm3\n\t"
    "vaddsubps %xmm0, %xmm3, %xmm0\n\t"     // ar*br - aj*bj, ... , ar*bj + aj*br
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // (ar*br - aj*bj)^2, ... , (ar*bj + aj*br)^2
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"     // 0123 = 00011011 = 1B
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vsqrtps %xmm1, %xmm1\n\t"              // Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vdivps %xmm1, %xmm0, %xmm1\n\t"        // (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vxorps %xmm2, %xmm2, %xmm2\n\t"
//    "vcmpps $2, %xmm2, %xmm0, %xmm2\n\t"    // elements of xmm0 != 0
    "movq $0x3F800000, %rdx\n\t"
    "movq $0xBF800000, %rbx\n\t"
    "vextractps $3, %xmm0, %rcx\n\t"
    "shrq $31, %rcx\n\t"
    "cmp $1, %rcx\n\t"
    "cmove %rbx, %rdx\n\t"

    "vmovq %rdx, %xmm2\n\t"
    "vbroadcastss %xmm2, %xmm2\n\t"
    "movq $0x3fc90fdb, %rcx\n\t"
    "vmovq %rcx, %xmm3\n\t"
    "vbroadcastss %xmm5, %xmm5\n\t"
    "vmulps %xmm1, %xmm1, %xmm0\n\t"
    "vmulps %xmm0, %xmm0, %xmm0\n\t"
    "vmulps %xmm0, %xmm0, %xmm0\n\t"
    "vmulps %xmm5, %xmm0, %xmm0\n\t"
    "vaddps %xmm1, %xmm0, %xmm0\n\t"
    "vsubps %xmm0, %xmm3, %xmm0\n\t"
    "vmulps %xmm2, %xmm0, %xmm0\n\t"
    "jmp end\n\t"

"end: "
    "movq  %rbp, %rsp\n\t"
    "popq %rbp\n\t"
    "ret\n\t"
);

extern __m128 argz(__m128 *__restrict__ z);
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
    "jrcxz posx\n\t"                    // if (zr > 0) goto .posx

    "movq %xmm3, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz ynz\n\t"                       // if zj also != 0 continue on
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
//    "movq $0x42000000, %rdx\n\t"
//    "vmovq %rdx, %xmm2\n\t"
//    "vbroadcastss %xmm2, %xmm2\n\t"         // 32, 32, 32, 32 -> xmm2
//    "movq $0x41100000, %rdx\n\t"
//    "vmovq %rdx, %xmm3\n\t"
//    "vbroadcastss %xmm3, %xmm3\n\t"         // 9, 9, 9, 9 -> xmm3
//    "vmulps %xmm0, %xmm0, %xmm1\n\t "       // x*x, x*x, x*x, x*x -> xmm1
//    "vmulps %xmm3, %xmm1, %xmm1\n\t"        // 9xx ,.. -> xmm1
//    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // 9xx + 32, ... -> xmm1
//    "vmulps %xmm2, %xmm0, %xmm0\n\t"        // 32x, ... -> xmm0
//    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // 32x/(9xx + 32), ... -> xmm0

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

static inline void runTest(struct atanArgs *args, float *__restrict__ result) {

    args->fun(args->x, args->y, result);
}

static inline void arctan2Wrapper(const float x, const float y, float *__restrict__ result) {

    *result = atan2(x,y);
}

float atanTwo1(float x, float y) {
    float *result;
    aatan2(x, y, result);
    return *result;
}

static void runDebugTests() {
    static char *runNames[TIMING_RUNS] = {"aatan2 :: ", "atan2 :: "};
    static union vect v = {-5, -5, 10, 10};
    static union vect u = {5, 5, 10, 10};
    static union vect vbar = {-oneOverRootFive, -oneOverRootFive, twoOverRootFive, twoOverRootFive}; // => Sqrt[25 + 100] = 5*sqrt[5]
    static union vect ubar = {oneOverRootFive, oneOverRootFive, twoOverRootFive, twoOverRootFive};
    static union vect x = {0,0,0,0};
    static union vect y = {-1, -1, 0, 0};
    static union vect w;
    float results[TIMING_RUNS], delta;
    struct atanArgs args;

    int i,j;
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

//    printTimedRuns(runNames, TIMING_RUNS);
}

void foo() {
    int i, j, n = 0;
    struct timespec tstart, tend;
    float f = -1000000.f;
    float g;

    float x = g;
    float y = f;

    float a = g;
    float b = f;
    for (i = 0; i < 2000L; ++i, f += 0.1f) {
        for (g = -1000000.f, j = 0; j < 2000L; ++j, g += 0.1f, n++) {
            x = g;
            y = f;
            a = g;
            b = f;

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            absMaxMin(&x, &y);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(0, &tstart, &tend);

            clock_gettime(CLOCK_MONOTONIC, &tstart);

            flipAbsMaxMin(&a, &b);

            clock_gettime(CLOCK_MONOTONIC, &tend);
            findDeltaTime(1, &tstart, &tend);

            assert(a == x && b == y);
        }
    }

    static char *runNames[TIMING_RUNS] = {"absMaxMin :: ", "flipAbsMaxMin :: "};
    printTimedRuns(runNames, 2);
    printf("Iterations: %d\n" , n);
}

float bar(float a, float b, float c, float d) {
    float x = a*c - b*d;
    float y = a*d + b*c;
    float result = x*x + y*y;

    return fsqrtf(result);
}

int main(void) {

    int i, j;
    union vect z, w;

    for (i = -2; i < -1; ++i) {
//        for (j = -2; j < -1; ++j) {
//            for (k = -2; k < -1; ++k) {
//                for (m = -2; m < -1; ++m) {
//
        z.vect = argz2(1,2,3,4);
        printf("Arg[(%f + %fi)]\n", -5.0f, 10.f);
        printf("%f %f %f %f\n\n", z.arr[0], z.arr[1], z.arr[2], z.arr[3]);

        z.vect = argz2(4,3,2,1);
        printf("Arg[(%f + %fi)]\n", 5.0f, 10.f);
        printf("%f %f %f %f\n\n", z.arr[0], z.arr[1], z.arr[2], z.arr[3]);

        z.vect = argz2(-5,-10,1,0);
        printf("Arg[(%f + %fi)]\n", -5.0f, -10.f);
        printf("%f %f %f %f\n\n", z.arr[0], z.arr[1], z.arr[2], z.arr[3]);

        z.vect = argz2(5,-10,1,0);
        printf("Arg[(%f + %fi)]\n",5.f, -10.f);
        printf("%f %f %f %f\n\n", z.arr[0], z.arr[1], z.arr[2], z.arr[3]);

//            z = (union vect){.vect = argz2(i,j,j,i)};
//            printf("%f %f %f %f\n", z.arr[0], z.arr[1], z.arr[2], z.arr[3]);
    }

    return 0;
}
