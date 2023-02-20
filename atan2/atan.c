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
#include <assert.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <math.h>

#define TIMING_RUNS 2
#include "timed_functions.h"

#define MAX_ERROR 1e-2
#define MIN -200
#define MAX 200
#define STEP 1
//#define DEBUG

union vect {
    float arr[4] __attribute__((aligned(16)));
    __m128 vect;
};
typedef void (*atanFun)(const float y, const float x, float *__restrict__ result);
struct atanArgs {
    atanFun fun;
    float x;
    float y;
};

extern float ffabs(float f);
__asm__(
#ifdef __APPLE_CC__
"_ffabs: "
#else
"ffabs: "
#endif
    "movq %xmm0, %rax\n\t"
    "andl $0x7FFFFFFF, %eax\n\t"
    "movq %rax, %xmm0\n\t"
    "ret"
);

extern int isNegZero(float f);
__asm__(
#ifdef __APPLE_CC__
"_isNegZero: "
#else
"isNegZero: "
#endif
    "movq %xmm0, %rax\n\t"
    "andl $0x80000000, %eax\n\t"
    "ret"
);
//extern float argz(__m128 *z0);
//__asm__(
//#ifdef __APPLE_CC__
//"_argz: "
//#else
//"argz: "
//#endif
//    "vmovaps	(%rdi), %xmm0\n\t"
//    "vmulps     (%rsi), %xmm0, %xmm0\n\t"
//    "vmovaps %xmm0,  (%rdi)\n\t"
//    "vpermilps	$177, %xmm0, %xmm0\n\t"
//    "vmovaps %xmm0,  (%rsi)\n\t"
//    "vaddsubps	(%rdi), %xmm0, %xmm1\n\t"
//    "vpermilps	$240, %xmm1, %xmm0\n\t"
//    "vmovaps %xmm0,  (%rdi)\n\t"
//    /* xmm -> x87 "faptain" section */
//    "sub        $16, %rsp\n\t"   // emulate push, s.t. xmm can be accessed by the fpu
//    "vextractps $3, %xmm0,  (%rsp)\n\t"   // extractps seems to work better here,
//    //  "vmovhps %xmm0,  (%rsp)\n\t"   // which corresponds to the latency
//    "flds       (%rsp)\n\t"   // and throughput tables from intel.
//    "sub        $16, %rsp\n\t"   // albeit marginally, that is.
//    "vextractps $0, %xmm0,  (%rsp)\n\t"
//    "flds       (%rsp)\n\t"
//    "fpatan\n\t"
//    "fstps      (%rsp)\n\t"
//    // B I G pop emulation
//    "vmovq    (%rsp), %xmm0\n\t"
//    "add        $32, %rsp\n\t"
//    "ret"
//);

float __attribute__((aligned(16))) aatan(float z __attribute__((aligned(16)))) {

    return (0.97239411 - 0.19194795 * z * z) * z;
}
extern float argz(__m128 *z);
__asm__(
#ifdef __APPLE_CC__
    "_argz: "
    #else
    "argz: "
    #endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"
    "and $-16, %rsp\n\t"

    "vmovaps (%rdi), %xmm0\n\t"
    "vpermilps $0x0, %xmm0, %xmm2\n\t" // {x, x, x, x}
    "vpermilps $0xFF, %xmm0, %xmm3\n\t" // {y, y, y, y}

    "xorq %rdx, %rdx\n\t"
    "movq %xmm2, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz xynz\n\t"          // if x != 0 continue on

    "movq %xmm3, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz xynz\n\t"          // if y also != 0 continue on
    "jmp return\n\t"        // otherwise, bye, felicia

"xynz: "
    "vmulps %xmm0, %xmm0, %xmm0\n\t" // x*x, x*x, y*y, y*y
    "vpermilps $0x1B, %xmm0, %xmm1\n\t"
    "vaddps %xmm1, %xmm0, %xmm0\n\t" // x*x + y*y, ...
    "vsqrtps %xmm0, %xmm0\n\t"      // Sqrt[x*x + y*y], ...
    "vaddsubps %xmm2, %xmm0, %xmm0\n\t" // Sqrt[...] +- x alternately, avoids an extra instruction
    "vmovaps %xmm0, (%rdi)\n\t"         // store in return argument

    "vxorps %xmm1, %xmm1, %xmm1\n\t"
    "vcmpless %xmm1, %xmm2, %xmm1\n\t"
    "movq %xmm1, %rcx\n\t"
    "jrcxz posx\n\t"            // if (x > 0) gote .posx

    "movq %xmm3, %rcx\n\t"
    "cmp %rdx, %rcx\n\t"
    "jnz ynz\n\t"                // if y also != 0 continue on
    "movl $0x40490fdb, %ecx\n\t"
    "vbroadcastss (%rcx), %xmm0\n\t"
    "jmp return\n\t"            // otherwise return Pi

"ynz: "
    "vdivps %xmm0, %xmm3, %xmm0\n\t" // y / (Sqrt[...] + x), ...
    "jmp homestretch\n\t"

"posx: "
    "vdivps %xmm3, %xmm0, %xmm0\n\t" // (Sqrt[...] - x) / y, ...
    "jmp homestretch\n\t"

"homestretch: "
    "vextractps $1, %xmm0, %rdx\n\t" // TODO replace this with function call to a one that takes a vector
    "movq %rdx, %xmm0\n\t"
    "sub $128, %rsp\n\t"
    "mov %rsp, %rdx\n\t"
    "call _atanf\n\t"
    "mov %rdx, %rsp\n\t"
    "add $128, %rsp\n\t"

    "movq $0x40000000, %rdx\n\t" // TODO replace this with vector multiply
    //0x7FFEEA0FF930 all single packed 2s?
    "vmovq %rdx, %xmm1\n\t"
    "mulss %xmm1, %xmm0\n\t"

"return: "
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
        if (ffabs(x) > ffabs(y)) {
            *result =  x > 0.f ? aatan(y / x) : y >= 0.f ? aatan(y / x) + M_PI : aatan(y / x) - M_PI;
        } else {
            *result =  y > 0.f ? -aatan(x / y) + M_PI_2 : -aatan(x / y) - M_PI_2;
        }
    }
}

static inline void runTest(struct atanArgs *args, float *__restrict__ result) {
    args->fun(args->x, args->y, result);
}

void arctan2Wrapper(const float x, const float y, float *__restrict__ result) {
    *result = atan2(x,y);
}

int main(void) {

    static char *runNames[TIMING_RUNS] = {"aatan2 :: ", "atan2 :: "};
    union vect v = {-5, -5, 10, 10};
    union vect u = {5, 5, 10, 10};
    int i, j;
    struct atanArgs args;
    float results[TIMING_RUNS], delta;

//    printf("%X %llX\n", _MM_SHUFFLE(0,1,2,3), u.vect);
    for (i = j = MIN; i < MAX; j += STEP) {
        args.x = i;
        args.y = j;

        args.fun = aatan2;
        timeFun((timedFun) runTest, &args, (void *) &(results[0]), 0);

        args.fun = arctan2Wrapper;
        timeFun((timedFun) runTest, &args, (void *) &(results[1]), 1);

        delta = ffabs(results[1]) - ffabs(results[0]);
        printf("(%d, %d) -> (%f, %f) :: delta: %f\n", i, j, results[0], results[1], delta);

        assert(ffabs(delta < MAX_ERROR));
        if (j >= MAX) {
            j = MIN;
            i++;
        }
    }
    union vect x = {0,0,0,0};
    union vect y = {-1, -1, 0, 0};

    float az = argz(&v.vect);
    printf("%f\n", az);

    az = argz(&u.vect);
    printf("%f\n", az);

    az = argz(&x.vect);
    printf("%f\n", az);

    az = argz(&y.vect);
    printf("%f\n", az);

    printTimedRuns(runNames, TIMING_RUNS);
    return 0;
}