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
#include <time.h>
#include <stdlib.h>
#include <math.h>
#define TIMING_RUNS 3
#include "timed_functions.h"

#define MAX_ERROR 10e-5
#define MIN 0
#define MAX 20
#define STEP 0.001

#if defined(__x86_64__) || defined(_M_X64)
///return "x86_64";
#define X86
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
//return "x86_32";
#define X86
#endif
extern float ffabsf(float x);

float invSqrt(float x) {

    if (x <= 0.f){
        return 0.f;
    }

    uint32_t i = *(uint32_t *) &x;            // store floating-point bits in integer
    float xhalf = 0.5f * x;
    i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
    x = *(float *) &i;              // convert new bits into float

    return x*(1.5f - xhalf * x * x);     // One round of Newton's method
}


extern float asmInvSqrt(float x);
__asm__(
#ifdef __clang__
"_asmInvSqrt: "
#else
"asmInvSqrB: "
#endif
#ifndef X86
    "ldr r1, %0\n\t"

        "vmov s2, r1\n\t"
        "vldr s1, %1\n\t"
        "vmul.F32 s1, s2, s1\n\t"

        "asr r1, r1, 1\n\t"
        "ldr r2, =0x5f375a85\n\t"
        "sub r1, r2, r1\n\t"
        //"str r1, %0\n\t"

        //"vldr s2, %0\n\t"
        "vmov s2, r1\n\t"
        "vmul.F32 s3, s2, s2\n\t"   // x^2
        "vmul.F32 s1, s1, s3\n\t"   // x/2*x^2
        "vldr s3, %2\n\t"
        "vsub.F32 s1, s3, s1\n\t"   // 1.5 - x/2*x^2
        "vmul.F32 s1, s1, s2\n\t"   // x*(1.5 - x/2*x^2)
        "vmov r1, s1\n\t"
        "str r1, %0\n\t"
    //return sqrtf(x);
#else
    "xorps %xmm1, %xmm1\n\t"
    "vucomiss %xmm0, %xmm1\n\t"
    "jae return\n\t"

    "movq %xmm0, %rax\n\t"
    "movl $0xBf000000, %edx\n\t"
    "pushq %rax\n\t"
    "pushq %rdx\n\t"
    "flds (%rsp)\n\t"
    "pop %rdx\n\t"
    "flds (%rsp)\n\t"
    "fmulp\n\t"
    "pop %rdx\n\t"

    "sarl $1, %eax\n\t"
    "negl %eax\n\t"
    "movl $0x5f375a85, %edx\n\t"
    "addl %edx, %eax\n\t"           // x = magic_number - *(uint32*)&x

    "movl $0x3fc00000, %edx\n\t"
    "pushq %rax\n\t"
    "flds (%rsp)\n\t"
    "flds (%rsp)\n\t"
    "fmulp\n\t"         // x^2
    "fmulp\n\t"         // x^2 * x0/2
    "pushq %rdx\n\t"
    "flds (%rsp)\n\t"
    "faddp\n\t"         // (1.5 -(x^2 * x0/2))
    "popq %rdx\n\t"
    "flds (%rsp)\n\t"
    "fmulp\n\t"         // x*(1.5 -(x^2 * x0/2))
    "fstps (%rsp)\n\t"
    "pop %rax\n\t"
    "movq %rax, %xmm0\n\t"

    "ret"
#endif
);

extern float asmInvSqrtB(float x);
__asm__(
#ifdef __clang__
"_asmInvSqrtB: "
#else
"asmInvSqrtB: "
#endif
    "xorps %xmm1, %xmm1\n\t"
    "vucomiss %xmm0, %xmm1\n\t"
    "jae return\n\t"

    "movq %xmm0, %rax\n\t"// load x0
    "movl $0xBf000000, %edx\n\t"
    "movq %rdx, %xmm1\n\t"           // -1/2 -> xmm1
    "mulss %xmm0, %xmm1\n\t" // -x0/2 -> xmm1

    "shrl $1, %eax\n\t"
    "negl %eax\n\t"
    "addl $0x5f3759df, %eax\n\t"
    "movq %rax, %xmm0\n\t"           // x = 0x5f3759df - (x0 >> 1) -> xmm1

    "vmulps %xmm0, %xmm0, %xmm2\n\t" // x*x -> xmm2
    "mulss %xmm2, %xmm1\n\t" //-x0/2x * x*x -> xmm1
    "movl $0x3fc00000, %eax\n\t"
    "movq %rax, %xmm2\n\t"           // 3/2 -> xmm2
    "addss %xmm2, %xmm1\n\t" // 3/2 - x0/2 * x*x
    "mulss %xmm1, %xmm0\n\t" // x*(3/2 - x/2 * x*x)

"return: "
    "ret"
);

extern float fsqrtf(float f);

int main(void) {

    float a, b, c, f;
    uint64_t n;
    long double deltas[2], d = 0.l;
    struct timespec tstart, tend;


    for (f = MIN; f < MAX; f+=STEP, ++n) {
        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        a = invSqrt(f);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(0, &tstart, &tend);
        // END TIMED

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        b = asmInvSqrt(f);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(1, &tstart, &tend);
        // END TIMED

        // START_TIMED
        clock_gettime(CLOCK_MONOTONIC, &tstart);

        c = asmInvSqrtB(f);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(2, &tstart, &tend);
        // END TIMED

        printf("Sqrt[%f] = %f, %f, %f, %f\n", f, 1/fsqrtf(f), a, b, c);

        deltas[0] = (long double) ffabsf(ffabsf(a) - ffabsf(b));
        assert(deltas[0] < MAX_ERROR);

        deltas[1] = (long double) ffabsf(ffabsf(a) - ffabsf(c));
        assert(deltas[1] < MAX_ERROR);

        d += deltas[0] + deltas[1];
    }
    char *runNames[3]
            = {"carmack :: ", "asmInvSqrt :: ", "asmInvSqrtB :: "};

    printTimedRuns(runNames, TIMING_RUNS);
    d /= n;
    printf("\navg delta: %Le\n", d);
    return 0;
}

