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
#define TIMING_RUNS 2
#include "timed_functions.h"

#if defined(__x86_64__) || defined(_M_X64)
///return "x86_64";
#define X86
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
//return "x86_32";
#define X86
#endif

void invSqrt(float *x) {

    float xhalf = 0.5f * *x;
    int i = *(int*)x;            // store floating-point bits in integer
    i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
    *x = *(float*)&i;              // convert new bits into float
    *x = *x*(1.5f - xhalf**x**x);     // One round of Newton's method
}

void asmInvSqrt(float *result) {

    const float half = 0.5f;
    const float threeHavles = 1.5f;

    float x = *result;

    __asm__ __volatile__(
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
        "flds %0\n\t"
        "flds %1\n\t"
        "fmulp\n\t"

        "movl %0, %%edx\n\t" 
        "movl %%edx, %3\n\t"
        "sarl $1, %%edx\n\t"        
        "movl $0x5f375a85, %%eax\n\t"
        "subl %%edx, %%eax\n\t"
        "movl %%eax, %0\n\t"        //0x5f3759df - x

        "flds %0\n\t"
        "flds %0\n\t"
        "fmulp\n\t"
        "fmulp\n\t"         // x^2
        "fchs\n\t"
        "flds %2\n\t"
        "faddp\n\t"
        "flds %0\n\t"
        "fmulp\n\t"
        "fstps %0\n\t"
#endif
        : "+m" (x)
        : "m" (half), "m" (threeHavles)
        : "memory"
    );

    *result = x;
}

int main(void) {

    float f = 1.0f;
    uint64_t cnt;
    long double d = 0.;

    for (cnt = 0; cnt < (1<<25); cnt++, f+=0.0001f){

        float a, b;
        a = b = f;
        timeFun((timedFun) invSqrt, &a, NULL, 0);
        timeFun((timedFun) asmInvSqrt, &b, NULL, 0);
        //printf("sqrt[%f] = %f %f\n", f, a, b);
        d += fabsf(fabsf(a) - fabsf(b));
    }
    char *runNames[2]
            = {"carmack :: ", "asm :: "};

    printTimedRuns(runNames, TIMING_RUNS);
    d /= (long double) cnt;
    printf("\navg delta: %Le\n", d);
    return 0;
}

