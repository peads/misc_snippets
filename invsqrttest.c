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

#if defined(__x86_64__) || defined(_M_X64)
///return "x86_64";
#define X86
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
//return "x86_32";
#define X86
#endif

static time_t rollingTimeSums[5];

void findDeltaTime(int idx, struct timespec tstart, struct timespec tend, char *timediff) {
    time_t deltaTsec = tend.tv_sec - tstart.tv_sec;
    time_t deltaTNanos = tend.tv_nsec - tstart.tv_nsec;

    if (deltaTsec > 0) {
        sprintf(timediff, "%lu.%lu s",deltaTsec, deltaTNanos);
    } else {
        sprintf(timediff, "%lu ns", deltaTNanos);
        rollingTimeSums[idx] += deltaTNanos;
        //printf("%s\n", timediff);
    }
}

float invSqrt(float x) {

    float xhalf = 0.5f * x;
    int i = *(int*)&x;            // store floating-point bits in integer
    i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
    x = *(float*)&i;              // convert new bits into float
    x = x*(1.5f - xhalf*x*x);     // One round of Newton's method

    return x;
}

float asmInvSqrt(float x) {

    const float half = 0.5f;
    const float threeHavles = 1.5f;

    __asm__ __volatile__(
#ifndef X86
        "ldr r1, %0\n\t"

        "vmov s2, r1\n\t"
        "vldr s1, %1\n\t"
        "vmul.F32 s1, s2, s1\n\t"

        "asr r1, r1, 1\n\t"
        "ldr r2, =0x5f3759df\n\t"
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
        "movl $0x5f3759df, %%eax\n\t"
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

    return x;
}

float timeAsmInvSqrt(float s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = asmInvSqrt(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(0, tstart, tend, timediff);

    free(timediff);

    return s;
}

float timeInvSqrt(float s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = invSqrt(s);;

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(1, tstart, tend, timediff);

    free(timediff);

    return s;
}

int main(void) {
    float f = 1.0f;
    long cnt;
    double d = 0.;

    for (cnt = 0; cnt < 10e9; cnt++, f+=0.01f){
        float a = timeInvSqrt(f);
        float b = timeAsmInvSqrt(f);
        //printf("sqrt[%f] = %f %f\n", f, a, b);
        d += fabs(a - b);

    }
    
    printf("Carmack avg ns: %f,  asm avg ns: %f\navg delta: %f\n", 
        (float) rollingTimeSums[0] / cnt, 
        (float) rollingTimeSums[1] / cnt,
        d / cnt);
    return 0;
}

