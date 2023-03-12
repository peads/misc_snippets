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

#define TIMING_RUNS 2
#define DEBUG
#include "timed_functions.h"

#ifdef DEBUG
    #define PRINTF printf
#else
    #define PRINTF //
#endif

#define MAX_ERROR 0.1
#define MIN -500
#define MAX 500
#define STEP 1
//#define DEBUG

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
    "vextractps $1, %xmm0, -8(%rsp) \n\t"
    "flds -8(%rsp) \n\t"
    // push
    "vextractps $2, %xmm0, -8(%rsp) \n\t"
    "flds -8(%rsp) \n\t"
    "fpatan \n\t"
    "fstps -8(%rsp) \n\t"

    // pop and return
    "vmovq -8(%rsp), %xmm0 \n\t"
    "jmp bye\n\t"

"zero: "
    "vxorps %xmm0, %xmm0, %xmm0\n\t"
"bye: "
    "ret"
);
/**
 * Takes packed float representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns their argument as a float
 **/
extern float argzB(__m128 a);
__asm__(

".section: .rodata:\n\t"
".p2align 4\n\t"
"LC0: "
    ".quad 4791830004637892608\n\t"
"LC1: "
    ".quad 4735535009282654208\n\t"
"LC2: "
    ".quad 4765934306774482944\n\t"
".text\n\t"

#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif

    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm3\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm3, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    "vxorps %xmm3, %xmm3, %xmm3\n\t"
    "vpermilps $0x01, %xmm0, %xmm2\n\t"
    "vcomiss %xmm2, %xmm3\n\t"
    "jne showtime\n\t"
    "vpermilps $0x02, %xmm0, %xmm2\n\t"
    "vcomiss %xmm2, %xmm3\n\t"
    "jg showtime\n\t"
    "jl pi\n\t"
    "jmp zero\n\t"

"showtime: "                                // approximating atan2 with atan(z)
                                            //   = z/(1 + (9/32) z^2) for z = y/x
    "vrsqrtps %xmm1, %xmm1\n\t"             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "movddup LC0(%rip), %xmm2\n\t"          // 64
    "movddup LC1(%rip), %xmm3\n\t"          // 23

    "vmulps %xmm2, %xmm0, %xmm2\n\t"        // 64*zj
    "vmulps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr
    "movddup LC2(%rip), %xmm0\n\t"          // 41
    "vaddps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr + 41
    "vpermilps $0x1B, %xmm3, %xmm3\n\t"
//    "vdivps %xmm3, %xmm2, %xmm0\n\t"        // 64*zj / ||z|| * (23*zr / ||z|| + 41)^-1
    "vrcpps %xmm3, %xmm3\n\t"
    "vmulps %xmm3, %xmm2, %xmm0\n\t"

    "vpermilps $0x01, %xmm0, %xmm0\n\t"
    "jmp done\n\t"

"pi: "
    "movl $0x40490fdb, %eax\n\t"
    "vmovq %rax, %xmm0\n\t"
"done: "
    "ret \n\t"
);

int main(void) {
    float i, j, k, m, zr, zj, theta, phi, phiB, avg, stdDev;
    float sum = 0.f;
    float N = powf((MAX - MIN) / STEP, 4.f);
    uint32_t errCount = 0;
    float errs[((uint32_t) N) << 1];
    long double deltas[2], d = 0.l;
    struct timespec tstart, tend;

    for (i = MIN; i < MAX; i += STEP) {
        for (j = MIN; j < MAX; j += STEP) {
            for (k = MIN; k < MAX; k += STEP) {
                for (m = MIN; m < MAX; m += STEP) {
                    __m128 z = {i, j, k, m};
                    union vect temp = {.vect = z};

                    zr = temp.arr[0] * temp.arr[2] - temp.arr[1] * temp.arr[3];
                    zj = temp.arr[0] * temp.arr[3] + temp.arr[1] * temp.arr[2];
                    theta = atan2f(zj, zr);

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    phi = argz(z);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(0, &tstart, &tend);
                    // END TIMED

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    phiB = argzB(z);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(1, &tstart, &tend);
                    // END TIMED


                    avg = (theta + phiB) / 2.f;//(theta + phi + phiB) / 3.f;
                    stdDev = sqrtf((powf((theta - avg), 2.f) /*+ powf((phi - avg), 2.f)*/
                                          + powf((phiB - avg), 2.f)) / 2.f);

                    if (stdDev >= MAX_ERROR) {
//                        printf("(%.01f + %.01fI).(%.01f + %.01fI) = (%.01f + %.01fI), Phase: %f\n",
//                               temp.arr[0], temp.arr[1], temp.arr[2], temp.arr[3],
//                               zr, zj, theta);
//                        printf("Phase from argz: %f\n", phi);
//                        printf("Phase from argzB: %f\n", phiB);
//                        printf("std. dev.: %f\n", stdDev);

                        sum += fabsf(theta) + fabsf(phiB);
                        errs[errCount++] = theta;
                        errs[errCount++] = phiB;
                    }
                }
            }
        }
    }

    int n = 0;
    float sd = 0.f;
    float mu = sum/errCount;
    for (; n < (errCount << 1); n+=2) {
        sd += powf((theta - mu), 2.f) + powf((phiB - mu), 2.f);
    }
    sd = sqrtf(sd / errCount);

    printf("%f %u %f %f\n", N, errCount, errCount / N, sd);

    char *runNames[2]
            = {"argz :: ", "argzB :: "};

    printTimedRuns(runNames, TIMING_RUNS);
    return 0;
}
