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

#define MAX_ERROR 10e-7
#define MIN -20
#define MAX 20
#define STEP 1
//#define DEBUG

/**
 * Takes two packed floats representing the complex numbers
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
 * Takes two packed floats representing the complex numbers
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
    "vrsqrtps %xmm1, %xmm1\n\t"             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

                                            // approximating atan2 with atan(z)
                                            //   = z/(1 + (9/32) z^2) for z = y/x
    "movddup LC0(%rip), %xmm2\n\t"          // 64
    "movddup LC1(%rip), %xmm3\n\t"          // 23

    "vmulps %xmm2, %xmm0, %xmm2\n\t"        // 64*zj
    "vmulps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr
    "movddup LC2(%rip), %xmm0\n\t"          // 41
    "vaddps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr + 41
    "vpermilps $0x1B, %xmm3, %xmm3\n\t"
    "vdivps %xmm3, %xmm2, %xmm0\n\t"        // 64*zj / ||z|| * (23*zr / ||z|| + 41)^-1

    "vextractps $1, %xmm0, %rax\n\t"
    "vmovq %rax, %xmm0 \n\t"
    "ret \n\t"
);

int main(void) {

    __m128 z = {1.f, 2.f, 3.f, 4.f};
    __m128 w = {41.f, 41.f, 41.f, 41.f};
    __m128i wint = _mm_castps_si128(w);

    union vect temp = {.vect = z};
    float zr = temp.arr[0]*temp.arr[2] - temp.arr[1]*temp.arr[3];
    float zj = temp.arr[0]*temp.arr[3] + temp.arr[1]*temp.arr[2];

    printf("(%.01f + %.01fI).(%.01f + %.01fI) = (%.01f + %.01fI), Phase: %f\n",
           temp.arr[0], temp.arr[1], temp.arr[2], temp.arr[3],
           zr, zj, atan2f(zj, zr));

    printf("Phase from argz: %f\n", argz(z));
    printf("Phase from argzB: %f\n", argzB(z));
    printf("%X\n", _MM_SHUFFLE(0,1,2,3));

    return 0;
}
