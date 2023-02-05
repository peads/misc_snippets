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

float fun(float x) {

    const float half = 0.5f;
    const float threeHavles = 1.5f;

    __asm__ __volatile__(

        "vldr s1, %0\n\t"
        "vldr s2, %1\n\t"
        "vmul.F32 s1, s2, s1\n\t"

        "ldr r1, %0\n\t"
        "asr r1, r1, 1\n\t"
        "ldr r2, =0x5f3759df\n\t"
        "sub r1, r2, r1\n\t"
        "str r1, %0\n\t"

        "vldr s2, %0\n\t"
        "vmul.F32 s3, s2, s2\n\t"   // x^2
        "vmul.F32 s1, s1, s3\n\t"   // x/2*x^2
        "vldr s3, %2\n\t"
        "vsub.F32 s1, s3, s1\n\t"   // 1.5 - x/2*x^2
        "vmul.F32 s1, s1, s2\n\t"   // x*(1.5 - x/2*x^2)
        "vmov r1, s1\n\t"
        "str r1, %0\n\t"

        : "+m" (x) 
        : "m" (half), "m" (threeHavles)
    );
    
    printf("%f, %X\n",x,x);
    return x;
}


int main(void) {
    printf("%X", fun(2.0f));
}
