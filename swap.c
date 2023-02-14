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

#include <stdint.h>
#include <stdio.h>

//static void swap1(double *x, double *y) {
//    uint64_t **i = (uint64_t**)&x;
//    uint64_t **j = (uint64_t**)&y;
//
//    **i ^= **j;
//    **j ^= **i;
//    **i ^= **j;
//}

//static void swap(double *x, double *y) {
//
//    __asm__ (
//        "xorq   %0, %1\n\t"
//        "xorq   %1, %0\n\t"
//        "xorq   %0, %1\n\t"
//        "movq   (%0), %0\n\t"
//        "movq   (%1), %1\n\t"
//        : "=r" (*x), "=r" (*y) : "0" (x), "1" (y)
//    );
//}
static void swap1(void *x, void *y) {
    uintptr_t **i = (uintptr_t**)&x;
    uintptr_t **j = (uintptr_t**)&y;

    **i ^= **j;
    **j ^= **i;
    **i ^= **j;
}

extern void swap2(void *x, void *y);
__asm__(
#ifdef __APPLE_CC__
"_swap2: "
#else
"swap2: "
#endif
    "movq (%rsi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "xorq (%rdi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "movq %rax, (%rsi)\n\t"
    "ret"
);


static void swap(void *x, void *y) {
    __asm__ (
        "movq (%%rsi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "xorq (%%rdi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "movq %%rax, (%%rsi)\n\t"
        ://:::"rax"
    );
}

int main(void) {
    double x = 1.0;
    double y = 2.0;

    printf("%f %f\n", x, y);

    swap1(&x, &y);

    printf("%f %f\n", x, y);

    swap(&x, &y);

    printf("%f %f\n", x, y);
    swap2(&x, &y);

    printf("%f %f\n", x, y);
    return 0;
}

