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

#define MUL_PLUS_J_INT(X, J)    \
    tmp = X[J]; \
    X[J] = -X[J+1]; \
    X[J+1] = tmp

#define MUL_MINUS_J_INT(X, J) \
    tmp = X[J]; \
    X[J] = X[J+1]; \
    X[J+1] = -tmp

extern void swapNegate(short *x, short *y);
__asm__ (
#ifdef __APPLE_CC__
"_swapNegate: "
#else
"swapNegate: "
#endif
    "mov (%rsi), %bx\n\t"
    "mov (%rdi), %cx\n\t"
    "mov %bx, (%rdi)\n\t"
    "neg %cx\n\t"
    "mov %cx, (%rsi)\n\t"
    "ret"
);

union combine {
    short arr[sizeof(__m128i) / sizeof(uint16_t)] __attribute__((aligned(16)));
    __m128i vect;
};

static const union combine X = {.arr = {1,1,1,1,-1,-1,1,-1}};
static const union combine Y = {.arr = {1,1,-1,1,1,1,1,1}};

void fun(short *a, int len) {

    short *firstHalf __attribute__((aligned(16)));
    short *secondHalf __attribute__((aligned(16)));
    short temp[8];

    int i, j, k = 0;
    int max = len >> 2;
    __m128i arr[max >> 1];

    for (i = 0; i < max; i += 2) {
        j = (i << 2);
        firstHalf = a + j;
        secondHalf = a + j + 4;

        arr[k++] = _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i*) firstHalf),
                _mm_loadl_epi64((const __m128i*) secondHalf));
    }

    for (i = 0; i < max/2; ++i) {
        arr[i] = _mm_mullo_epi16(arr[i], X.vect);
        arr[i] = _mm_mullo_epi16(arr[i], Y.vect);
        _mm_storeu_si128((__m128i_u *) temp, arr[i]);
        printf("%d %d %d %d %d %d %d %d\n",
               temp[0],
               temp[1],
               temp[2],
               temp[3],
               temp[4],
               temp[5],
               temp[6],
               temp[7]);
    }
}

int main(void) {

    short tmp, xx, yy;
    short ints[4] = {21, 7, 1, 2};
    short a[32] =
    {1, 2, 3, 4, 5, 6, 7,
     8, 9, 10, 11, 12, 13, 14, 15, 16,
     17, 18 , 19, 20, 21, 22, 23, 24,
     25, 26, 27, 28, 29, 30, 31, 32};

    xx = ints[1];
    yy = -ints[0];
    MUL_MINUS_J_INT(ints, 0);
    assert(ints[0] == xx && ints[1] == yy);

    ints[0] = 21;
    ints[1] = 7;
    swapNegate(&ints[0], &ints[1]);
    assert(ints[0] == xx && ints[1] == yy);

    ints[0] = 21;
    ints[1] = 7;
    xx = -ints[1];
    yy = ints[0];
    MUL_PLUS_J_INT(ints, 0);
    assert(ints[0] == xx && ints[1] == yy);

    ints[0] = 21;
    ints[1] = 7;
    swapNegate(&ints[1], &ints[0]);
    assert(ints[0] == xx && ints[1] == yy);
    fun(a, sizeof(a) / sizeof(*a));
}