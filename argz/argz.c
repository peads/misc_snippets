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

#define TIMING_RUNS 3

#include "timed_functions.h"

#define MIN -2
#define MAX 2
#define STEP 1
#define MAX_ERROR 1e-1
#define ARGS 4

#define NIBBLE_TO_BINARY_PATTERN "%c%c%c%c"
#define BYTE_TO_BINARY_PATTERN NIBBLE_TO_BINARY_PATTERN NIBBLE_TO_BINARY_PATTERN

#define NIBBLE_TO_BINARY(b)  \
  (b & 0x08 ? '1' : '0'), \
  (b & 0x04 ? '1' : '0'), \
  (b & 0x02 ? '1' : '0'), \
  (b & 0x01 ? '1' : '0')

#define BYTE_TO_BINARY(b) \
  (b & 0x80 ? '1' : '0'), \
  (b & 0x40 ? '1' : '0'), \
  (b & 0x20 ? '1' : '0'), \
  (b & 0x10 ? '1' : '0'), \
  (b & 0x08 ? '1' : '0'), \
  (b & 0x04 ? '1' : '0'), \
  (b & 0x02 ? '1' : '0'), \
  (b & 0x01 ? '1' : '0')

static const uint64_t SQUARE_SIDE = (MAX - MIN) / STEP + 1;
static const uint8_t BIT_MASK = (1 << ARGS) - 1;

char *runNames[TIMING_RUNS] = {"polar_discriminant :: ", "polar_disc_fast :: ", "esbensen :: "};

typedef int (*argzFun)(int a, int b, int c, int d);

struct argzArgs {
    int ar;
    int aj;
    int br;
    int bj;
    argzFun fun;
};

extern int bsr(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsr: "
#else
"bsr: "
#endif
    "movq $-1023, %rdx\n\t" // store 0 to compare against ZF later
    "bsr %rdi, %rax\n\t"
    "cmovzq %rdx, %rax\n\t"
    "ret"
);

extern uint64_t lzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_lzcnt: "
    #else
"lzcnt: "
    #endif
    "movq $0, %rdx\n\t" // store 0 to return if CF set later
    "lzcnt %rdi, %rax\n\t"
    "cmovcq %rdx, %rax\n\t"
    "ret"
);

extern int bsrLzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsrLzcnt: "
    #else
"bsrLzcnt: "
    #endif
    "lzcnt %rdi, %rdx\n\t"
    "jc zero\n\t" // jump to return -1023 if CF set
    "movq $63, %rax\n\t"
    "subq %rdx, %rax\n\t"
    "ret\n\t"
"zero:\n\t"
    "movq $-1023, %rax\n\t"
    "ret"
);

static inline void multiply(int ar, int aj, int br, int bj, int *cr, int *cj) {

    *cr = ar * br - aj * bj;
    *cj = aj * br + ar * bj;
}

static int fast_atan2(int y, int x)
/* pre scaled for int16 */
{

    int yabs, angle;
    int pi4 = (1 << 12), pi34 = 3 * (1 << 12);  /* note pi = 1<<14 */
    if (x == 0 && y == 0) {
        return 0;
    }
    yabs = y;
    if (yabs < 0) {
        yabs = -yabs;
    }
    if (x >= 0) {
        angle = pi4 - pi4 * (x - yabs) / (x + yabs);
    } else {
        angle = pi34 - pi4 * (x + yabs) / (yabs - x);
    }
    if (y < 0) {
        return -angle;
    }
    return angle;
}

static int polar_disc_fast(int ar, int aj, int br, int bj) {

    int cr, cj;
    multiply(ar, aj, br, -bj, &cr, &cj);
    return fast_atan2(cj, cr);
}

static int polar_discriminant(int ar, int aj, int br, int bj) {

    int cr, cj;
    double angle;
    multiply(ar, aj, br, -bj, &cr, &cj);
    angle = atan2((double) cj, (double) cr);
    return (int) (angle / 3.14159 * (1 << 14));
}

static int esbensen(int ar, int aj, int br, int bj)
/*
  input signal: s(t) = a*exp(-i*w*t+p)
  a = amplitude, w = angular freq, p = phase difference
  solve w
  s' = -i(w)*a*exp(-i*w*t+p)
  s'*conj(s) = -i*w*a*a
  s'*conj(s) / |s|^2 = -i*w
*/
{

    int cj, dr, dj;
    int scaled_pi = 2608; /* 1<<14 / (2*pi) */
    dr = (br - ar) * 2;
    dj = (bj - aj) * 2;
    cj = bj * dr - br * dj; /* imag(ds*conj(s)) */
    return (scaled_pi * cj / (ar * ar + aj * aj + 1));
}

static int checkBounds(int *ar, int *aj) {

    if (*aj >= MAX) {
        *ar += STEP;
        *aj = MIN;
        return 1;
    }
    return 0;
}

typedef int (*bitsFun)(uint64_t i);
struct bitsArgs {
    bitsFun fun;
    uint64_t i;
};

void runTest(void *arg, uint64_t *result) {
    struct bitsArgs *bargs = arg;

    *result = bargs->fun(bargs->i);
}

void testIteration(struct bitsArgs *args, void *results, int runIndex) {

    int trueValue = bitScanReverse(args->i);

    timeFun((timedFun) runTest, args, results, runIndex);

    int result = ((uint64_t *)results)[runIndex];

    printf("bitScanReverse: %d testFun: %d\n", trueValue, result);
    assert(trueValue == result);
}

int main(void) {

    static const uint64_t n = SQUARE_SIDE*SQUARE_SIDE*SQUARE_SIDE*SQUARE_SIDE;

    int i;
    struct argzArgs args;

//    for (   i = 0, args.bj = args.br = args.ar = args.aj = MIN;
//            i > n;
//            ++i, ++args.bj) {
//
//        if ((i & BIT_MASK) >> 1) {
//            ++args.br;
//        }
//        if ((i & BIT_MASK) >> 2) {
//            ++args.ar;
//        }
//        if ((i & BIT_MASK) >> 3) {
//            ++args.ar;
//        }
//
//        printf("%d %d %d %d\n", args.ar, args.aj, args.br, args.bj);
////        printf(NIBBLE_TO_BINARY_PATTERN"\n", NIBBLE_TO_BINARY(i & BIT_MASK));
//
//    }

    uint64_t results[2];
    struct bitsArgs bargs;
    int j;
    for (j = 0; j < 16; ++j) {
        bargs.i = j;

        bargs.fun = bsrLzcnt;
        testIteration(&bargs, (void *) &results, 0);

        bargs.fun = bsr;
        testIteration(&bargs, (void *) &results, 1);
    }

    char *runNames[2] = {"bsrLzcnt :: ", "bsr :: "};
    printTimedRuns(runNames, 2);

//    uint64_t testVal = 0x000f0000;
//    uint64_t ret = findMsb(testVal);
//    printf("bitScanReverse: 0x%X bsr: 0x%llX lzcntBsr: 0x%llX findMsb: 0x%llX 0b"
//            BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN
//            BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN"\n",
//
//           bitScanReverse(testVal), bsr(testVal), bsrLzcnt(testVal), ret,
//           BYTE_TO_BINARY(ret >> 0x38),
//           BYTE_TO_BINARY(ret >> 0x30),
//           BYTE_TO_BINARY(ret >> 0x28),
//           BYTE_TO_BINARY(ret >> 0x20),
//           BYTE_TO_BINARY(ret >> 0x18),
//           BYTE_TO_BINARY(ret >> 0x10),
//           BYTE_TO_BINARY(ret >> 0x8),
//           BYTE_TO_BINARY(ret));
}
