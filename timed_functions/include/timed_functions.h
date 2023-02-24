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

#ifndef TIMED_FUNCTIONS_H
#define TIMED_FUNCTIONS_H
#ifndef TIMING_RUNS
    #define TIMING_RUNS 8
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#define NINE_THIRTY_SECONDS 0.28125F
#define NINE_THIRTY_SECONDS_F 0x3e900000
#define FLOAT_S_FLIP_MASK  0x7FFFFFFFU
#define FLOAT_E_MASK       0x7F8FFFFFU
#define FLOAT_S_MASK       0x80000000U
#define FLOAT_M_MASK       0x007FFFFFU
#define FLOAT_SHIFT        23U
#define DOUBLE_S_FLIP_MASK  0x7FFFFFFFFFFFFFFFLU
#define DOUBLE_E_MASK       0x7F8FFFFFFFFFFFFFLU
#define DOUBLE_S_MASK       0x8000000000000000LU
#define DOUBLE_M_MASK       0x007FFFFFFFFFFFFFLU
#define DOUBLE_SHIFT        52LU
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
      (byte & 0x80 ? '1' : '0'), \
      (byte & 0x40 ? '1' : '0'), \
      (byte & 0x20 ? '1' : '0'), \
      (byte & 0x10 ? '1' : '0'), \
      (byte & 0x08 ? '1' : '0'), \
      (byte & 0x04 ? '1' : '0'), \
      (byte & 0x02 ? '1' : '0'), \
      (byte & 0x01 ? '1' : '0')
#if defined(__x86_64__) || defined(_M_X64)
    #define x86_64
    #define X86
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)

#define x86_32
    #define X86
#elif defined(__ARM_ARCH_2__)
    #define ARM
    #define ARM2
#elif defined(__ARM_ARCH_3__) || defined(__ARM_ARCH_3M__)
    #define ARM
    #define ARM3
#elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARM_4T)
    #define ARM
    #define ARM4T
#elif defined(__ARM_ARCH_5_) || defined(__ARM_ARCH_5E_)
    #define ARM
    #define ARM5
#elif defined(__ARM_ARCH_6T2_) || defined(__ARM_ARCH_6T2_)
    #define ARM
    #define ARM6T2
#elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
    #define ARM
    #define ARM6
#elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7
#elif defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7
    #define ARM7A
#elif defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7
    #define ARM7R
#elif defined(__ARM_ARCH_7M__)
    #define ARM
    #define ARM7
    #define ARM7M
#elif defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7
    #define ARM7S
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARM
    #define ARM64
#endif

typedef float (*timedFunF)(float n);
typedef double (*timedFunD)(double n);
typedef void (*timedFun)(void *n, void *result);

union unDouble {
    double f;
    uint64_t i;
};

union unFloat {
    float f;
    uint32_t i;
};

static const uint32_t FLOAT_E_MAX = FLOAT_E_MASK >> FLOAT_SHIFT;
static const uint32_t FLOAT_IMPLIED_BIT = FLOAT_M_MASK + 1U;

static const uint64_t DOUBLE_E_MAX = DOUBLE_E_MASK >> DOUBLE_SHIFT;
static const uint64_t DOUBLE_IMPLIED_BIT = DOUBLE_M_MASK + 1LU;

static long double rollingTimeAvgs[TIMING_RUNS];

void findDeltaTime(int idx  __attribute__((aligned(16))), const struct timespec *__restrict__ tstart, const struct timespec *__restrict__ tend);

void timeFun(timedFun fun, void *__restrict__ s, void **__restrict__ result, int i  __attribute__((aligned(16))));

union unFloat __attribute__((aligned(16))) timeFunf(timedFunF fun, float s __attribute__((aligned(16))), int i  __attribute__((aligned(16))));

union unDouble __attribute__((aligned(16))) timeFund(timedFunD fun, double s __attribute__((aligned(16))), int i __attribute__((aligned(16))));

void __attribute__((aligned(16))) printTimedRuns(char **__restrict__ runNames, const uint32_t length __attribute__((aligned(16))));

int __attribute__((aligned(16))) signumf(float y __attribute__((aligned(16))));

int __attribute__((aligned(16))) signum(double y __attribute__((aligned(16))));

uint64_t __attribute__((aligned(16))) findMsb(uint64_t n __attribute__((aligned(16))));

int __attribute__((aligned(16))) bitScanReverse(uint64_t bb __attribute__((aligned(16))));

float __attribute__((aligned(16))) sqrtApproxf(const float z __attribute__((aligned(16))));

double __attribute__((aligned(16))) sqrtApprox(const double z __attribute__((aligned(16))));

float __attribute__((aligned(16))) dividByPow2f(float x __attribute__((aligned(16))), int16_t n __attribute__((aligned(16))));

float __attribute__((aligned(16))) aatan(const float z __attribute__((aligned(16))));

float __attribute__((aligned(16))) sqrtApproxf(const float z __attribute__((aligned(16))));

double __attribute__((aligned(16))) sqrtApprox(const double z __attribute__((aligned(16))));

void aatan2(const float y __attribute__((aligned(16))), const float x __attribute__((aligned(16))), float *__restrict__ result);

float __attribute__((aligned(16))) aatanTwo(float  __attribute__((aligned(16))) y, float  __attribute__((aligned(16))) x);

void flipAbsMaxMin(float *__attribute__((aligned(16))) x, float *__attribute__((aligned(16))) y);

// TODO implement double precision version of below
__asm__( // ffabsf
#ifdef __clang__
"_ffabsf: "
#else
"ffabsf: "
#endif
    "movq %xmm0, %rax\n\t"
    "andl $0x7FFFFFFF, %eax\n\t"
    "movq %rax, %xmm0\n\t"
    "ret"
);

__asm__ ( // fsqrt
#ifdef __clang__
"_fsqrt: "
#else
"fsqrt: "
#endif
#ifndef X86
//TODO implement below in ARM asm
#else
    "subq $16, %rsp\n\t"
    "movdqu %xmm0, (%rsp)\n\t"

    "fldl (%rsp)\n\t"
    "fsqrt\n\t"
    "fstpl (%rsp)\n\t"

    "movdqu (%rsp), %xmm0\n\t"
    "addq $16, %rsp\n\t"

    "ret"
#endif
);

__asm__ (// fsqrtf
#ifdef __clang__
"_fsqrtf: "
#else
"fsqrtf: "
#endif
#ifndef X86
//TODO implement below in ARM asm
#else
    "subq $16, %rsp\n\t"
    "movdqu %xmm0, (%rsp)\n\t"

    "flds (%rsp)\n\t"
    "fsqrt\n\t"
    "fstps (%rsp)\n\t"

    "movdqu (%rsp), %xmm0\n\t"
    "addq $16, %rsp\n\t"

    "ret"
#endif
);

__asm__( // isNegZero
#ifdef __clang__
"_isNegZero: "
#else
"isNegZero: "
#endif
    "movq %xmm0, %rax\n\t"
    "andl $0x80000000, %eax\n\t"
    "ret"
);

__asm__ (
#ifdef __APPLE_CC__
"_swap2: "
#else
"swap2: "
#endif
    "movq (%rsi), %xmm0\n\t"
    "movq (%rdi), %xmm1\n\t"
    "movq %xmm1, (%rsi)\n\t"
    "movq %xmm0, (%rdi)\n\t"
    "ret"
);

__asm__( // absMaxMin
#ifdef __clang__
"_absMaxMin: "
#else
"absMaxMin: "
#endif
    "movl (%rdi), %eax\n\t"
    "movl (%rsi), %ecx\n\t"
    "andl $0x7FFFFFFF, %eax\n\t"
    "andl $0x7FFFFFFF, %ecx\n\t"

    "movq %rax, %xmm0\n\t"
    "movq %rcx, %xmm1\n\t"

    "comiss %xmm1, %xmm0\n\t"
    "jna flip\n\t"

    "movl %eax, (%rdi)\n\t"
    "movl %ecx, (%rsi)\n\t"
    "ret\n\t"

"flip: "
    "movl %ecx, (%rdi)\n\t"
    "movl %eax, (%rsi)\n\t"
    "ret"
);
#endif //TIMED_FUNCTIONS_H
