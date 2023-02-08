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
    #define ARM7A
#elif defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7R
#elif defined(__ARM_ARCH_7M__)
    #define ARM
    #define ARM7M
#elif defined(__ARM_ARCH_7S__)
    #define ARM
    #define ARM7S
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARM
    #define ARM64
#else
    #error "Unsupported architecture!"
#endif

#define TIMING_RUNS 7

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

typedef union {
    float f;
    uint32_t i; 
} val;

static time_t rollingTimeSums[TIMING_RUNS];

void findDeltaTime(int idx, struct timespec tstart, struct timespec tend, char *timediff) {
    time_t deltaTsec = tend.tv_sec - tstart.tv_sec;
    time_t deltaTNanos = tend.tv_nsec - tstart.tv_nsec;

    if (deltaTsec > 0) {
        sprintf(timediff, "%lu.%lu s",deltaTsec, deltaTNanos);
    } else {
        sprintf(timediff, "%lu ns", deltaTNanos);
        rollingTimeSums[idx] += deltaTNanos;
    }
}

//double u64tod_inv( uint64_t u64 ) {                                 
//    __asm__( "#annot0" );                                      
//    union {                                                    
//        double f;                                                
//        struct {                                                 
//            unsigned long m:52; // careful here with endianess     
//            unsigned long x:11;                                    
//            unsigned long s:1;                                     
//        } u64;                                                   
//        uint64_t u64i;                                           
//    } z,                                                       
//          magic0 = { .u64 = { 0, (1<<10)-1 + 52, 0 } },        
//          magic1 = { .u64 = { 0, (1<<10)-1 + (52+12), 0 } },   
//          magic2 = { .u64 = { 0, 2046, 0 } };                  
//
//    __asm__( "#annot1" );                                      
//    if( u64 < (1UL << 52UL ) ) {                               
//        z.u64i = u64 + magic0.u64i;                              
//        z.f   -= magic0.f;                                       
//    } else {                                                   
//        z.u64i = ( u64 >> 12 ) + magic1.u64i;                    
//        z.f   -= magic1.f;                                       
//    }                                                          
//    __asm__( "#annot2" );                                      
//
//    z.u64i = magic2.u64i - z.u64i;                             
//
//    return z.f;                                                
//}

float sqrt_approx(float z) {   

    val un = {z};	/* Convert type, preserving bit pattern */
    /*
     * To justify the following code, prove that
     *
     * ((((val.i / 2^m) - b) / 2) + b) * 2^m = ((val.i - 2^m) / 2) + ((b + 1) / 2) * 2^m)
     *
     * where
     *
     * b = exponent bias
     * m = number of mantissa bits
     */
    un.i -= 1 << 23;	/* Subtract 2^m. */
    un.i >>= 1;		/* Divide by 2. */
    un.i += 1 << 29;	/* Add ((b + 1) / 2) * 2^m. */

    return un.f;		/* Interpret again as float */
}

double sqrt_approxd(double z) {
#if !defined(ARM64) && !defined(X86)
    return sqrt_approx(z);
#else
    union {
        double f;
        uint64_t j;
    } un = {z};         /* Convert type, preserving bit pattern */

    /*
     * To justify the following code, prove that
     *
     * ((((val.i / 2^m) - b) / 2) + b) * 2^m = ((val.i - 2^m) / 2) + ((b + 1) / 2) * 2^m)
     *
     * where
     *
     * m = number of mantissa bits => 52
     * b = exponent bias => 1023 => ((b + 1) / 2) * 2^m = (1024 * 2^(51)) = 2^(10+51) = 1 << 61
     */
    un.j -= 1LU << 52;                /* Subtract 2^m. */ 
    un.j >>= 1;                                  /* Divide by 2. */           
    un.j += 1LU << 61;                /* Add ((b + 1) / 2) * 2^m. */           

    return un.f;        /* Interpret again as float */
#endif
}

float vectorSqrtf(float n) {

    __asm__ __volatile__(
#ifdef X86
        "sqrtss  %0, %0" 
#else
        "VSQRT.F32 %0, %0"        
#endif
        : "+x" (n)
    );
    return n;
}

double vectorSqrt(double n) {
#if !defined(x86_64) && !defined(ARM64)
    return vectorSqrtf(n); 
#else
    __asm__ __volatile__(
#ifdef X86
        "sqrtsd %0, %0" 
#else
        "VSQRT.F64 %0, %0"
#endif
        : "+x" (n)
    );

    return n;
#endif
}

float fsqrtf(float n) {
#ifndef X86
    return vectorSqrtf(n);
#else
    __asm__ __volatile__(
        "flds %1\n\t"
        "fsqrt\n\t"
        "fstps %0\n\t"
        : "=m" (n) : "m" (n)
    );

    return n;
#endif
}

double fsqrt(double n) {
#ifndef X86
    return vectorSqrt(n);
#else
#ifndef x84_64
    return fsqrtf(n);
#else
    __asm__ __volatile__(
        "fldl %1\n\t"
        "fsqrt\n\t"
        "fstpl %0\n\t"
        : "=m" (n) : "m" (n)
    );

    return n;
#endif
#endif
}

typedef float (*timedFunF)(float n);
typedef double (*timedFunD)(double n);

val timeFunD(timedFunD fun, double s, int i) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(i, tstart, tend, timediff);

    free(timediff);

    return (val){s};

}

val timeFunF(timedFunF fun, float s, int i) {

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(i, tstart, tend, timediff);

    free(timediff);

    return (val){s};
}

int main(void) {

    int j;
    int i = 0;
    float x = 1.0;
    double actual;
    double errDiffSums[TIMING_RUNS];
    char *runNames[TIMING_RUNS] 
        = {"approx :: ", "approxd :: ", "vsqrt :: ", "vsqrtd :: ", "fsqrt :: ", "fsqrtf :: ", "c_sqrt_fn :: "};
    val s[TIMING_RUNS];

    for (; i < 1000001; ++i, x += 0.01) {

        s[0] = timeFunF(sqrt_approx, x, 0);
        s[1] = timeFunD(sqrt_approxd, x, 1);
        s[2] = timeFunF(vectorSqrtf, x, 2);
        s[3] = timeFunD(vectorSqrt, x, 3);
        s[4] = timeFunD(fsqrt, x, 4);
        s[5] = timeFunF(fsqrtf, x, 5);
        s[6] = timeFunD(sqrt, x, 6);
        actual = s[6].f;
        //printf("approx :: sqrt[%f] = %f %X\n", x, s5.f, s5.i);
        //printf("fast :: sqrt[%f] = %f %X\n", x, s4.f, s4.i);
        //printf("propper :: sqrt[%f] = %f %X\n", x, s3.f, s3.i);
        //printf("fsqrt :: sqrt[%f] = %f %X\n", x, s6.f, s6.i);
        //printf("fastf :: sqrt[%f] = %f %X\n\n", x, s7.f, s7.i);

        for (j = 0; j < TIMING_RUNS - 1; ++j) {
            errDiffSums[j] += fabs(actual - (double) s[j].f);
        }
    }

    for (j = 0; j < TIMING_RUNS; ++j) {
        const char *runName = runNames[j];
        printf("%sAverage time: %f ns\n", runName, rollingTimeSums[j] / ((double) i));
        printf("%sAverage error: %f\n\n", runName, errDiffSums[j] / ((double) i));
    }
    return 0;
}

