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
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

typedef union {
    float f;
    uint32_t i; 
} val;

static time_t rollingTimeSums[5];

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

float invSqrt(float x) {

    float xhalf = 0.5f * x;
    int i = *(int*)&x;            // store floating-point bits in integer
    i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
    x = *(float*)&i;              // convert new bits into float
    x = x*(1.5f - xhalf*x*x);     // One round of Newton's method

    return x;
}

float asmInvSqrt(float x) {

    int i;

    __asm__ __volatile__(
        "mov %%rdx, %0\n\t"
        "sar $1, %%rdx\n\t"  
        "sub $1597463007, %%rdx\n\t"
        "mov %1, %%rdx"
        : "=m" (i) : "m" (x)
    );

    return x;
}

float fastSqrtf(float n) { 
    __asm__ __volatile__(
        "sqrtss  %1, %0" : "=x" (n) : "x" (n)
    );
    return n;
}

double fastSqrt(double n) {
    __asm__ __volatile__(
        "sqrtsd %1, %0" : "=x" (n) : "x" (n)
    );
    return n;
}

double fsqrt(double n) {
    __asm__ __volatile__(
        "fldl %0\n\t"
        "fsqrt\n\t"
        "fstpl %0\n\t"
        : "=m" (n) : "m" (n)
    );

    return n;
}

val timeFastf(float n) {

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    n = fastSqrtf(n);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(4, tstart, tend, timediff);

    free(timediff);

    return (val){n};
}

//float newtonSqrt(float s) {

    //const float inv2s = u64tod_inv(s * 2.0);

    //float x = 1.;
    //int i = 0;

    //for (; i < 5; ++i) {
    //    x = x - (x*x - s) * inv2s;
    //}
    //// x = x - (x*x - s) / 2*s;

    //return x;
//}

double sqrt_approxd(double z) {

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
    un.j -= ((uint64_t)1) << 52;                /* Subtract 2^m. */ 
    un.j >>= 1;                                  /* Divide by 2. */           
    un.j += ((uint64_t)1) << 61;                /* Add ((b + 1) / 2) * 2^m. */           

    return un.f;        /* Interpret again as float */
}

val timeSqrt(float s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = sqrt(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(0, tstart, tend, timediff);

    free(timediff);

    return (val){s};
}

val timeFastSqrt(float s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    float x = fastSqrt(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(1, tstart, tend, timediff);

    free(timediff);

    return (val){x};
}

val timeApproxSqrtD(double s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    double x = sqrt_approxd(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(4, tstart, tend, timediff);

    free(timediff);

    return (val){x};
}

val timeFsqrt(double s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    s = fsqrt(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(2, tstart, tend, timediff);

    free(timediff);

    return (val){s};
}

val timeInvInvSqrt(float s) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    float x = 1./invSqrt(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(3, tstart, tend, timediff);

    free(timediff);

    return (val){x};
}

int main(void) {

    float x = 1.0;
    double errDiffSums[4];

    for (; x < 1000001.0; ++x) {
        val s3 = timeSqrt(x);
        //printf("propper :: sqrt[%f] = %f %X\n", x, s3.f, s3.i);

        val s4 = timeFastSqrt(x);
        //printf("fast :: sqrt[%f] = %f %X\n", x, s4.f, s4.i);

        val s5 = timeFsqrt(x);
        //printf("approx :: sqrt[%f] = %f %X\n", x, s5.f, s5.i);

        val s6 = timeInvInvSqrt(x);
        //printf("fsqrt :: sqrt[%f] = %f %X\n", x, s6.f, s6.i);

        val s7 = timeFastf(x);
        //printf("fastf :: sqrt[%f] = %f %X\n\n", x, s7.f, s7.i);

        errDiffSums[0] += fabs((double) s3.f - (double) s4.f);
        errDiffSums[1] += fabs((double)s3.f - (double) s5.f);
        errDiffSums[2] += fabs((double)s3.f - (double)s6.f);
        errDiffSums[3] += fabs((double)s3.f - (double)s7.f);
    }


    int i = 0;
    char *runNames[5] = {"propper :: ", "fast :: ", "fsqrt :: ", "dual inverse :: ", "fastf :: "};

    for (; i < 5; ++i) {
        const char *runName = runNames[i];
        printf("%sAverage time: %f ns\n", runName, rollingTimeSums[i] / x);

        if (i > 0) {
            printf("%sAverage error: %f\n", runName, errDiffSums[i-1] / x);
        }
        printf("\n");
    }
    return 0;
}

