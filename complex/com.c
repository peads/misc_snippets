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

#include <immintrin.h>
//#include <emmintrin.h>

#define TIMING_RUNS 4
#include "timed_functions.h"

#define MAX_ITERATIONS 16
#define MAX_ERROR 1e-15L
#define _mm_mul_pi32(u1, v1) _mm_mulhi_pi16(u1, v1) << 16 | _mm_mullo_pi16(u1, v1)
#define _mm_extract_pi32(u0, n) _mm_extract_pi16(u0, n) << 16 | _mm_extract_pi16(u0, n-1)

int8_t DEBUG = 0;

typedef double (*phaseFun)(double vr, double vj);

struct phaseArgs {
    struct complexArgs *cargs;
    phaseFun fun;
};

struct complexArgs {
    int ar;
    int aj;
    int br;
    int bj;
};

struct {
    uint64_t totalErrors;
    long double maxAvgErr;
    long double namesWorst;
    const char *name;
} errorInfo;

static long double absRes;

const char *runNames[TIMING_RUNS]
        = {"calcAtan2 :: ", "calcVectPd :: ",
            "polarAvx16 :: ", "caclVect :: "};//, "esbensen :: "};

static inline double norm(double x, double y) {

    return sqrt(x*x + y*y);
}

static inline void multiply(double xr, double xj, 
                            double yr, double yj, 
                            double *resultR, double *resultJ) {
    *resultR = xr*yr - xj*yj;
    *resultJ = xr*yj + xj*yr;
}

static inline double lambdaAsin(double vr, double vj) {

    return asin(vr / norm(vr, vj));
}

static inline double lambdaAtan2(double vr, double vj) {

    return atan2(vr, vj);
}


static void calcVectPd(struct complexArgs *cargs, double *result) {
    union vect {
        __m256d vect;
    };

    union vect u = {cargs->ar, cargs->aj, cargs->ar, cargs->br};
    union vect v = {cargs->br, cargs->bj, cargs->bj, cargs->aj};

    double zr;

    u.vect = _mm256_mul_pd(u.vect, v.vect);
    u.vect = _mm256_addsub_pd(u.vect, _mm256_permute_pd(u.vect, 0b0101));
    zr = u.vect[0];
    u.vect = _mm256_mul_pd(u.vect, u.vect);
    u.vect = _mm256_permute_pd(u.vect, _MM_SHUFFLE(3, 2, 1, 0));
    u.vect = _mm256_add_pd(u.vect, _mm256_permute2f128_pd(u.vect, u.vect, 1));
    u.vect = _mm256_sqrt_pd(u.vect);

    *result = asin(zr / u.vect[0]);
    if (DEBUG > 0) printf("calcVectPd ::\t\tphase: %f zr: %f, r: %f\n", *result, zr, u.vect[0]);
    assert(fabsl(absRes - fabsl(*result)) < MAX_ERROR);
}

static void polarAvx16(void *args, double *result) {

    struct complexArgs *cargs = args;

    double zr;
    double r;
    __m64 u1 = _mm_set_pi32(cargs->ar, cargs->ar);
    __m64 v1 = _mm_set_pi32(cargs->br, cargs->bj);
    __m64 u0 = _mm_set_pi32(cargs->aj, cargs->br);
    __m64 v0 = _mm_set_pi32(cargs->bj, cargs->aj);

    v1 = _mm_mul_pi32(u1, v1);
    v0 = _mm_mul_pi32(u0, v0);

    u1 = _mm_sub_pi32(v1, v0);
    u0 = _mm_add_pi32(v1, v0);


    zr = _mm_extract_pi32(u1, 3);//_mm_extract_pi16(u1, 3) << 16 | _mm_extract_pi16(u1, 2);
//    zr = _m_pextrw(u1, 3) << 16 | _m_pextrw(u1, 2);
//    zr = _mm_extract_pi16(u1, 1);
//    zr = _m_pextrw(u1, 1);
//    zr = _mm_extract_pi16(u1, 0);
//    zr = _m_pextrw(u1, 0);

    u0 = _mm_mul_pi32(u0, u0);
    u1 = _mm_mul_pi32(u1, u1);
    u0 = _mm_add_pi32(u0,_mm_shuffle_pi16(u1, _MM_SHUFFLE(1, 0, 1, 0)));

    r = sqrt(_mm_extract_pi32(u0, 1));//_mm_extract_pi16(u0, 0);
    *result = asin(zr / r);

    if (DEBUG > 0) {

        printf("polarAvx16 ::\t\tphase: %f zr: %f, r: %f\n", *result, zr, r);
    }
}

static void calcVect(void *args, double *result) {

    struct complexArgs *cargs = args;

    double zr;
    double r;

    union vect {
        __m128i vect;
    };

    union vect u1 = {cargs->ar, cargs->ar};
    union vect v1 = {cargs->br, cargs->bj};
    union vect u0 = {cargs->aj, cargs->br};
    union vect v0 = {cargs->bj, cargs->aj};

    v1.vect = _mm_mul_epi32(u1.vect, v1.vect); // => {ar*br, ar*bj}
    v0.vect = _mm_mul_epi32(u0.vect, v0.vect); // => {aj*bj, br*aj}

    u1.vect = _mm_sub_epi64(v1.vect,    // => {ar*br - aj*bj, ar*bj - br*aj}
                            v0.vect);   // *we don't care about the second entry

    u0.vect = _mm_add_epi32(v1.vect,    // => {ar*br + aj*bj, ar*bj + br*aj}
                            v0.vect);   // *we don't care about the first entry

    zr = u1.vect[0];

    u0.vect = _mm_mul_epi32(u0.vect, u0.vect);
    u1.vect = _mm_mul_epi32(u1.vect, u1.vect);
    u0.vect = _mm_add_epi32(u0.vect,
                            _mm_shuffle_epi32(u1.vect, _MM_SHUFFLE(1, 0, 1, 0)));

    r = sqrt(u0.vect[1]);
    *result = asin(zr / r);

    if (DEBUG > 0) printf("calcVect ::\t\t\tphase: %f zr: %f, r: %f\n", *result, zr, r);
    assert(fabsl(absRes - fabsl(*result)) < MAX_ERROR);
}

static void calcResult(struct phaseArgs *pargs, double *result) {
    
    struct complexArgs *cargs = pargs->cargs;

    double vr;// = cargs->ar * cargs->br - cargs->aj * cargs->bj;
    double vj;// = cargs->aj * cargs->br + cargs->ar * cargs->bj;

    multiply(cargs->ar, cargs->aj, cargs->br, cargs->bj, &vr, &vj);

    *result = pargs->fun(vr, vj);

    if (DEBUG > 0) printf("calcAsin/Atan2 ::\tphase: %f zr: %f, r: %f\n", *result, vr, sqrt(vr*vr + vj*vj));
}

static void calcAsin(void *args, double *result) {

    struct complexArgs *cargs = args;
    struct phaseArgs pargs;
    pargs.cargs = cargs;
    pargs.fun = lambdaAsin;

    calcResult(&pargs, result);
    assert(fabsl(absRes - fabsl(*result)) < MAX_ERROR);
}

static void calcAtan2(void *args, double *result) {

    struct complexArgs *cargs = args;
    struct phaseArgs pargs;
    pargs.cargs = cargs;
    pargs.fun = lambdaAtan2;

    calcResult(&pargs, result);
    absRes = fabsl(*result);
}

//void esbensen(void *args, double *result)
///*
//  input signal: s(t) = a*exp(-i*w*t+p)
//  a = amplitude, w = angular freq, p = phase difference
//  solve w
//  s' = -i(w)*a*exp(-i*w*t+p)
//  s'*conj(s) = -i*w*a*a
//  s'*conj(s) / |s|^2 = -i*w
//*/
//{
//
//    const int32_t scaled_pi = 2608;// 1<<14 / (2*pi) */
//    struct complexArgs *cargs = args;
//
//    int cj, dr, dj;
//    int br = (int) cargs->br;
//    int ar = (int) cargs->ar;
//    int bj = (int) cargs->bj;
//    int aj = (int) cargs->aj;
//
//    dr = (br - ar) << 1;
//    dj = (bj - aj) << 1;
//    cj = bj * dr - br * dj; /* imag(ds*conj(s)) */
//
//    int32_t denom = ar*ar + aj*aj + 1;
//    printf("dr = %d dj = %d cj = %d dividing %d * %d / %d\n", dr, dj, cj, scaled_pi, cj, denom);
//    double d = (double) cj / denom;
//
//    *result = sin(scaled_pi * d) / 2.0;
//}

static void mult(int ar, int aj, int br, int bj/*, int *vr, int *vj*/) {

//    long double avgError;
//    long double localMax;
    struct complexArgs cargs;
    double results[TIMING_RUNS];
    uint32_t i;

    cargs.aj = aj;
    cargs.ar = ar;
    cargs.br = br;
    cargs.bj = bj;

    timeFun((timedFun) calcAtan2, &cargs, (void *) &results, 0);
    timeFun((timedFun) calcVectPd, &cargs, (void *) &results, 1);
    timeFun((timedFun) polarAvx16, &cargs, (void *) &results, 2);
    timeFun((timedFun) calcVect, &cargs, (void *) &results, 3);
//    timeFun((timedFun) esbensen, &cargs, (void *)&results, 4);
//    long double errs[TIMING_RUNS - 1] = {absRes-fabsl(results[1]),
//                           absRes-fabsl(results[2]),
//                           absRes-fabsl(results[3]),
//                           };//results[0]-results[4]};

//    avgError = errs[0];
//    avgError *= errs[1];
//    avgError *= errs[2];
//    avgError = cbrtl(avgError);

//    if (avgError > errorInfo.maxAvgErr) {
//        errorInfo.maxAvgErr = avgError;
//    }
//
//        for (localMax = -1.0, i = 0; i < 3; ++i) {
//            if (errs[i] > localMax) {
//                localMax = errs[i];
//                errorInfo.name = runNames[i+1];
//                errorInfo.namesWorst = localMax;
////                errorInfo.totalErrors++;
//            }
//        }

    if (DEBUG > -1) {
        double zr, zj;
        multiply(ar, aj, br, bj, &zr, &zj);
        printf("(%d + %di) . (%d + %di) = (%f + %fi)\n"
               "phase: %f, %f, %f, %f\n",
               ar, aj,
               br, bj,
               zr, zj,
               results[0], results[1], results[2], results[3]);

    }
//    if (DEBUG > 0) printf("Max error tolerance: %LE, %s%LE\n", MAX_ERROR, errorInfo.name, errorInfo.namesWorst);
//    assert(/*errs[0] < MAX_ERROR &&*/ errs[1] < MAX_ERROR && errs[2] < MAX_ERROR);
}

int main(const int argc, const char **argv) {

    if (argc > 1) {
        switch (argv[1][0]) {
            case 'v':
                DEBUG = 1;
                break;
            case 'q':
                DEBUG = -1;
            default:
                break;
        }
    }
    errorInfo.totalErrors = 0;
    errorInfo.namesWorst = -1.0;
    errorInfo.maxAvgErr = -1.0;
    errorInfo.name = "unset";
    uint32_t i,j,k,m;
    double f,n,p,q;
    for (i = 1, f = -10.0; i < MAX_ITERATIONS; ++i,             f += 0.01) {
        for (j = 1, n = -10.0; j < MAX_ITERATIONS; ++j,         n += 0.01) {
            for (k = 1, p = -10.0; k < MAX_ITERATIONS; ++k,     p += 0.01) {
                if (0 == i && 0 == k) continue;
                for (m = 1, q = -10.0; m < MAX_ITERATIONS; ++m, q += 0.01) {
                    if (0 == j && f == -k) continue;
                    mult(i,j,k,m);
                    //mult(f,n,p,q);
                }
            }
        }
    }

    mult((int16_t) 1, (int16_t) 2, (int16_t) 3, (int16_t) 4);
    mult(5, 6, 7, 8);
    mult(9, 10, 11, 12);

    printTimedRuns(runNames, TIMING_RUNS);
    printf( "Worst: %s %Le\n"
            "Max avg error: %Le, total above threshold: %lu, total calculations: %lu\n", 
        errorInfo.name, errorInfo.namesWorst, errorInfo.maxAvgErr, errorInfo.totalErrors, 
        MAX_ITERATIONS*MAX_ITERATIONS*MAX_ITERATIONS*MAX_ITERATIONS);
    return 0;
}
