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
#include <emmintrin.h>
#include <math.h>
#define TIMING_RUNS 4
#include "timed_functions.h"
#define MAX_ERROR 1e-6
#define MIN -2
#define MAX 2
#define STEP 1
//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
    #define PRINTF printf
#else
    #define PRINTF //
#endif

extern float asmArgz_ps(__m128 *z0, __m128 *z1);
__asm__(
#ifdef __APPLE_CC__
"_asmArgz_ps: "
#else
"asmArgz_ps: "
#endif
    "vmovaps	(%rdi), %xmm0           \n\t"
    "vmulps     (%rsi), %xmm0,  %xmm0   \n\t"
    "vmovaps	%xmm0,  (%rdi)          \n\t"
    "vpermilps	$177,   %xmm0,  %xmm0   \n\t"
    "vmovaps	%xmm0,  (%rsi)          \n\t"
    "vaddsubps	(%rdi), %xmm0,  %xmm1   \n\t"
    "vpermilps	$240,   %xmm1,  %xmm0   \n\t"
    "vmovaps	%xmm0,  (%rdi)          \n\t"
    /* xmm -> x87 "faptain" section */
    "sub        $16,    %rsp            \n\t"   // emulate push, s.t. xmm can be accessed by the fpu
    "vextractps $3,     %xmm0,  (%rsp)  \n\t"   // extractps seems to work better here,
//  "vmovhps    %xmm0,  (%rsp)          \n\t"   // which corresponds to the latency
    "flds       (%rsp)                  \n\t"   // and throughput tables from intel.
    "sub        $16,    %rsp            \n\t"   // albeit marginally, that is.
    "vextractps $0,     %xmm0,  (%rsp)  \n\t"
    "flds       (%rsp)                  \n\t"
    "fpatan                             \n\t"
    "fstps      (%rsp)                  \n\t"
    // B I G pop emulation
    "vmovq    (%rsp), %xmm0             \n\t"
    "add        $32,    %rsp            \n\t"
    "ret"
);

extern void conjz(__m128 *z);
__asm__ (
#ifdef __APPLE_CC__
"_conjz: "
#else
"conjz: "
#endif
    "movq $0x100003f20, %rcx\n\t"
    "pushq %rcx\n\t"
    "vbroadcastss (%rsp), %xmm2\n\t" // all ones
    "subps (%rdi), %xmm2\n\t"
    "vmovaps %xmm2, (%rdi)\n\t"
    "pop %rcx\n\t"
    "ret"
);

union vect {
    float arr[4] __attribute__((aligned(16)));
    __m128 vect;
};

void conjz_ps(__m128 *z) {

    union vect T = {-1, -1, -1, -1};
    *z = _mm_mul_ps(*z, T.vect);
//    *z = _mm_permute_ps(*z, _MM_SHUFFLE(3,2,1,0));
}

static long errs;

static inline float approximateAtan2(const float z, const float y) {
    static const float M_3_PI_2 = 3.f * M_PI_2;
    // undefined mathematically ***this differs from the atan2 built-in***
    if (!z && !y) {
        return NAN;
    }

    if (!z) {
        return y > 0.f ? M_PI_2 : -M_PI_2;
    }

    if (!y && z < 0.f) {
        return (*(uint32_t *)&y & 0x80000000) ? -M_PI : M_PI;
    }

    const float x = z < 0.f && y > 0.f ? z/y: y/z;
    const float magX = fabs(x);
    float result = x * (M_PI_4 - (magX - 1) * (0.2447 + 0.0663 * magX));

    if (z < 0.f) {
        result = y < 0.f ? result - M_PI : M_PI_2 - result;
    }

    // z >= 0 at this point
    return result;

//    if (fabs(y) > fabs(z)) return M_PI_2*x - result; //result = M_PI_2 - result;
//    if (x < 0.f) return  M_PI*x - result; //result = M_PI - result;
//    if (y < 0.f) return -M_PI_2*x + result;  //result = -result;
//
//    return M_PI_4*x - result;
}

float argz_ps(__m128 *z0, __m128 *z1){// z0 = (ar, aj), z1 = (br, bj) => z0*z1 = (ar*br - aj*bj, ar*bj + br*aj)

    *z0 = _mm_mul_ps(*z0, *z1);                // => ar*br, aj*bj, ar*bj, br*aj
    *z1 = _mm_permute_ps(*z0, _MM_SHUFFLE(2,3,0,1));
    *z0 = _mm_addsub_ps(*z1, *z0);
    *z0 = _mm_permute_ps(*z0, _MM_SHUFFLE(3,3,0,0));

    long temp = _mm_extract_ps(*z0, 3);
    float y = *(float*)&temp;
    temp = _mm_extract_ps(*z0, 0);
    float x = *(float*)&temp;
    float approxAtan2 = approximateAtan2(x, y);
    float arg= atan2f(y, x);


    if (fabs(arg - approxAtan2) > M_2_PI) {
        int sgn = signum(y);
        printf( "*** for (%.2f + %.2fi) phase: %f, approximated phase: %f%s***\n", x, y, arg, approxAtan2,
                (signum(x) & sgn) ? " " : " x and y of opposite sign ");
//        if (sgn == -0) printf("NEGATIVE ZERO\n");
        errs++;
    }

    return arg;
}

//float argz(float ar, float aj, float br, float bj, __m128 *z0) {
//    float result;
//    union vect u = { .arr =  { aj,ar,aj,ar }};
//    union vect v = {.arr = { bj,br,br,bj }};
//
//    result = argz_ps(&u.vect, &v.vect);
//    if (z0 != NULL) {
//        *z0 = _mm_permute_ps(u.vect, _MM_SHUFFLE(3,3,0,0));
//    }
//
//    return result;
//}

int main(void){
    static char *runNames[TIMING_RUNS] = {"asmArgz_ps :: ", "argz_ps :: ", "conjz :: ", "conjz_ps :: "};\
    float ar, aj, br, bj, deltaArg, deltaR, deltaJ, deltaConjR, deltaConjJ, approxAtan2;
    float arg[2], zr[2], zj[2], cnjR[2], cnjJ[2];
    struct timespec tstart, tend;
    uint64_t n = 0;

    for (ar = MIN; ar < MAX; ++ar)
        for (aj = MIN; aj < MAX; ++aj)
            for (br = MIN; br < MAX; ++br)
                for (bj = MIN; bj < MAX; ++bj, ++n){
                    union vect u = { aj,ar,aj,ar };
                    union vect v = { bj,br,br,bj };

                    union vect u1 = { aj,ar,aj,ar };
                    union vect v1 = { bj,br,br,bj };

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    arg[0] = asmArgz_ps(&u.vect, &v.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(0, &tstart, &tend);
                    // END TIMED

                    zr[0] = u.vect[0];
                    zj[0] = u.vect[3];
                    PRINTF("Iteration: %d\n", n);

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    conjz(&u.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(2, &tstart, &tend);
                    // END TIMED

                    cnjR[0] = u.vect[0];
                    cnjJ[0] = u.vect[3];
                    PRINTF("conjugate: (%.1f + %.1fi)\n",  u.vect[0], u.vect[3]);

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    arg[1] = argz_ps(&u1.vect, &v1.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(1, &tstart, &tend);
                    // END TIMED

                    zr[1] = u1.vect[0];
                    zj[1] = u1.vect[3];
                    PRINTF("(%.1f + %.1fi) . (%.1f + %.1fi)",  ar, aj, br, bj);
                    PRINTF(" = (%.1f + %.1fi)\nphase: %f\n", zr[1], zj[1], arg[1]);

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    conjz_ps(&u1.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(3, &tstart, &tend);
                    // END TIMED

                    cnjR[1] = u1.vect[0];
                    cnjJ[1] = u1.vect[3];
                    PRINTF("conjugate: (%.1f + %.1fi)\n\n",  u1.vect[0], u1.vect[3]);

                    deltaR = fabs(zr[0] - zr[1]);
                    deltaJ = fabs(zj[0] - zj[1]);
                    deltaArg = fabs(arg[0] - arg[1]);
                    deltaConjR = fabs(cnjR[0] - cnjR[1]);
                    deltaConjJ = fabs(cnjJ[0] - cnjJ[1]);
                    assert(deltaArg < MAX_ERROR);
                    assert(deltaR < MAX_ERROR);
                    assert(deltaJ < MAX_ERROR);
                    assert(deltaConjR < MAX_ERROR);
                    assert(deltaConjJ < MAX_ERROR);
                }

    printTimedRuns(runNames, TIMING_RUNS);
    printf("Iterations: %ld, Approximation errors: %ld, Percent: %.2f%%\n", n, errs, 100.0*errs/n);
}