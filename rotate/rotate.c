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
#define MIN -50
#define MAX 50
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
    "vmovaps	(%rdi), %xmm0\n\t"
    "vmulps	(%rsi), %xmm0, %xmm0\n\t"
    "vmovaps	%xmm0, (%rdi)\n\t"
    "vpermilps	$177, %xmm0, %xmm0\n\t"
    "vmovaps	%xmm0, (%rsi)\n\t"
    "vaddsubps	(%rdi), %xmm0, %xmm1\n\t"
    "vpermilps	$240, %xmm1, %xmm0\n\t"
    "vmovaps	%xmm0, (%rdi)\n\t"

    // push
    "sub $16, %rsp\n\t"
    "vextractps $3, %xmm0, (%rsp)\n\t"
    // /push
    "flds (%rsp)\n\t" // push top of CPU stack to x87 stack
    // push
    "sub $16, %rsp\n\t"
    "vextractps $0, %xmm0, (%rsp)\n\t"
    // /push
    "flds (%rsp)\n\t"
    //pop
    "vmovq (%rsp), %xmm0\n\t"
    "add $16, %rsp\n\t"
    // /pop
    "fpatan\n\t"
    "fstps (%rsp)\n\t"
    // pop
    "vmovq (%rsp), %xmm0\n\t"
    "add $16, %rsp\n\t"
    // /pop
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

float argz_ps(__m128 *z0, __m128 *z1){// z0 = (ar, aj), z1 = (br, bj) => z0*z1 = (ar*br - aj*bj, ar*bj + br*aj)

    *z0 = _mm_mul_ps(*z0, *z1);                // => ar*br, aj*bj, ar*bj, br*aj
    *z1 = _mm_permute_ps(*z0, _MM_SHUFFLE(2,3,0,1));
    *z0 = _mm_addsub_ps(*z1, *z0);
    *z0 = _mm_permute_ps(*z0, _MM_SHUFFLE(3,3,0,0));

    return atan2f((*z0)[3], (*z0)[0]);
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

float approximateAtan2(float y, float x) {

    if (x == 0 && y == 0) {
        return 0.f;
    }

    float max, min;
    if (x > y) {
        max = x;
        min = y;
    } else {
        max = y;
        min = x;
    }
    float a = min/max;
    float s = a * a;
    float r = a * (1 + s * (-0.327623 + (0.159314 - 0.0464965 * s) *s));//((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;

    if (fabs(y) > fabs(x))  {
        return  M_PI_2 - r;
    }
    if (x < 0) {
        return M_PI - r;
    }
    if (y < 0) {
       return -r;
    }

    return r;
}

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

//                    arg[0] = asmArgz_ps(&u.vect, &v.vect);
                    arg[0] = asmArgz_ps(&u.vect, &v.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(0, &tstart, &tend);
                    // END TIMED

//                    arg[0] = atan2f(u.vect[3], u.vect[0]);
//                    approxAtan2 = approximateAtan2(u.vect[3], u.vect[0]);
                    zr[0] = u.vect[0];
                    zj[0] = u.vect[3];
                    PRINTF("(%.1f + %.1fi) . (%.1f + %.1fi)",  1., 2., 3., 4.);
                    PRINTF(" = (%.1f + %.1fi)\nphase: %f diff: %f\n", zr[0], zj[0], arg[0], sqrt(fabs(approxAtan2*arg[0])));
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

//                    arg[1] = atan2f(u1.vect[3], u1.vect[0]);
                    zr[1] = u1.vect[0];
                    zj[1] = u1.vect[3];
                    PRINTF("(%.1f + %.1fi) . (%.1f + %.1fi)",  1., 2., 3., 4.);
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
    printf("Iterations: %ld\n", n);
}