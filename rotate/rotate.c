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

//#define DEBUG

#include <immintrin.h>
//#include <emmintrin.h>
#define TIMING_RUNS 5
#include "timed_functions.h"
#define MAX_ERROR 1e-6
#define MIN -20
#define MAX 20
#define STEP 1

#ifdef DEBUG
#include <stdio.h>
    #define PRINTF printf
#else
    #define PRINTF //
#endif

extern float ffabsf(float f);

/**
 * Takes two 128bit packed floats
 * and returns argument of the complex numbers they hold.
 * @param z0
 * @param z1
 * @return
 */
extern float argz(__m128 *z0, __m128 *z1);
__asm__(
#ifdef __APPLE_CC__
"_argz: "
#else
"argz: "
#endif
    "vmovaps (%rdi), %xmm0 \n\t"
    "vmulps (%rsi), %xmm0, %xmm0 \n\t"
    "vmovaps %xmm0, (%rdi) \n\t"
    "vpermilps $177, %xmm0, %xmm0 \n\t"
    "vmovaps %xmm0, (%rsi) \n\t"
    "vaddsubps (%rdi), %xmm0, %xmm1 \n\t"
    "vpermilps $240, %xmm1, %xmm0 \n\t"
    "vmovaps %xmm0, (%rdi) \n\t"
    /******* xmm -> x87 "faptain" section *******/
    "sub $16, %rsp \n\t"                    // emulate push, s.t. xmm can be accessed by the fpu
    "vextractps $0, %xmm0, (%rsp) \n\t"     // extractps seems to work better here,
    //  "vmovhps %xmm0, (%rsp) \n\t"        // which corresponds to the latency
    "flds (%rsp) \n\t"                      // and throughput tables from intel.
    "sub $16, %rsp \n\t"                    // albeit marginally, that is.
    "vextractps $3, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    "fpatan \n\t"
    "fstps (%rsp) \n\t"
    // B I G pop
    "vmovq (%rsp), %xmm0 \n\t"
    "add $32, %rsp \n\t"
    "ret"
);
/**
 * takes four float representing the complex numbers
 * (ar + iaj) * (br + ibj), * s.t. z = {ar, aj, br, bj}
 * and return their argument.
 **/
extern float argzB(float ar ,
                   float aj ,
                   float br ,
                   float bj );
__asm__(
#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif
    "vshufps $0x11, %xmm0, %xmm1, %xmm0\n\t"
    "vpermilps $0xDD, %xmm0, %xmm0\n\t"      // aj, aj, ar, ar -> xmm0

    "vshufps $0, %xmm3, %xmm2, %xmm1\n\t"
    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1

    "vmulps %xmm0, %xmm1, %xmm0\n\t"        // aj*bj, ar*br, aj*br, ar*bj
    "vpermilps $0xB1, %xmm0, %xmm3\n\t"
    "vaddsubps %xmm0, %xmm3, %xmm0\n\t"     // ar*br - aj*bj, ... , ar*bj + aj*br
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // (ar*br - aj*bj)^2, ... , (ar*bj + aj*br)^2
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"     // 0123 = 00011011 = 1B
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vsqrtps %xmm1, %xmm1\n\t"              // Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    // push
    "sub $16, %rsp \n\t"
    "vextractps $0, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    // push
    "sub $16, %rsp \n\t"
    "vextractps $3, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    "fpatan \n\t"
    "fstps (%rsp) \n\t"

    // B I G pop and return
    "vmovq (%rsp), %xmm0 \n\t"
    "add $32, %rsp \n\t"
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

    long temp = _mm_extract_ps(*z0, 3);
    const float y = *(float*)&temp;
    temp = _mm_extract_ps(*z0, 0);
    const float x = *(float*)&temp;

    return atan2f(x, y);
}

int main(void){
    static char *runNames[TIMING_RUNS]
    = {"argz :: ", "argz_ps :: ","conjz :: ", "conjz_ps :: ", "argzB :: "};

    float ar ;
    float aj ;
    float br ;
    float bj ;
    float true;

    float deltaArg, deltaR, deltaJ, deltaConjR, deltaConjJ;
    float arg[3], zr[2], zj[2], cnjR[2], cnjJ[2];
    struct timespec tstart, tend;
    uint64_t n = 0;
    int i;

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

                    arg[0] = argz(&u.vect, &v.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(0, &tstart, &tend);
                    // END TIMED

                    zr[0] = u.vect[0];
                    zj[0] = u.vect[3];

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    conjz(&u.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(2, &tstart, &tend);
                    // END TIMED

                    cnjR[0] = u.vect[0];
                    cnjJ[0] = u.vect[3];

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    arg[1] = argz_ps(&u1.vect, &v1.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(1, &tstart, &tend);
                    // END TIMED

                    zr[1] = u1.vect[0];
                    zj[1] = u1.vect[3];

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    conjz_ps(&u1.vect);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(3, &tstart, &tend);
                    // END TIMED

                    cnjR[1] = u1.vect[0];
                    cnjJ[1] = u1.vect[3];

                    // START_TIMED
                    clock_gettime(CLOCK_MONOTONIC, &tstart);

                    arg[2] = argzB(ar,aj,br,bj);

                    clock_gettime(CLOCK_MONOTONIC, &tend);
                    findDeltaTime(4, &tstart, &tend);
                    // END TIMED

                    true = atan2f(ar * br - aj * bj, ar * bj + aj * br);

                    deltaR = fabs(zr[0] - zr[1]);
                    deltaJ = fabs(zj[0] - zj[1]);
                    deltaConjR = fabs(cnjR[0] - cnjR[1]);
                    deltaConjJ = fabs(cnjJ[0] - cnjJ[1]);

                    for (i = 0; i < 3; ++i) {
                        deltaArg = fabs(ffabsf(true) - ffabsf(arg[i]));
                        if (!isnan(deltaArg)) {
                            assert(deltaArg < MAX_ERROR);
                        }
                    }

                    assert(deltaR < MAX_ERROR);
                    assert(deltaJ < MAX_ERROR);
                    assert(deltaConjR < MAX_ERROR);
                    assert(deltaConjJ < MAX_ERROR);
                }

    printTimedRuns(runNames, TIMING_RUNS);
    printf("Iterations: %ld\n", n);
}