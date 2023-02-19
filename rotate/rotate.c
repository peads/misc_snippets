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
#define MAX_ERROR 1e-6
extern float asmArgz_ps(__m128 *z0, __m128 *z1);
__asm__(
#ifdef __APPLE_CC__
"_asmArgz_ps: "
#else
"asmArgz_ps: "
#endif
    "vmovaps (%rsi), %xmm1\n\t"
    "vmulps (%rdi), %xmm1, %xmm2\n\t"
    "vpermilps $0xB1, %xmm2, %xmm1\n\t"
    "vaddsubps %xmm2, %xmm1, %xmm1\n\t"
    "vpermilps $0xF0, %xmm1, %xmm1\n\t"
    "vmovaps %xmm1, (%rdi)\n\t"
    "movhlps %xmm1, %xmm2\n\t"
    // push
    "sub $16, %rsp\n\t"
    "vmovss %xmm2, (%rsp)\n\t"
    "flds (%rsp)\n\t"
    // push
    "sub $16, %rsp\n\t"
    "vmovss %xmm1, (%rsp)\n\t"
    "flds (%rsp)\n\t"
    //pop
    "add $16, %rsp\n\t"
    "fpatan\n\t"
    "fstps (%rsp)\n\t"
    "vmovq(%rsp), %xmm0\n\t"
    // pop
    "add $16, %rsp\n\t"
    "ret"
);

extern void conjz(__m128 *z);
__asm__ (
//".section:\n.data:\naarr word $1.0, $1.0, $-1.0, $-1.0;\n\t"
#ifdef __APPLE_CC__
"_conjz: "
#else
"conjz: "
#endif
    "push $0x3f800000\n\t"
    "vbroadcastss (%rsp), %xmm1\n\t" // all ones
    "pop %rbx\n\t"
    "xorps %xmm2, %xmm2\n\t" // load zeroes
    "subps %xmm1, %xmm2\n\t" // -ones
    "insertps $0xb4, %xmm2, %xmm1\n\t" // 0x44
    "mulps (%rdi), %xmm1\n\t"
//    "vpermilps $0x1B, %xmm1, %xmm1\n"
    "vmovaps %xmm1, (%rdi)\n\t"
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

int main(void){
    union vect z1 = {4,3,3,4};
    float ar, aj, br, bj, deltaArg, deltaR, deltaJ, deltaConjR, deltaConjJ;
    float arg[2], zr[2], zj[2], cnjR[2], cnjJ[2];

    for (ar = -9; ar < 9; ++ar)
        for (aj = -9; aj < 9; ++aj)
            for (br = -9; br < 9; ++br)
                for (bj = -9; bj < 9; ++bj){
                    union vect u = { aj,ar,aj,ar };
                    union vect v = { bj,br,br,bj };

                    union vect u1 = { aj,ar,aj,ar };
                    union vect v1 = { bj,br,br,bj };

                    arg[0] = asmArgz_ps(&u.vect, &v.vect);
                    zr[0] = u.vect[0];
                    zj[0] = u.vect[3];
                    printf("(%.1f + %.1fi) . (%.1f + %.1fi)",  1., 2., 3., 4.);
                    printf(" = (%.1f + %.1fi)\nphase: %f\n", zr[0], zj[0], arg[0]);

                    conjz(&u.vect);
                    cnjR[0] = u.vect[0];
                    cnjJ[0] = u.vect[3];
                    printf("conjugate: (%.1f + %.1fi)\n",  u.vect[0], u.vect[3]);

                    arg[1] = argz_ps(&u1.vect, &v1.vect);
                    zr[1] = u1.vect[0];
                    zj[1] = u1.vect[3];
                    printf("(%.1f + %.1fi) . (%.1f + %.1fi)",  1., 2., 3., 4.);
                    printf(" = (%.1f + %.1fi)\nphase: %f\n", zr[1], zj[1], arg[1]);

                    conjz_ps(&u1.vect);
                    cnjR[1] = u1.vect[0];
                    cnjJ[1] = u1.vect[3];
                    printf("conjugate: (%.1f + %.1fi)\n\n",  u1.vect[0], u1.vect[3]);

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
}