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

//static const union combine X
//        = {.arr = {1, 1, 1, 1, -1, -1, 1, -1}};
//static const union combine Y
//        = {.arr = {1, 1, -1, 1, 1, 1, 1, 1}};

//static __m128i zero;

//static void printShorts(short *__restrict__ dst) {
//
//    //printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
////           dst[0],
////           dst[1],
////           dst[2],
////           dst[3],
////           dst[4],
////           dst[5],
////           dst[6],
////           dst[7],
////           dst[8],
////           dst[9],
////           dst[10],
////           dst[11],
////           dst[12],
////           dst[13],
////           dst[14],
////           dst[15]);
//}

//static void printAndStoreEpi16(short *__restrict__ dst, __m128i *__restrict__ vect, int len) {
//
//    __m128i lower;
//    __m128i higher;
//    int i;
//
//    for (i = 0; i < len; ++i) {
//
//        lower = _mm_unpacklo_epi8(vect[i], zero);
//        higher = _mm_unpackhi_epi8(vect[i], zero);
//
//        _mm_storeu_si128((__m128i_u *) (dst + 8), higher);
//        _mm_storeu_si128((__m128i_u *) dst, lower);
//
//        printShorts(dst);
//    }
//}
//
//// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
//double sumV(__m128i v) {
//
//    __m128i vsum = _mm_sad_epu8(v, _mm_setzero_si128());
//    return _mm_cvtsi128_si32(vsum) + _mm_extract_epi16(vsum, 4);
//}
//
//double sumVs(__m128i u, __m128i v) {
//
//    __m128i vsum = _mm_add_epi64(u, v);
//    return _mm_cvtsi128_si32(vsum) + _mm_extract_epi16(vsum, 4);
//}
//
//void fun1(unsigned char *a, int len) {
//
//    __m128i Z = _mm_set1_epi16(127);
//    int i, k = 0;
//    int max = len;
//    int maxOver16 = max >> 4;
//    __m128i arr[max];
//    __m128i lower;
//    __m128i higher;
//    unsigned char *firstHalf __attribute__((aligned(16)));
//    unsigned char *secondHalf __attribute__((aligned(16)));
//    short temp[16] __attribute__((aligned(16)));
////    __m256i result;
//    double d = 0.0;
//
//    for (i = 0; i < max; i += 16) {
//        firstHalf = a + i;
//        secondHalf = a + i + 8;
//
//        arr[k] = _mm_unpacklo_epi64(
//                _mm_loadl_epi64((const __m128i *) firstHalf),
//                _mm_loadl_epi64((const __m128i *) secondHalf));
//        d += sumV(arr[k]);
//        k++;
//    }
//    //printf("%f\n", (d / k));
//
//
//    printAndStoreEpi16(temp, arr, maxOver16);
//    for (i = 0; i < maxOver16; ++i) {
//        lower = _mm_unpacklo_epi8(arr[i], zero);
//        higher = _mm_unpackhi_epi8(arr[i], zero);
////        result = _mm256_set_m128i(_mm_sub_epi16(higher, Z), _mm_sub_epi16(lower, Z));
//        _mm_storeu_si128((__m128i_u *) (temp + 8), _mm_sub_epi16(higher, Z));
//        _mm_storeu_si128((__m128i_u *) temp, _mm_sub_epi16(lower, Z));
//        printShorts(temp);
//    }
//}

//void fun3(short *lp, int len){
//    int i, k = 0, max = len >> 2;
//    __m128i U, V;
//    short result[len];
//    double pcm;
//    short ar  __attribute__((aligned(16)));
//    short aj  __attribute__((aligned(16)));
//    short br  __attribute__((aligned(16)));
//    short bj  __attribute__((aligned(16)));
//
//    ar = 0;
//    aj = 0;
//    br = lp[0];
//    bj = lp[1];
//
//    U = _mm_set_epi16(0,0,0,0,ar, aj, ar, br);
//    V = _mm_set_epi16(0,0,0,0,br, bj, bj, aj);
//
//    U = _mm_mulhi_epi16(U, V); //ar*br, aj*bj, ar*bj, br*aj
//    V = _mm_srli_epi16(U , 1);
//    U = _mm_addsub_ps(U, V);
//    // ar*ar - aj*bj, aj*bj + ar*br, ar*bj - br*aj, br*aj + ar*bj
//
//    pcm = atan2(_mm_extract_epi16(U, 0), _mm_extract_epi16(U, 3));
//    result[0] = pcm;
//
//    //printf("(%d + %di) . (%d + %di) = (%d + %di): phase %f\n", ar, aj, br, bj, _mm_extract_epi16(U, 0), _mm_extract_epi16(U, 3), pcm);
//
//    //TODO fm->result = ???
////    result[0] = pcm[0];
//
//    for (i = 2; i < max - 1; i=i+2, ++k){
//        ar = lp[i];
//        aj = lp[i + 1];
//        br = lp[i - 2];
//        bj = lp[i - 1];
//
//        U = _mm_set_epi16(0,0,0,0,ar, aj, ar, br);
//        V = _mm_set_epi16(0,0,0,0,br, bj, bj, aj);
//
//        U = _mm_mulhi_epi16(U, V); //ar*br, aj*bj, ar*bj, br*aj
//        V = _mm_slli_epi8(U , 1);
//        //printf("%d %d %d %d\n",_mm_extract_epi16(U,0),  _mm_extract_epi16(U, 1),  _mm_extract_epi16(U, 2),  _mm_extract_epi16(U, 3));
//        //printf("%d %d %d %d\n",_mm_extract_epi16(V,0),  _mm_extract_epi16(V, 1),  _mm_extract_epi16(V, 2),  _mm_extract_epi16(V, 3));
//        U = _mm_addsub_ps(U, V);
//        pcm = atan2(_mm_extract_epi16(U, 0), _mm_extract_epi16(U, 3));
//        result[i >> 1] = pcm;
//
//        //printf("(%d + %di) . (%d + %di) = (%d + %d)i: phase %f\n", ar, aj, br, bj ,_mm_extract_epi16(U, 0), _mm_extract_epi16(U, 3), pcm);
//        //TODO set fm->result;
//    }
//}
//double fun2(short *a, int len) {
//
//    double dc = 1.0 / len;
//    double err;
//    double tstep;
//    int i, k = 0;
//    int num;
//    int max = len >> 2;
//    int maxOver2 = max >> 1;
//    __m128i arr[max];
//    short *firstHalf __attribute__((aligned(16)));
//    short *secondHalf __attribute__((aligned(16)));
//    uint8_t temp[16] __attribute__((aligned(16)));
//    __m128i p = _mm_setzero_si128();
//    __m128i t = _mm_setzero_si128();
//
//    for (i = 0; i < max; i += 8, k++) {
//        firstHalf = a + i;
//        secondHalf = a + i + 4;
//
//        arr[k] = _mm_unpacklo_epi64(
//                _mm_loadl_epi64((const __m128i *) firstHalf),
//                _mm_loadl_epi64((const __m128i *) secondHalf));
//
//        t = _mm_add_epi16(t, arr[k]);
//        p = _mm_add_epi16(p, _mm_mullo_epi16(arr[k], arr[k]));
//    }
//    int sum = sumV(t);
//    int powSum = sumV(p);
//
//    dc *= sum;
//    err = dc * ((sum >> 1) - 1);
//
//    return sqrt((powSum - err) / len);
//}
//
//void fun(short *a, int len) {
//
//    short *firstHalf __attribute__((aligned(16)));
//    short *secondHalf __attribute__((aligned(16)));
//    short temp[8] __attribute__((aligned(16)));
//
//    int i, j, k = 0;
//    int max = len >> 2;
//    int maxOverTwo = max >> 1;
//    __m128i arr[maxOverTwo];
//
//    for (i = 0; i < max; i += 2) {
//        j = i << 2;
//        int m = j + 4;
//        firstHalf = a + j;
//        secondHalf = a + m;
//
//        arr[k++] = _mm_unpacklo_epi64(
//                _mm_loadl_epi64((const __m128i *) firstHalf),
//                _mm_loadl_epi64((const __m128i *) secondHalf));
//    }
//
//    for (i = 0; i < maxOverTwo; ++i) {
//        arr[i] = _mm_mullo_epi16(arr[i], X.vect);
//        arr[i] = _mm_mullo_epi16(arr[i], Y.vect);
//        _mm_store_si128((__m128i_u *) temp, arr[i]);
//        //printf("%d %d %d %d %d %d %d %d\n",
////               temp[0],
////               temp[1],
////               temp[2],
////               temp[3],
////               temp[4],
////               temp[5],
////               temp[6],
////               temp[7]);
//    }
//}

//int main(void) {
//
////    zero = _mm_setzero_si128();
////    short tmp, xx, yy;
////    short ints[4] = {21, 7, 1, 2};
////    short a[64] =
////            {1, 2, 3, 4, 5, 6, 7,
////             8, 9, 10, 11, 12, 13, 14, 15, 16,
////             17, 18, 19, 20, 21, 22, 23, 24,
////             25, 26, 27, 28, 29, 30, 31, 32,
////             1, 2, 3, 4, 5, 6, 7,
////             8, 9, 10, 11, 12, 13, 14, 15, 16,
////             17, 18, 19, 20, 21, 22, 23, 24,
////             25, 26, 27, 28, 29, 30, 31, 32};
////    unsigned char b[64] =
////            {1, 2, 3, 4, 5, 6, 7,
////             8, 9, 10, 11, 12, 13, 14, 15, 16,
////             17, 18, 19, 20, 21, 22, 23, 24,
////             25, 26, 27, 28, 29, 30, 31, 32,
////             1, 2, 3, 4, 5, 6, 7,
////             8, 9, 10, 11, 12, 13, 14, 15, 16,
////             17, 18, 19, 20, 21, 22, 23, 24,
////             25, 26, 27, 28, 29, 30, 31, 32};
////
////    xx = ints[1];
////    yy = -ints[0];
////    MUL_MINUS_J_INT(ints, 0);
////    assert(ints[0] == xx && ints[1] == yy);
////
////    ints[0] = 21;
////    ints[1] = 7;
////    swapNegate(&ints[0], &ints[1]);
////    assert(ints[0] == xx && ints[1] == yy);
////
////    ints[0] = 21;
////    ints[1] = 7;
////    xx = -ints[1];
////    yy = ints[0];
////    MUL_PLUS_J_INT(ints, 0);
////    assert(ints[0] == xx && ints[1] == yy);
////
////    ints[0] = 21;
////    ints[1] = 7;
////    swapNegate(&ints[1], &ints[0]);
////    assert(ints[0] == xx && ints[1] == yy);
////    fun(a, sizeof(a) / sizeof(*a));
////    fun1(b, sizeof(b) / sizeof(*b));
////    //printf("%f\n", fun2(a, sizeof(a) / sizeof(*a)));
////
////    int i, s, t= 0, p=0;
////    for (i = 0; i < sizeof(a) / sizeof(*a); i++) {
////        s = a[i];
////        t += s;
////        p += s * s;
////    }
//// // always 1
////    double dc = t / (sizeof(a) / sizeof(*a));
////    double err = dc * ((t << 1) - 1);
////    double ratio = (p - err) / (sizeof(a) / sizeof(*a));
////    double result = sqrt(ratio);
////    //printf("%f\n", result);
////    union vect {
////        __m128 vect;
////    };

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

union vect {
    float arr[sizeof(__m128) / sizeof(float)] __attribute__((aligned(16)));
    __m128 vect;
};

__m128 rotateByNPi(__m128 *z, int n) {
    
}

float argz_ps(__m128 *z0, __m128 *z1){// z0 = (ar, aj), z1 = (br, bj) => z0*z1 = (ar*br - aj*bj, ar*bj + br*aj)

    *z0 = _mm_mul_ps(*z0, *z1);                // => ar*br, aj*bj, ar*bj, br*aj
//    printf("%X\n", _MM_SHUFFLE(2,3,0,1));
    *z1 = _mm_permute_ps(*z0, _MM_SHUFFLE(2,3,0,1));
    *z0 = _mm_addsub_ps(*z1, *z0);
    printf(" = (%.1f + %.1fi)\n", (*z0)[0], (*z0)[3]);
    return atan2f((*z0)[3], (*z0)[0]);
}

float argz(float ar, float aj, float br, float bj) {
    union vect z0 = {.arr = {aj,ar,aj,ar}}; // aj, ar, aj, ar
    union vect z1 = {.arr = {bj,br,br,bj}}; // bj, br, br, bj

    return argz_ps(&z0.vect, &z1.vect);
}

int main(void){
    printf("(%.1f + %.1fi) . (%.1f + %.1fi)",  1., 2., 3., 4.);
    float z = argz(1,2,3,4);
    printf("phase: %f\n", z);


}