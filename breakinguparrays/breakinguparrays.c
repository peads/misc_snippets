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
#include <string.h>
#include "timed_functions.h"

union m256_8 {
    uint8_t buf[32] /*__attribute__((aligned(32))*/;
    __m256i v;
    int16_t buf16[16];

};

//union m256_16 {
//    __m256i v;
//    int16_t buf[16];
//};

static __m256i *buf8;
//static __m256i *buf16;

int isCheckADCMax = 1;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    __m128i hi_lane = _mm256_extracti128_si256(data, 1);
    return _mm256_castsi128_si256(_mm_packus_epi16(lo_lane, hi_lane));
}

static void convertTo16Bit(uint32_t len) {

//    const __m256i zero = _mm256_setzero_si256();
    const __m256i Z = _mm256_set1_epi16(0x7E7F);
    const __m256i z = _mm256_set1_epi8(127);

    static char sampleMax = 0;

    int i;
    int samplePowCount = 0;

    __m256i lower;
    __m256i upper;
    __m256i sm;
    __m256i sampleP = _mm256_setzero_si256();
    __m256i samplePowSum = _mm256_setzero_si256();

    if (isCheckADCMax) {

        sm = _mm256_max_epi8(buf8[0], buf8[1]);

        for (i = 2; i < len; ++i) {
            sm = _mm256_max_epi8(sm, buf8[i]);
        }

        sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));

        for (i = 0; i < len; i+=2) {

            lower = _mm256_sub_epi8(buf8[i], z);
            upper = _mm256_sub_epi8(buf8[i+1], z);

            sampleP = _mm256_add_epi16(_mm256_mullo_epi16(lower, upper), sampleP);
            samplePowSum = _mm256_avg_epu16(_mm256_add_epi16(samplePowSum, sampleP), sampleP);
            ++samplePowCount;
        }
    }

    for (i = 0; i < len; i++) {
        buf8[i] = _mm256_sub_epi16(buf8[i], Z);
    }
}

static void breakit(uint8_t *arr, const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    uint32_t unit = sizeof(uint8_t);
    long leftToProcess = len;
    uint32_t step = unit << 5;
    uint32_t chunk = step * unit;
    union m256_8 z;

    buf8 = calloc(size * chunk, sizeof(__m256i));
//    buf16 = calloc(size * chunk, sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, arr + i, chunk);
        buf8[j] = z.v;
        leftToProcess -= step;
    }
}

int main(void) {

    srand(time(NULL));

    __m256i tmp;
    uint8_t arr[(1 << 6) + 27] /*__attribute__((aligned(32))*/;
    uint32_t len = sizeof(arr) / sizeof(*arr);
    uint32_t size = len / (sizeof(char) << 5) + 1;
    uint8_t val /*__attribute__((aligned(32))*/;
    uint16_t *temp;
    int j, i;

    assert(size > 0);

    for (i = 0; i < len; ++i) {
        if (i % 32 == 0) printf("\n");
        val = (100 + rand()) % 10;
        arr[i] = val;
        printf("%d, ", val);
    }
    printf("\n\n");

    breakit(arr, len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf8[i]};
        for (j = 0; j < 32; ++j) {
            printf("%X, ", z.buf[j]);
        }
        printf("\n");
    }
    printf("\n");

    convertTo16Bit(size);

    for (i = 0; i < size; ++i) {

        union m256_8 z = {.v = buf8[i]};

        for (j = 0; j < 32; ++j) {
            printf("%X, ", z.buf[j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < size; ++i) {

        union m256_8 z = {.v = buf8[i]};

        for (j = 0; j < 32; ++j) {
            printf("%X, ", z.buf16[j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < size; ++i) {

        temp = calloc(32, sizeof(uint16_t));
        _mm256_storeu_epi16(temp, buf8[i]);

        for (j = 0; j < 32; ++j) {
            printf("%X, ", temp[j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i=0; i<(int)len; i++) {
        if (i % 32 == 0) printf("\n");
        printf("%X, ",  (arr[i] - 127) & 0xFF);
    }
    printf("\n");
}