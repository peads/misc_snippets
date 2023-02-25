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

#define LOG2_LENGTH 4
#define LENGTH (1 << LOG2_LENGTH)

union m256_8 {
    uint8_t buf[LENGTH] /*__attribute__((aligned(32))*/;
    __m256i v;
    int16_t buf16[LENGTH];

};

int isCheckADCMax = 1;

//static const __m256i zero = {0,0,0,0};
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};

static __m256i *buf8;


static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm256_cvtepi8_epi16(lo_lane);
}

static void convertTo16Bit(uint32_t len) {

    static char sampleMax = 0;
    static int samplePowCount = 0;

    int i;

    __m256i sm;
    __m256i sampleP = _mm256_setzero_si256();
    __m256i samplePowSum = _mm256_setzero_si256();

    if (isCheckADCMax) {

        sm = _mm256_max_epi8(buf8[0], buf8[1]);

        for (i = 2; i < len; ++i) {

            sm = _mm256_max_epi8(sm, buf8[i]);
        }

        sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
    }

    for (i = 0; i < len; i++) {
        buf8[i] = convert(_mm256_sub_epi8(buf8[i], Z));
    }

    if (isCheckADCMax) {
        for (i = 0; i < len; i+=2) {

            sampleP = _mm256_add_epi16(_mm256_mullo_epi16(buf8[i], buf8[i+1]), sampleP);
            samplePowSum = _mm256_avg_epu16(_mm256_add_epi16(samplePowSum, sampleP), sampleP);
            ++samplePowCount;
        }
    }
}

static void breakit(uint8_t *arr, const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    uint32_t unit = sizeof(uint8_t);
    long leftToProcess = len;
    uint32_t step = unit << LOG2_LENGTH;
    uint32_t chunk = step * unit;
    union m256_8 z;

    buf8 = calloc(size * chunk, sizeof(__m256i));

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
    uint32_t size = len / (sizeof(char) << LOG2_LENGTH) + 1;
    uint8_t val /*__attribute__((aligned(32))*/;
    uint16_t *temp;
    int j, i;

    assert(size > 0);

    for (i = 0; i < len; ++i) {
        if (i % LENGTH == 0) printf("\n");
        val = (100 + rand()) % 255;
        arr[i] = val;
        printf("%X, ", val);
    }
    printf("\n\n");

    breakit(arr, len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf8[i]};
        for (j = 0; j < LENGTH; ++j) {
            printf("%X, ", z.buf[j]);
        }
        printf("\n");
    }
    printf("\n");

    convertTo16Bit(size);

//    for (i = 0; i < size; ++i) {
//
//        union m256_8 z = {.v = buf8[i]};
//
//        for (j = 0; j < LENGTH; ++j) {
//            printf("%X, ", z.buf[j]);
//        }
//        printf("\n");
//    }
//    printf("\n");

    for (i = 0; i < size; ++i) {

        union m256_8 z = {.v = buf8[i]};

        for (j = 0; j < LENGTH; ++j) {
            printf("%X, ", z.buf16[j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i=0; i<(int)len; i++) {
        if (i % LENGTH == 0) printf("\n");
        printf("%X, ",  ((short)arr[i]) - 127);
    }
    printf("\n");
}