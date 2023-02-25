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

union v {
    uint8_t buf[32] /*__attribute__((aligned(32))*/;
    __m256i v;
//    uint16_t *buf16;

};


int isCheckADCMax = 1;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m128i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    __m128i hi_lane = _mm256_extracti128_si256(data, 1);
    return _mm_packus_epi16(lo_lane, hi_lane);
}
static __m256i *buf8;

static void convertTo16Bit(uint32_t len) {

    int i;
    static char sampleMax = 0;
    int samplePowCount = 0;

    __m256i lower;
    __m256i higher;
    __m256i sm;
    __m256i sampleP = _mm256_setzero_si256();
    __m256i samplePowSum = _mm256_setzero_si256();
    __m256i Z = _mm256_set1_epi16(127);

    if (isCheckADCMax) {

        sm = _mm256_max_epi8(buf8[0], buf8[1]);

        for (i = 2; i < len; ++i) {
            sm = _mm256_max_epi8(sm, buf8[i]);
        }

        sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
    }

    for (i = 0; i < len - 1; ++i) {

        if (isCheckADCMax) {
            sampleP = _mm256_add_epi16(_mm256_mullo_epi16(lower, higher), sampleP);
            samplePowSum = _mm256_avg_epu16(_mm256_add_epi16(samplePowSum, sampleP), sampleP);
            ++samplePowCount;
        }

        buf8[i] = _mm256_sub_epi16(buf8[i], Z);
    }
}

static void breakit(  uint8_t *arr, const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    uint32_t unit = sizeof(uint8_t);
    long leftToProcess = len;
    uint32_t step = unit << 5;
    uint32_t chunk = step * unit;
//    uint32_t remainder = len - (len / step << 5);
//    uint8_t temp[32];
    union v z;

    buf8 = calloc(size * chunk, sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, arr + i, chunk);
        buf8[j] = z.v;
        leftToProcess -= step;
    }
//    if (leftToProcess) {
//        memset(temp+remainder, 0, step-remainder);
//        z = (union v){ temp };
//        buf8[j-1] = z.v;
//    }
}

int main(void) {

    srand(time(NULL));

    uint8_t arr[(1 << 6) + 27] /*__attribute__((aligned(32))*/;
    uint32_t len = sizeof(arr) / sizeof(*arr);
    uint32_t size = len / (sizeof(char) << 4) + 1;
    uint8_t val /*__attribute__((aligned(32))*/;
    int j, i;

    assert(size > 0);

    for (i = 0; i < len; ++i) {
        if (i % 16 == 0) printf("\n");
        val = (100 + rand()) % 10;
        arr[i] = val;
        printf("%d, ", val);
    }
    printf("\n\n");

    breakit(arr, len, size);

    for (i = 0; i < size; ++i) {
        union v z = {.v = buf8[i]};
        for (j = 0; j < 32; ++j) {
            printf("%X, ", z.buf[j]);
        }
        printf("\n");
    }

    convertTo16Bit(size);

    for (i = 0; i < size; ++i) {

        uint16_t *temp = calloc(32, sizeof(uint16_t));
        _mm256_storeu_epi16(temp, buf8[i]);

        for (j = 0; j < 16; ++j) {

            printf("%X, ", temp[j]);
        }
        printf("\n");
    }
}