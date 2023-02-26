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
//#define LENGTH (1 << LOG2_LENGTH)

static const int LENGTH = 1 << LOG2_LENGTH;

union m256_8 {
    uint8_t buf[LENGTH] /*__attribute__((aligned(32))*/;
    __m256i v;
    int16_t buf16[LENGTH];
};

union m256_16 {
    int16_t buf[4] /*__attribute__((aligned(32))*/;
    __m256i v;
};

static const __m256i zero = {0,0,0,0};
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};

static __m256i *buf;
static __m256i *bufx4;
static uint8_t sampleMax = 0;
static uint32_t samplePowSum = 0;
static uint32_t overRun;
static uint8_t arr[(1 << 6) + 27];

int isCheckADCMax = 1;
int rdcBlockScalar = 9 + 1;
int dcAvgI;
int dcAvgQ;
int isRdc = 1;
int isOffsetTuning = 1;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm256_cvtepi8_epi16(lo_lane);
}

static void filterDcBlockRaw(const int len) {
    __m256i sumIq = {0,0,0,0};
    __m256i avgIq = {0,0,0,0};
    // len x 4 matrix => 4*(len / 2) = 2*len or since avg  = x / len * 2^-1 = x/len >> 1
    __m256i oneOverLen = _mm256_set1_epi16(16384/len); // test if fixed point full-range then shift vs this is faster
    int i, j;

    long localSumI = 0;
    long localSumQ = 0;
    int localAvgI = 0;
    int localAvgQ = 0;

    int avgI = 0;
    int avgQ = 0;
    long sumI = 0;
    long sumQ = 0;

    for (i = 0; i < len; ++i) {

        union m256_16 a = {.v = bufx4[i]};
        localSumI += a.buf[0] + a.buf[2];
        localSumQ += a.buf[1] + a.buf[3];
//        for (i = 0; i < 4; i += 2) {
//
//            localSumI += a.buf[i];// ((int)arr[j]) - 127;
//            localSumI += a.buf[i + 1];//((int)arr[j+1]) - 127;
//        }
        union m256_16 b = {.v =
                (sumIq = _mm256_add_epi16(sumIq, _mm256_add_epi16(bufx4[i],
                                       _mm256_shufflelo_epi16(bufx4[i],  _MM_SHUFFLE(1,0,3,2)))))};

        printf("%hX %hX vs %lX %lX\n", b.buf[0], b.buf[1], localSumI, localSumQ);
        assert(b.buf[0] == localSumI && b.buf[1] == localSumQ);
    }
    printf("\n");

    union m256_16 b = {.v = (avgIq = _mm256_mulhrs_epi16(sumIq, oneOverLen))};
    short lai = b.buf[0];
    short laq = b.buf[1];
    localAvgI += localSumI / (len << 1);
    localAvgQ += localSumQ / (len << 1);

    printf("%hX %hX vs %X %X\n\n", lai, laq, localAvgI, localAvgQ);
    int deltas[2] = {abs(localAvgI) - abs(lai), abs(localAvgQ) - abs(laq)};
    assert(abs(deltas[0]) <= 1 && abs(deltas[0]) <= 1);


    for (j = 0; j < len; j++) {
        union m256_16 a = {.v = bufx4[j]};
        for (i = 0; i < 4; i += 2) {

            sumI += a.buf[i];
            sumQ += a.buf[i + 1];
        }
    }

    avgI = sumI / (len << 1);
    avgQ = sumQ / (len << 1);

    printf("%hX %hX vs %X %X\n", lai, laq, avgI, avgQ);
    avgI = (avgI + dcAvgI * rdcBlockScalar) / (rdcBlockScalar + 1);
    avgQ = (avgQ + dcAvgQ * rdcBlockScalar) * (rdcBlockScalar + 1);

    for (i = 0; i < len; i += 2) {
        arr[i] -= avgI;
        arr[i+1] -= avgQ;
    }

    dcAvgI = (int) avgI;
    dcAvgQ = (int) avgQ;
}

static uint32_t convertTo16Bit(uint32_t len) {

    int i,j,k;
    bufx4 = calloc(len << LOG2_LENGTH, sizeof(__m256i));
    __m256i sm;

    if (isCheckADCMax) {

        for (i = 0; i < len; ++i) {

            sm = _mm256_max_epu8(sm, buf[i]);
        }
        sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
    }

    for (i = 0; i < len; i++) {

        buf[i] = convert(_mm256_sub_epi8(buf[i], Z));
    }

    for (j = 0, k = 0; j < len; ++j) { // TODO place this someplace more sensible- or is it sensible here?

        union m256_8 w = {.v = buf[j]};

        for (i = 0; i < LENGTH; i += 4) {

            union m256_16 z = {w.buf16[i], w.buf16[i+1], w.buf16[i+2], w.buf16[i+3]};
            bufx4[k++] = z.v;
        }
    }

    return k;
}
 // size = (len*LENGTH) << 2
static void breakit(const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    uint32_t unit = sizeof(uint8_t);
    long leftToProcess = len;
    uint32_t step = unit << LOG2_LENGTH;
    uint32_t chunk = step * unit;
    union m256_8 z;

    overRun = len - (size-1)*(step);
    buf = calloc(size * chunk, sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, arr + i, chunk);
        buf[j] = z.v;
        leftToProcess -= step;
    }
}

int main(void) {

    srand(time(NULL));

//    __m256i tmp;
    uint32_t len = sizeof(arr) / sizeof(*arr);
    uint32_t size = len / (sizeof(char) << LOG2_LENGTH) + 1;
//    uint32_t samplePowCount = 0;
    uint8_t val /*__attribute__((aligned(32))*/;
//    uint16_t *temp;
    int j, i;

    assert(size > 0);
//    arr16 = calloc(len * LENGTH, sizeof(short));

    for (i = 0; i < len; ++i) {
        if (i % LENGTH == 0) printf("\n");
        val = (100 + rand()) % 255;
        arr[i] = val;
//        arr16[i] = ((int)val - 127) & 0xFFFF;
        printf("%X, ", val);
        samplePowSum += val*val;
    }
    samplePowSum += samplePowSum / (LENGTH*len);

    printf("\n\n");

    breakit(len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf[i]};
        for (j = 0; j < LENGTH; ++j) {
            printf("%X, ", z.buf[j]);
        }
        printf("\n");
    }
    printf("\n");

    int k = convertTo16Bit(size);

    for (i = 0; i < size; ++i) {

        union m256_8 z = {.v = buf[i]};

        for (j = 0; j < LENGTH; ++j) {
            printf("%X, ", z.buf16[j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i=0; i<(int)len; i++) {
        if (i % LENGTH == 0) printf("\n");
        printf("%X, ",  (short)((short)arr[i]) - 127);
    }
    printf("\n\n");

    for (j = 0; j < k; ++j) {
        union m256_16 w = {.v = bufx4[j]};
        for (i = 0; i < 4; ++i)
            printf("%X, ", w.buf[i]);
        printf("\n");
    }
    printf("\n");

    filterDcBlockRaw(k);

//    free(arr16);
}