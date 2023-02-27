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
    uint8_t buf[LENGTH];
    __m256i v;
    int16_t buf16[LENGTH];
};

union m256_16 {
    int16_t buf[4];
    __m256i v;
};

struct rotationMatrix {
    const union m256_16 a1;
    const union m256_16 a2;
};

static const __m256i zero = {0,0,0,0};
static const __m256i one = {1,1,1,1};
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};

static const struct rotationMatrix piOverTwo = {
        {0,-1,0,-1},
        {1,0,1,0}
};

static const struct rotationMatrix negPiOverTwo = {
        {0,1, 0,1},
            {-1,0, -1,0}
};

static __m256i *buf;
static __m256i *bufx4;
static __m256i dcAvgIq = {0,0,0,0};
static uint8_t sampleMax = 0;
static uint32_t samplePowSum = 0;
//static uint32_t overRun;
static uint8_t arr[(1 << 6) + 27];
static int rdcBlockScalar = 9 + 1;

int isCheckADCMax = 1;
int isRdc = 1;
int isOffsetTuning = 0;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm256_cvtepi8_epi16(lo_lane);
}
extern void swapNegateY(short *x, short *y);
__asm__ (
#ifdef __APPLE_CC__
"_swapNegateY: "
#else
"swapNegateY: "
#endif
    "mov (%rsi), %bx\n\t"
    "mov (%rdi), %cx\n\t"
    "mov %bx, (%rdi)\n\t"
    "neg %cx\n\t"
    "mov %cx, (%rsi)\n\t"
    "ret"
);

/**
 * Takes a 4x4 matrix and applied it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. {u1,u2,v1,v2}
 *  -> {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
static __m256i applyRotationMatrix(const struct rotationMatrix T, const __m256i u) {

    __m256i temp, temp1;

    temp = _mm256_mullo_epi16(T.a1.v, u);   // u1*a11, u2*a12, u3*a13, ...
    temp1 = _mm256_mullo_epi16(T.a2.v, u);  // u1*a21, u2*a22, ...
    return _mm256_blend_epi16(
            _mm256_add_epi16(temp,             // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
                _mm256_shufflelo_epi16(temp, _MM_SHUFFLE(2,3,0,1))),
            _mm256_add_epi16(temp1,            // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
                _mm256_shufflelo_epi16(temp1, _MM_SHUFFLE(2,3,0,1))),
            0xA);                                 // u1*a11 + u2*a12, u1*a21 + u2*a22,
                                                  // u3*a13 + u4*a14, u3*a23 + u4*a24
    // A = 0000 1010 = 00 22 => _MM_SHUFFLE(0,0,2,2)
}


static struct rotationMatrix generateRotationMatrix(float theta) {

    struct rotationMatrix result;
    short cosT = cos(theta) * (1 << 14);
    short sinT = sin(theta) * (1 << 14);
    short a1[2] = {cosT, -sinT};
    short a2[2] = {sinT, cosT};

    return result;
}

static void filterDcBlockRaw(const int len) {
    // TODO figure out to make these a compile-time consts, or at least,
    //  run-time ones, and in either case, without shuffles
    const __m256i rdcBlockRVector = _mm256_shufflelo_epi16(_mm256_blend_epi16(one,
        _mm256_set1_epi16(32768/(rdcBlockScalar+1)), 0xa), _MM_SHUFFLE(0,3,0,3));
    const __m256 rdcBlockVector = _mm256_set1_epi16(rdcBlockScalar);
    const __m256i rdcBlockVect1 = _mm256_shufflelo_epi16(_mm256_blend_epi16(one,
        _mm256_set1_epi16(rdcBlockScalar+1), 0xa), _MM_SHUFFLE(3,0,3,0));
    const __m256i oneOverLen = _mm256_set1_epi16(16384/len);

    __m256i sumIq = {0,0,0,0};
    __m256i avgIq;

    int i;
    for (i = 0; i < len; ++i) {
        sumIq = _mm256_add_epi16(sumIq, _mm256_add_epi16(bufx4[i],
                _mm256_shufflelo_epi16(bufx4[i], _MM_SHUFFLE(2,3,0,1))));
    }

    avgIq = _mm256_mulhrs_epi16(sumIq, oneOverLen);
    avgIq = _mm256_add_epi16(avgIq, _mm256_mullo_epi16(dcAvgIq, rdcBlockVector));
    avgIq = _mm256_mulhrs_epi16(avgIq, rdcBlockRVector);
    avgIq = _mm256_mullo_epi16(avgIq, rdcBlockVect1);

    for (i = 0; i < len; ++i) {
        bufx4[i] = _mm256_sub_epi16(bufx4[i], avgIq);
    }

    dcAvgIq = avgIq;
}

void rotateForOffsetTuning(int len) {
    int i, j;
    for(i = 0, j = 0; i < len; ++i, j+=4) {
        bufx4[i] = applyRotationMatrix(piOverTwo, bufx4[i]);
    }
    printf("\n\n");
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
            bufx4[k++]
                = (union m256_16){w.buf16[i], w.buf16[i+1], w.buf16[i+2], w.buf16[i+3]}.v;
        }
    }

    if (isRdc) {
        filterDcBlockRaw(k);

        for(i = 0, j = 0; i < k; ++i, j+=4) {
            if (j % LENGTH == 0) printf("\n");
            union m256_16 w;
            bufx4[i] = applyRotationMatrix(piOverTwo, bufx4[i]);
            w.v = bufx4[i];
            printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
        }
    }

    if (!isOffsetTuning) {
        rotateForOffsetTuning(k);
    }

    return k;
}

static void breakit(const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    uint32_t unit = sizeof(uint8_t);
    long leftToProcess = len;
    uint32_t step = unit << LOG2_LENGTH;
    uint32_t chunk = step * unit;
    union m256_8 z;

//    overRun = len - (size-1)*(step);
    buf = calloc(size * chunk, sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, arr + i, chunk);
        buf[j] = z.v;
        leftToProcess -= step;
    }
}

void dc_block_raw_filter(int m, int n)
{
    /* derived from dc_block_audio_filter,
        running over the raw I/Q components
    */
    short i, j,k, avgI, avgQ;
    short bf[m*n];
    int dc_avgI = 0;
    int dc_avgQ = 0;
    int64_t sumI = 0;
    int64_t sumQ = 0;

    for (k = 0, i = 0; i < m; ++i){
        union m256_16 tmp = {.v = bufx4[i]};
        for (j = 0; j < n; ++j, ++k) {
            bf[k] = tmp.buf[j];
        }
    }

    for (i = 0; i < m*n; i += 2) {
        sumI += bf[i];
        sumQ += bf[i + 1];
    }

    avgI = sumI / ( m*n / 2 );
    avgQ = sumQ / ( m*n / 2 );
    printf("%hX, %hX, %hX, %hX\n", avgI, avgQ,avgI, avgQ);
    avgI = (avgI + dc_avgI * rdcBlockScalar) / ( rdcBlockScalar + 1 );
    avgQ = (avgQ + dc_avgQ * rdcBlockScalar) / ( rdcBlockScalar+1 );

    for (i = 0; i < m*n; i += 2) {
        bf[i] -= avgI;
        bf[i + 1] -= avgQ;
    }

    dc_avgI = avgI;
    dc_avgQ = avgQ;
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
        printf("%hX, ", val);
        samplePowSum += val*val;
    }
    printf("\n\n");

    samplePowSum += samplePowSum / (LENGTH*len);


    breakit(len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf[i]};
        for (j = 0; j < LENGTH; ++j) {
            printf("%hX, ", z.buf[j]);
        }
        printf("\n");
    }
    printf("\n");

    int depth = convertTo16Bit(size);

//    for (i = 0; i < size; ++i) {
//
//        union m256_8 z = {.v = buf[i]};
//
//        for (j = 0; j < LENGTH; ++j) {
//            printf("%hX, ", z.buf16[j]);
//        }
//        printf("\n");
//    }
//    printf("\n");

    for (j = 0, i = 0; j < depth; ++j, i+=4) {
        if (i % LENGTH == 0) printf("\n");
        union m256_16 w = {.v = bufx4[j]};

        printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
    }
    printf("\n\n");
}