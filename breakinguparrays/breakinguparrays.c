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

static uint8_t sampleMax = 0;
static uint8_t arr[(1 << 6) + 27];
static uint32_t samplePowSum = 0;
static uint32_t rdcBlockScalar = 9 + 1;
static uint32_t isCheckADCMax;
static uint32_t isRdc;
static uint32_t isOffsetTuning;
static uint32_t downsample;
static __m256i *buf;
static __m256i *bufx4;
static __m256i *lowPassed;
static __m256i dcAvgIq = {0,0,0,0};
static __m256i rdcBlockVector;
static __m256i rdcBlockRVector;
static __m256i rdcBlockVect1;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i convert(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm256_cvtepi8_epi16(lo_lane);
}

/**
 * Takes a 4x4 matrix and applied it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. {u1,u2,v1,v2}
 *  -> {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
static __m256i applyRotationMatrix(const struct rotationMatrix T, const __m256i u) {
    // TODO integrate abliity to use fixed point encoded values from the
    // generate function (scaling factor: 2^13)
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


static struct rotationMatrix generateRotationMatrix(float theta, float phi) {

    int16_t cosT = cos(theta) * (1 << 13);
    int16_t sinT = sin(phi) * (1 << 13);
    struct rotationMatrix result = {
            .a1 = {cosT, -sinT, cosT, -sinT},
            .a2 = {sinT, cosT, sinT, cosT}
    };

    return result;
}

static void filterDcBlockRaw(const int len) {

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
}

static int splitToQuads(uint32_t len) {

    int i, j, k;

    for (j = 0, k = 0; j < len; ++j) {

        union m256_8 w = {.v = buf[j]};

        for (i = 0; i < LENGTH; i += 4) {
            bufx4[k++]
                    = (union m256_16){w.buf16[i], w.buf16[i+1], w.buf16[i+2], w.buf16[i+3]}.v;
        }
    }

    return k;
}

static void findMaxSample(uint32_t len) {

    int i;
    __m256i sm;

    for (i = 0; i < len; ++i) {
        sm = _mm256_max_epu8(sm, buf[i]);
    }
    sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
}

static uint32_t convertToNx4_16BitMatrix(uint32_t len) {

    int i,depth;
    bufx4 = calloc(len << LOG2_LENGTH, sizeof(__m256i));

    if (isCheckADCMax) {

        findMaxSample(len);
    }

    for (i = 0; i < len; i++) {
        buf[i] = convert(_mm256_sub_epi8(buf[i], Z));
    }

    depth = splitToQuads(len);

    if (isRdc) {
        filterDcBlockRaw(depth);
    }

    if (!isOffsetTuning) {
        rotateForOffsetTuning(depth);
    }

    return depth;
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
//    memset(buf, 0, size * chunk * sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, arr + i, chunk);
        buf[j] = z.v;
        leftToProcess -= step;
    }
}

int16_t *lowpassed;
int16_t *lowpassed2;

void low_pass(int len)
/* simple square window FIR */
{
    union m256_16 temp;
    int16_t now_r, now_j, prev_index;
    int i, k, j;

    now_j = now_r = prev_index = 0;

    for (k = 0, j = 0; j < len; ++j) {

        prev_index = 0;
        now_r = 0;
        now_j = 0;
        i = 0;
        temp.v = bufx4[j];

        lowpassed2[k] = temp.buf[0] + temp.buf[2];
        lowpassed2[k+1] = temp.buf[1] + temp.buf[3];

        while (i < 4) {
            now_r += temp.buf[i];
            now_j += temp.buf[i + 1];
            i += 2;
            prev_index++;
            if (prev_index < downsample) {
                continue;
            }
            lowpassed[k] = now_r; /* * d->output_scale; */
            lowpassed[k + 1] = now_j; /* * d->output_scale; */

            k += 2;
        }

    }
}

void applyLowPass(int len) {
    int i;

    lowPassed = calloc(len << 1, sizeof(__m256i));

    for (i = 0; i < len; ++i) {
        lowPassed[i]
            = _mm256_add_epi16(bufx4[i],
               _mm256_shufflelo_epi16(bufx4[i], _MM_SHUFFLE(1,0,3,2)));
    }
}

void initializeEnv() {

    const int16_t scalarP1 = rdcBlockScalar + 1;
    rdcBlockVector = _mm256_set1_epi16(rdcBlockScalar);
    rdcBlockVect1 = (union m256_16){scalarP1, 0x1, scalarP1, 0x1}.v;
    rdcBlockRVector = (union m256_16){32768/scalarP1, 0x1, 32768/scalarP1, 0x1}.v;
}

int main(int argc, char **argv) {

    srand(time(NULL));

    uint32_t len = sizeof(arr) / sizeof(*arr);
    uint32_t size = len / (sizeof(char) << LOG2_LENGTH) + 1;
    uint8_t val;
    int j, i;

    if (argc <= 1) {
        isCheckADCMax = 0;
        isRdc = 0;
        isOffsetTuning = 0;
        downsample = 2;
    } else {
        for (i = 0; i < argc; ++i) {
            switch (argv[i][0]){
                case 'r':
                    isRdc = 1;
                    break;
                case 'a':
                    isCheckADCMax = 1;
                    break;
                case 'o':
                    isOffsetTuning = 1;
                default:
                    break;
            }
        }
    }

    initializeEnv();

    for (i = 0; i < len; ++i) {
        if (i % LENGTH == 0) printf("\n");
        val = (100 + rand()) % 255;
        arr[i] = val;
        printf("%hhd, ", val);
        samplePowSum += val*val; // TODO implement this with data parallelism
    }
    printf("\n\n");

    samplePowSum += samplePowSum / (LENGTH*len);

    breakit(len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf[i]};
        for (j = 0; j < LENGTH; ++j) {
            printf("%hhd, ", z.buf[j]);
        }
        printf("\n");
    }

    int depth = convertToNx4_16BitMatrix(size);

    for (j = 0, i = 0; j < depth; ++j, i+=4) {
        if (i % LENGTH == 0) printf("\n");
        union m256_16 w = {.v = bufx4[j]};

        printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
    }
    printf("\n");

    lowpassed = calloc(depth << 1, sizeof(int16_t));
    lowpassed2 = calloc(depth << 1, sizeof(int16_t));
    low_pass(depth);
    
    for (i = 0; i < depth << 1; i+=4) {
        if (i % (LENGTH) == 0) printf("\n");
        printf("%hd, %hd, %hd, %hd, ",  lowpassed[i],  lowpassed[i+1], lowpassed[i+2] , lowpassed[i+3]);
    }
    printf("\n");

    for (i = 0; i < depth << 1; i+=4) {
        if (i % (LENGTH) == 0) printf("\n");
        printf("%hd, %hd, %hd, %hd, ",  lowpassed2[i],  lowpassed2[i+1], lowpassed2[i+2] , lowpassed2[i+3]);
    }
    printf("\n");

    applyLowPass(depth);
    for (j = 0, i = 0; j < depth; ++j, i+=4) {
        if (j % (LENGTH>>1) == 0) printf("\n");
        union m256_16 w = {.v = lowPassed[j]};

        printf("%hd, %hd, ", w.buf[0],w.buf[1]);
    }
    printf("\n");
    free(bufx4);
    free(buf);
}