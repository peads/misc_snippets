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
#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#define DEBUG
#ifdef DEBUG
#include <assert.h>
#endif
// sizeof(uint8_t)
#define INPUT_ELEMENT_BYTES 1
// sizeof(float)
#define OUTPUT_ELEMENT_BYTES 5
// sizeof(__m128)
#define MATRIX_ELEMENT_BYTES 16
#define VECTOR_WIDTH 4
#define LOG2_VECTOR_WIDTH 2
#define MAXIMUM_BUF_SIZE 1L << 33

union m128_f {
    float buf[4];
    __m128 v;
};

struct rotationMatrix {
    const union m128_f a1;
    const union m128_f a2;
};

static const __m128 NEGATE_B_IM = {1.f,1.f,1.f,-1.f};
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};
static const __m128i FLOAT_ABS // all 0x7FFFFFFFUs
    = {9223372034707292159, 9223372034707292159};
static const struct rotationMatrix PI_OVER_TWO_ROTATION = {
        {0,-1,0,-1},
        {1,0,1,0}
};

static const struct rotationMatrix THREE_PI_OVER_TWO_ROTATION = {
        {0,1, 0,1},
        {-1,0, -1,0}
};

static const struct rotationMatrix CONJ_TRANSFORM = {
        {1, 0, 1, 0},
        {0, -1, 0, -1}
};

static uint8_t sampleMax = 0;
static uint8_t isCheckADCMax;
static uint8_t isRdc;
static uint8_t isAdc;
static uint8_t isOffsetTuning;
static uint32_t samplePowSum = 0;
static uint32_t rdcBlockScalar = 9 + 1;
static uint32_t adcBlockScalar = 9 + 1;
static __m128 rdcBlockVect;
static __m128 rdcBlockRVectP1;
static __m128 adcBlockVect;
static __m128 adcBlockRVectP1;

/**
 * Takes two packed int16_ts representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, br, aj, bj}
 * and returns their argument as a float
 **/
extern float argzB(__m128 a);
__asm__(
#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif
    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm3\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm3, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vsqrtps %xmm1, %xmm1\n\t"              // ..., Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    // push
    "sub $16, %rsp \n\t"
    "vextractps $1, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    // push
    "sub $16, %rsp \n\t"
    "vextractps $2, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    "fpatan \n\t"
    "fstps (%rsp) \n\t"

    // B I G pop and return
    "vmovq (%rsp), %xmm0 \n\t"
    "add $32, %rsp \n\t"
    "ret"
);

extern float ffabsf(float x);
__asm__( // ffabsf
#ifdef __clang__
"_ffabsf: "
#else
"ffabsf: "
#endif
    "movq %xmm0, %rax\n\t"
    "andl $0x7FFFFFFF, %eax\n\t"
    "movq %rax, %xmm0\n\t"
    "ret"
);

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m128 mm256Epi8convertPs(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_cvtepi8_epi16(lo_lane)));
}

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}, Tu =
 * {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
static __m128 apply4x4_4x1Transform(const struct rotationMatrix T, const __m128 u) {

    __m128 temp, temp1;

    temp = _mm_mul_ps(T.a1.v, u);           // u1*a11, u2*a12, u3*a13, ...
    temp1 = _mm_mul_ps(T.a2.v, u);          // u1*a21, u2*a22, ...
    return _mm_blend_ps(_mm_add_ps(temp,       // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
            _mm_permute_ps(temp, _MM_SHUFFLE(2,3,0,1))),
                _mm_add_ps(temp1,              // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
            _mm_permute_ps(temp1, _MM_SHUFFLE(2,3,0,1))),
        0xA);                                     // u1*a11 + u2*a12, u1*a21 + u2*a22,
                                                  // u3*a13 + u4*a14, u3*a23 + u4*a24
    // A = 0000 1010 = 00 22 => _MM_SHUFFLE(0,0,2,2)
}

static struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {

    const int16_t cosT = cos(theta) * (1 << 13);
    const int16_t sinT = sin(phi) * (1 << 13);

    struct rotationMatrix result = {
            .a1 = {cosT, -sinT, cosT, -sinT},
            .a2 = {sinT, cosT, sinT, cosT}
    };

    return result;
}

static uint32_t downSample(__m128 *buf, uint32_t len, const uint32_t downsample) {

    int i,j;

    for (j = 0; j < downsample; ++j) {
        for (i = 0; i < len; ++i) {
            buf[i >> 1] = _mm_add_ps(buf[i],_mm_permute_ps(buf[i],
                _MM_SHUFFLE(1, 0, 3, 2)));
        }
    }

    return len >> downsample;
}

//static void filterDCAudio(__m128 *buf, const uint32_t len) {
//    static __m128 dcAvg = {0,0,0,0};
//
//    const __m128 halfLen = _mm_set1_ps(1.f/(len << 2));
//
//    uint32_t i;
//    __m128 sum = {0,0,0,0};
//    __m128 avg;
//    for (i = 0; i < len; ++i) {
//        sum = _mm_add_ps(sum, buf[i]);
//    }
//    sum =  _mm_add_ps(sum,_mm_permute_ps(sum, _MM_SHUFFLE(0,1,2,3)));
//
//    avg = _mm_mul_ps(sum, halfLen);
//    avg = _mm_add_ps(avg, _mm_mul_ps(dcAvg, adcBlockVect));
//    avg = _mm_mul_ps(avg, adcBlockRVectP1);
//}

void filterDCAudio(float *buf, const uint32_t len)
{
    static float dcAvg = 0.f;

    int i;
    float avg;
    float sum = 0.f;

    for (i=0; i < len; i++) {
        if (!isnan(buf[i])) sum += buf[i];
    }

    avg = ffabsf(sum) / len;
    avg = (avg + dcAvg * adcBlockScalar) / (adcBlockScalar + 1 );
    for (i=0; i < len; i++) {
        buf[i] -= avg;
    }
    dcAvg = avg;
}

static void filterDCRawIQ(__m128 *buf, const uint32_t len) {

    static __m128 dcAvgIq = {0,0,0,0};

    const __m128 halfLen = _mm_set1_ps(1.f/(len << 1)); // 1 / (length/2), for length = (depth*width)
                                                           // = depth*4 => 1/(2 depth) => 1/(2 len)
    __m128 sumIq = {0,0,0,0};
    __m128 avgIq;
    int i;

    for (i = 0; i < len; ++i) {
        sumIq = _mm_add_ps(sumIq, _mm_add_ps(buf[i],
         _mm_permute_ps(buf[i], _MM_SHUFFLE(0,1,3,2))));
    }
    sumIq = _mm_add_ps(sumIq,_mm_permute_ps(sumIq, _MM_SHUFFLE(0,1,3,2)));
    sumIq = _mm_and_si128(sumIq, FLOAT_ABS);

    avgIq = _mm_mul_ps(sumIq, halfLen);
    avgIq = _mm_add_ps(avgIq, _mm_mul_ps(dcAvgIq, rdcBlockVect));
    avgIq = _mm_mul_ps(avgIq, rdcBlockRVectP1);

    for (i = 0; i < len; ++i) {
        buf[i] = _mm_sub_ps(buf[i], avgIq);
    }
    dcAvgIq = avgIq;
}

static void rotateForNonOffsetTuning(__m128 *buf, const uint32_t len) {

    int i;

    for(i = 0; i < len; ++i) {
        buf[i] = apply4x4_4x1Transform(CONJ_TRANSFORM, buf[i]);
    }
}

static void findMaxSample(const __m256i *buf8, const uint32_t len) {

    uint32_t i;
    __m256i sm;

    for (i = 0; i < len; ++i) {
        sm = _mm256_max_epu8(sm, buf8[i]);
    }
    sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
}

static uint64_t demodulateFmData(__m128 *buf, const uint32_t len, float **result) {

    uint64_t i;

    *result = calloc(len, OUTPUT_ELEMENT_BYTES);
    for (i = 0; i < len; ++i) {
        (*result)[i] = argzB(_mm_mul_ps(buf[i], NEGATE_B_IM));
    }

    return i >> 1;
}


static uint32_t breakit(const uint8_t *buf, const uint32_t len, __m128 *result) {

    uint32_t j = 0;
    uint32_t i;
    int32_t leftToProcess = len;

    union {
        uint8_t buf[VECTOR_WIDTH];
        __m256i v;
    } z;

    for (i = 0; leftToProcess > 0; i += VECTOR_WIDTH) {

        memcpy(z.buf, buf + i, VECTOR_WIDTH);
        result[j++] = mm256Epi8convertPs(_mm256_sub_epi8(z.v, Z));
        leftToProcess -= VECTOR_WIDTH;
    }

    return j;
}

static uint32_t processMatrix(const uint8_t *buf, const uint32_t len, __m128 **buff) {


    uint32_t i, depth;
    uint32_t count = (len & 3) != 0 // len/VECTOR_WIDTH + (len % VECTOR_WIDTH != 0 ? 1 : 0))
            ? (len >> LOG2_VECTOR_WIDTH) + 1
            : (len >> LOG2_VECTOR_WIDTH);

    *buff = calloc(count, MATRIX_ELEMENT_BYTES);

    depth = breakit(buf, len, *buff);

    if (isCheckADCMax) {
        for (i = 0; i < len; ++i) { // TODO implement this with data parallelism
            samplePowSum += buf[i]*buf[i];
        }
        samplePowSum += samplePowSum / len;

//        findMaxSample(buf8, depth); // TODO fix this for nx4 float matrix
    }

    if (isRdc) {
        filterDCRawIQ(*buff, depth);
    }

    if (!isOffsetTuning) {
        rotateForNonOffsetTuning(*buff, depth);
    }

    return depth;
}

static inline uint32_t readFileData(char *path, uint8_t **buf) {
    *buf = calloc(MAXIMUM_BUF_SIZE, INPUT_ELEMENT_BYTES);
    FILE *file = fopen(path, "rb");
    uint32_t result = fread(*buf, INPUT_ELEMENT_BYTES, MAXIMUM_BUF_SIZE, file);

    fclose(file);

    return result;
}

static void initializeEnv(void) {

    rdcBlockVect = _mm_set1_ps(rdcBlockScalar);
    rdcBlockRVectP1 = _mm_set1_ps(1.f/(rdcBlockScalar + 1));

    adcBlockVect = _mm_set1_ps(adcBlockScalar);
    adcBlockRVectP1 = _mm_set1_ps(1.f/(adcBlockScalar + 1));
}

uint32_t permutePairsForDemod(__m128 *buf, uint64_t len, __m128 **result) {

    int i,j;

    *result = calloc(len << 1, MATRIX_ELEMENT_BYTES);

    for (i = 0, j = 0; i < len; ++i, j+=2) {
        (*result)[j] = buf[i];
        (*result)[j + 1] = _mm_blend_ps(buf[i], buf[i + 1], 0b0011);
#ifdef DEBUG
        union m128_f temp = {.v = (*result)[j]};
        printf("(%.01f + %.01fI),\t(%.01f + %.01fI)\n",
               temp.buf[0], temp.buf[1], temp.buf[2], temp.buf[3]);

        temp.v = (*result)[j + 1];
        printf("(%.01f + %.01fI),\t(%.01f + %.01fI)\n",
               temp.buf[0], temp.buf[1], temp.buf[2], temp.buf[3]);
#endif
    }
    return (len & 3) != 0 ? (len << 1) - 1 : len << 1;
}

int main(int argc, char **argv) {

    static uint8_t previousR, previousJ;

    int i = 1;
    uint64_t depth;
    uint32_t len;
    float *result;
    __m128 *lowPassed;
    __m128 *permuted;
    uint32_t downsample;
    int argsProcessed;

#ifndef DEBUG
    char *inPath, *outPath;
    argsProcessed = 3;
#else
    argsProcessed = 1;
#endif

    if (argc < argsProcessed) {
        return -1;
    } else {
        isCheckADCMax = 0;
        isAdc = isRdc = 0;
        isOffsetTuning = 0;
        downsample = 1;

#ifndef DEBUG
        inPath = argv[1];
        outPath = argv[2];
        i = 3;
#endif
        for (; i < argc; ++i) {
            switch (argv[i][0]){
                case 'r':
                    isRdc = 1;
                    argsProcessed++;
                    break;
                case 'a':
                    isAdc = 1;
                    argsProcessed++;
                    break;
                case 'o':
                    isOffsetTuning = 1;
                    argsProcessed++;
                default:
                    argsProcessed--;
                    break;
            }
        }

        if (argsProcessed != argc) {
            downsample = atoi(argv[argc - 1]);
        }
    }

#ifdef DEBUG
    uint8_t buf[18] = {128,129,130,131,132,133,134,135,
                       136,137,138,139,140,141,142,143, 0,0};
    len = sizeof(buf)/sizeof(*buf);

    printf("%hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu\n"
           "%hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu\n\n",
           buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
           buf[8],buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
           buf[16], buf[17]);
#else
    uint8_t *inBuf;
    uint8_t *buf;
    FILE *file;

    len = readFileData(inPath, &inBuf) + 2;
    buf = calloc(len, INPUT_ELEMENT_BYTES);

    buf[0] = previousR;
    buf[1] = previousJ;
    memcpy(buf + 2, inBuf, len - 2);
    previousJ = buf[len-1];
    previousR = buf[len-2];

    free(inBuf);
#endif

    initializeEnv();

    depth = processMatrix(buf, len, &lowPassed);

#ifdef DEBUG
    printf("Processed matrix:\n");
    for (i = 0; i < depth; ++i) {
        union m128_f temp = {.v = lowPassed[i]};
        printf("(%.01f + %.01fI),\t(%.01f + %.01fI)\n",
               temp.buf[0], temp.buf[1], temp.buf[2], temp.buf[3]);
    }
    printf("\n");
#endif

    depth = downSample(lowPassed, depth, downsample);

#ifdef DEBUG
    printf("Downsampled and windowed:\n");
    for (i = 0; i < depth; ++i) {
        union m128_f temp = {.v = lowPassed[i]};
        printf("(%.02f + %.02fI),\t(%.02f + %.02fI)\n",
            temp.buf[0], temp.buf[1], temp.buf[2], temp.buf[3]);
    }
    printf("\nPermuted pairs:\n");
#endif

    depth = permutePairsForDemod(lowPassed, depth, &permuted);
    free(lowPassed);

    depth = demodulateFmData(permuted, depth, &result);
    free(permuted);

    if (isAdc) {
        filterDCAudio(result, depth);
    }
#ifdef DEBUG
    printf("\nPhase angles:\n");
    for (i = 0; i < depth; ++i) {
        printf("%f, ", result[i]);
    }
    printf("\n");
#else
    file = fopen(outPath, "wb");
    fwrite(result, OUTPUT_ELEMENT_BYTES, depth, file);
    fclose(file);
#endif

    free(result);
}
