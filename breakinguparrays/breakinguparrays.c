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

#define DEBUG
// sizeof(uint8_t)
#define INPUT_ELEMENT_BYTES 1
// sizeof(float)
#define OUTPUT_ELEMENT_BYTES 4
#define VECTOR_WIDTH 4
#define MAXIMUM_BUF_SIZE 1L << 33

union m256_16 {
    int16_t buf[4];
    __m256i v;
};

struct rotationMatrix {
    const union m256_16 a1;
    const union m256_16 a2;
};

#ifndef DEBUG
static const __m256i NEGATE_B_IM = {281479271809023, 0, 0, 0};
//static const __m256i NEGATE_B_IM = {281483566579713, 0, 0, 0};
#else
static const __m256i NEGATE_B_IM = {281479271743489, 0, 0, 0};
#endif
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};
//static const __m256 FIXED_PT_SCALE
//    = {5215.18896, 5215.18896, 5215.18896, 5215.18896, 5215.18896, 5215.18896, 5215.18896, 5215.18896};

/**
 * Takes two packed int16_ts representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, br, aj, bj}
 * and returns their argument as a float
 **/
extern float argzB(__m256i a, __m256i b);
__asm__(
#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif
    "vpshuflw $0x05, %xmm0, %xmm0\n\t" // ar, br, aj, bj => (aj, aj, ar, ar)
    "vpshuflw $0xEB, %xmm1, %xmm1\n\t" // and               (bj, br, br, bj)
    "vpmovsxwd %xmm0, %xmm0\n\t"
    "vpmovsxwd %xmm1, %xmm1\n\t"
    "vcvtdq2ps %xmm0, %xmm0\n\t"
    "vcvtdq2ps %xmm1, %xmm1\n\t"

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm3\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm3, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vsqrtps %xmm1, %xmm1\n\t"              // ..., Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/r , zr/r = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

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

static const struct rotationMatrix PI_OVER_TWO_ROTATION = {
        {0,-1,0,-1},
        {1,0,1,0}
};

static const struct rotationMatrix THREE_PI_OVER_TWO_ROTATION = {
        {0,1, 0,1},
            {-1,0, -1,0}
};

static const struct rotationMatrix CONJ_TRANSFORM = {
        {1, 0,1, 0},
        {0, -1,0, -1}
};


static uint8_t sampleMax = 0;
static uint8_t isCheckADCMax;
static uint8_t isRdc;
static uint8_t isOffsetTuning;
static uint32_t samplePowSum = 0;
static uint32_t rdcBlockScalar = 9 + 1;
//static uint32_t downsample;
static __m256i *lowPassed;
static __m256i dcAvgIq = {0,0,0,0};
static __m256i rdcBlockVector;
static __m256i rdcBlockRVector;
static __m256i rdcBlockVect1;

static inline uint8_t uCharMax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline __m256i mm256Epi8convertEpi16(__m256i data) {
    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm256_cvtepi8_epi16(lo_lane);
}

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}
 *  -> {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
static __m256i apply4x4_4x1Transform(const struct rotationMatrix T, const __m256i u) {
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

static struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {

    int16_t cosT = cos(theta) * (1 << 13);
    int16_t sinT = sin(phi) * (1 << 13);
    struct rotationMatrix result = {
            .a1 = {cosT, -sinT, cosT, -sinT},
            .a2 = {sinT, cosT, sinT, cosT}
    };

    return result;
}

static void filterSimpleBox(const __m256i *buf, const uint32_t len) {
    int i;

    lowPassed = calloc(len, sizeof(__m256i));

    for (i = 0; i < len; ++i) {
        lowPassed[i]
            = _mm256_add_epi16(buf[i],
               _mm256_shufflelo_epi16(buf[i], _MM_SHUFFLE(1, 0, 3, 2)));
    }
}

static void filterRawDc(__m256i *buf, const uint32_t len) {

    const __m256i oneOverLen = _mm256_set1_epi16(16384/len); // 1/(len/2) = 2/len

    __m256i sumIq = {0,0,0,0};
    __m256i avgIq;

    int i;
    for (i = 0; i < len; ++i) {
        sumIq = _mm256_add_epi16(sumIq, _mm256_add_epi16(buf[i],
         _mm256_shufflelo_epi16(buf[i], _MM_SHUFFLE(2, 3, 0, 1))));
    }

    avgIq = _mm256_mulhrs_epi16(sumIq, oneOverLen);
    avgIq = _mm256_add_epi16(avgIq, _mm256_mullo_epi16(dcAvgIq, rdcBlockVector));
    avgIq = _mm256_mulhrs_epi16(avgIq, rdcBlockRVector);
    avgIq = _mm256_mullo_epi16(avgIq, rdcBlockVect1);

    for (i = 0; i < len; ++i) {
        buf[i] = _mm256_sub_epi16(buf[i], avgIq);
    }

    dcAvgIq = avgIq;
}

static void rotateForNonOffsetTuning(__m256i *buf, const uint32_t len) {

    int i, j;

    for(i = 0, j = 0; i < len; ++i, j+=4) {
        buf[i] = apply4x4_4x1Transform(CONJ_TRANSFORM, buf[i]);
    }
}

static int convertTo16Bit(const __m256i *buf, const uint32_t len, __m256i *buf16) {

    uint32_t i;

    for (i = 0; i < len; ++i) {
        buf16[i] = mm256Epi8convertEpi16(_mm256_sub_epi8(buf[i], Z));
    }

    return i;
}

static void findMaxSample(const __m256i *buf8, const uint32_t len) {

    int i;
    __m256i sm;

    for (i = 0; i < len; ++i) {
        sm = _mm256_max_epu8(sm, buf8[i]);
    }
    sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
}

static uint64_t demodulateFmData(const uint32_t len, float **result) {

    uint64_t i;

    *result = calloc(len >> 1, OUTPUT_ELEMENT_BYTES);

    for (i = 0; i < len; i+=2) {

        (*result)[i >> 1] = argzB(_mm256_mullo_epi16(
                lowPassed[i],                       NEGATE_B_IM),
                lowPassed[i+1]);
    }

    return i >> 1;
}


static uint32_t breakit(const uint8_t *buf, const uint32_t len, __m256i *buf8) {

    uint32_t j = 0;
    uint32_t i;
    int32_t leftToProcess = len;

    union {
        uint8_t buf[VECTOR_WIDTH];
        __m256i v;
    } z;

    for (i = 0; leftToProcess > 0; i += VECTOR_WIDTH, ++j) {

        memcpy(z.buf, buf + i, VECTOR_WIDTH);
        buf8[j] = z.v;
        leftToProcess -= VECTOR_WIDTH;
    }

    return j;
}

static uint32_t processMatrix(const uint8_t *buf, const uint32_t len, __m256i **buf16) {


    int i, depth;
    uint32_t count = len << 1;// len / VECTOR_WIDTH + ((len & 3) != 0 ? 1 : 0); // len % 4 != 0
    __m256i *buf8 = calloc(count, sizeof(__m256i));

    *buf16 = calloc(count, sizeof(__m256i));

    depth = breakit(buf, len, buf8);

    if (isCheckADCMax) {
        for (i = 0; i < len; ++i) { // TODO implement this with data parallelism
            samplePowSum += buf[i]*buf[i];
        }
        samplePowSum += samplePowSum / len;

        findMaxSample(buf8, depth);
    }

    depth = convertTo16Bit(buf8, depth, *buf16);

    free(buf8);

    if (isRdc) {
        filterRawDc(*buf16, depth);
    }

    if (!isOffsetTuning) {
        rotateForNonOffsetTuning(*buf16, depth);
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

    const int16_t scalarP1 = rdcBlockScalar + 1;
    rdcBlockVector = _mm256_set1_epi16(rdcBlockScalar);
    rdcBlockVect1 = _mm256_set1_epi16(scalarP1);//(union m256_16){scalarP1, 0x1, scalarP1, 0x1}.v;
    rdcBlockRVector = _mm256_set1_epi16(32768/scalarP1);//(union m256_16){32768/scalarP1, 0x1, 32768/scalarP1, 0x1}.v;
}

int main(int argc, char **argv) {

    int i;
    uint64_t depth;
    uint32_t len;
    float *result;
    __m256i *buf16;

#ifdef DEBUG
    len = 16;
    uint8_t buf[17] = {128,129,130,131,132,133,134,135,
                       136,137,138,139,140,141,142,143, 0};
    isCheckADCMax = 0;
    isRdc = 1;
    isOffsetTuning = 0;

    printf("%hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu,\n"
           "%hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu\n",
           buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
           buf[8],buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15], buf[16]);
    printf("\n");
#else
    uint8_t previousR, previousJ;
    uint8_t *inBuf;
    char *inPath, *outPath;
    uint8_t *buf;
    FILE *file;

    if (argc <= 2) {
        return -1;
    } else {
        isCheckADCMax = 0;
        isRdc = 0;
        isOffsetTuning = 0;
//        downsample = 2;
        inPath = argv[1];
        outPath = argv[2];

        for (i = 3; i < argc; ++i) {
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

    len = readFileData(inPath, &inBuf) + 2; //((1 << 6) + 27 + 2)
    buf = calloc(len, INPUT_ELEMENT_BYTES);

    buf[0] = previousR;
    buf[1] = previousJ;
    memcpy(buf + 2, inBuf, len - 2);
    previousJ = buf[len-1];
    previousR = buf[len-2];

    free(inBuf);
#endif

    initializeEnv();

    depth = processMatrix(buf, len, &buf16);

#ifdef DEBUG
    for (i = 0; i < depth; ++i) {
        union m256_16 temp = {.v = buf16[i]};
        printf("(%hd + %hdI),\t(%hd + %hdI)\n",
               temp.buf[0], temp.buf[1], temp.buf[2], temp.buf[3]);
    }
    printf("\n");
#endif

    filterSimpleBox(buf16, depth);

#ifdef DEBUG
    for (i = 0; i < depth; i+=2) {
        union m256_16 temp = {.v = lowPassed[i]};
        printf("(%hd + %hdI),\t",
               temp.buf[0], temp.buf[1]);

        temp.v = lowPassed[i+1];
        printf("(%hd + %hdI)\n",
            temp.buf[0], temp.buf[1]);
    }
    printf("\n");
#endif

    free(buf16);

    depth = demodulateFmData(depth, &result);

    free(lowPassed);

#ifdef DEBUG
//    printf("%f, %f\n", result[0], result[1]);
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
