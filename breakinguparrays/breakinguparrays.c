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
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

//#define DEBUG

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

static const __m128 HUNDREDTH = {0.01f, 0.01f, 0.01f, 0.01f};
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
//static uint8_t isCheckADCMax;
static uint8_t isRdc;
//static uint8_t isAdc;
static uint8_t isOffsetTuning;
static uint32_t samplePowSum = 0;

/**
 * Takes two packed floats representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns their argument as a float
 **/
extern float argz(__m128 a);
__asm__(
#ifdef __clang__
"_argz: "
#else
"argz: "
#endif
    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm3\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm3, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vrsqrtps %xmm1, %xmm1\n\t"              // ..., Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    "comiss %xmm0, %xmm1\n\t"
    "jp zero\n\t"
    // push
    "vextractps $1, %xmm0, -8(%rsp) \n\t"
    "flds -8(%rsp) \n\t"
    // push
    "vextractps $2, %xmm0, -8(%rsp) \n\t"
    "flds -8(%rsp) \n\t"
    "fpatan \n\t"
    "fstps -8(%rsp) \n\t"

    // pop and return
    "vmovq -8(%rsp), %xmm0 \n\t"
    "jmp return\n\t"

"zero: "
    "vxorps %xmm0, %xmm0, %xmm0\n\t"
"return: "
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

static inline __m128 mm256Epi8convertmmPs(__m256i data) {
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

    uint64_t i,j;

    for (j = 0; j < downsample; ++j) {
        for (i = 0; i < len; ++i) {
            buf[i >> 1] = _mm_add_ps(buf[i],_mm_permute_ps(buf[i],
                _MM_SHUFFLE(1, 0, 3, 2)));
        }
    }

    return len >> downsample;
}

//void filterDCAudio(float *buf, const uint32_t len)
//{
//    static float dcAvg = 0.f;
//
//    int i;
//    float avg;
//    float sum = 0.f;
//
//    for (i=0; i < len; i++) {
//        sum += buf[i];
//    }
//
//    avg = ffabsf(sum) / len;
//    avg = (avg + dcAvg * adcBlockScalar) / (adcBlockScalar + 1 );
//    for (i=0; i < len; i++) {
//        buf[i] -= avg;
//    }
//    dcAvg = avg;
//}

static void removeDCSpike(__m128 *buf, const uint32_t len) {

    static const __m128 ratio = {1e-05f,1e-05f,1e-05f,1e-05f};
    static __m128 dcAvgIq = {0,0,0,0};

    uint64_t i;

    for (i = 0; i < len; ++i) {
        dcAvgIq = _mm_add_ps(dcAvgIq, _mm_mul_ps(ratio, _mm_sub_ps(buf[i], dcAvgIq)));
        buf[i] = _mm_sub_ps(buf[i], dcAvgIq);
    }
}

static void rotateForNonOffsetTuning(__m128 *buf, const uint32_t len) {

    uint64_t i;

    for(i = 0; i < len; ++i) {
        buf[i] = apply4x4_4x1Transform(CONJ_TRANSFORM, buf[i]);
    }
}

static uint64_t demodulateFmData(__m128 *buf, const uint32_t len, float **result) {

    uint64_t i, j;

    *result = calloc(len << 1, OUTPUT_ELEMENT_BYTES);
    for (i = 0, j = 0; i < len; ++i, j += 2) {
        (*result)[j] = argz(_mm_mul_ps( buf[i], NEGATE_B_IM));
        (*result)[j+1] = argz(_mm_mul_ps(_mm_blend_ps(buf[i], buf[i+1], 0b0011), NEGATE_B_IM));
    }

    return j;
}


static uint32_t breakit(const uint8_t *buf, const uint64_t len, __m128 *result, __m128 *squelch) {

    uint64_t j = 0;
    uint64_t i;
    int64_t leftToProcess = len;
    __m128 rms, mask;

    union {
        uint8_t buf[VECTOR_WIDTH];
        __m256i v;
    } z;

    for (i = 0; leftToProcess > 0; i += VECTOR_WIDTH) {

        memcpy(z.buf, buf + i, VECTOR_WIDTH);
        result[j] = mm256Epi8convertmmPs(_mm256_sub_epi8(z.v, Z));

        if (squelch) {
            rms = _mm_mul_ps(result[j], result[j]);
            rms = _mm_mul_ps(HUNDREDTH,
                             _mm_add_ps(rms, _mm_permute_ps(rms, _MM_SHUFFLE(2, 3, 0, 1))));
            mask = _mm_cmp_ps(rms, *squelch, _CMP_GE_OQ);
            result[j] = _mm_and_ps(result[j], mask);
        }
        j++;
        leftToProcess -= VECTOR_WIDTH;
    }

    return j;
}

static uint64_t processMatrix(const uint8_t *buf, const uint64_t len, __m128 **buff, __m128 *squelch) {


    uint64_t depth;
    uint64_t count = (len & 3UL) != 0 // len/VECTOR_WIDTH + (len % VECTOR_WIDTH != 0 ? 1 : 0))
            ? (len >> LOG2_VECTOR_WIDTH) + 1UL
            : (len >> LOG2_VECTOR_WIDTH);

    *buff = calloc(count << 2, MATRIX_ELEMENT_BYTES);

    depth = breakit(buf, len, *buff, squelch);

    if (isRdc) {
        removeDCSpike(*buff, depth);
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
    *buf = realloc(*buf, INPUT_ELEMENT_BYTES * result);

    return result;
}

int main(int argc, char **argv) {

    static uint8_t downsample;
    static __m128 *squelch;

    int opt;
    uint64_t depth;
    uint64_t len;
    float *result;
    __m128 *lowPassed;

#ifndef DEBUG
    static uint8_t previousR, previousJ;
    char *inPath, *outPath;
    int argsProcessed = 3;
#else
    uint64_t i;
    int argsProcessed = 1;
#endif

    if (argc < argsProcessed) {
        return -1;
    } else {
        isRdc = 0;
        isOffsetTuning = 0;
        downsample = 0;

        while ((opt = getopt(argc, argv, "r:i:o:d:f:s:")) != -1) {
            switch (opt) {
                case 'r':
                    isRdc = atoi(optarg);
                    break;
                case 'f':
                    isOffsetTuning = atoi(optarg);
                    break;
                case 'd':
                    downsample = atoi(optarg);
                    break;
                case 's':   // TODO add parameter to take into account the impedence of the system
                            // currently calculated for 50 Ohms (i.e. Prms = ((I^2 + Q^2)/2)/50 = (I^2 + Q^2)/100)
                    squelch = malloc(MATRIX_ELEMENT_BYTES);
                    *squelch = _mm_set1_ps(powf(10.f, atof(optarg) / 10.f));
                    break;
#ifndef DEBUG
                case 'i':
                    inPath = optarg;
                    break;
                case 'o':
                    outPath = optarg;
                    break;
#endif
                default:
                    break;

            }
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

    depth = processMatrix(buf, len, &lowPassed, squelch);

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
    printf("\n");
#endif

    depth = demodulateFmData(lowPassed, depth, &result);
    free(lowPassed);

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
