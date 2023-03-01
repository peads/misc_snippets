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

// sizeof(uint8_t)
#define INPUT_ELEMENT_BYTES 1
#define LOG2_LENGTH 4
#define LENGTH (1 << LOG2_LENGTH)
#define DEFAULT_BUF_SIZE		32768
#define MAXIMUM_BUF_SIZE		(DEFAULT_BUF_SIZE << LOG2_LENGTH)
// therefore, max depth is  MAXIMUM_BUF_SIZE >> 2

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

static const uint32_t VECTOR_WIDTH = INPUT_ELEMENT_BYTES << LOG2_LENGTH;
static const __m256i ZERO = {0, 0, 0, 0};
static const __m256i ONE = {1, 1, 1, 1};
static const __m256i Z // all 127s
    = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};
/**
 * takes four float representing the complex numbers
 * (ar + iaj) * (br + ibj), * s.t. z = {ar, aj, br, bj}
 * and return their argument.
 **/
extern __m256i argzB(__m256i z, __m256i w);
__asm__(
#ifdef __clang__
"_argzB: "
#else
"argzB: "
#endif
//    "vshufps $0x11, %xmm0, %xmm1, %xmm0\n\t"
//    "vpermilps $0xDD, %xmm0, %xmm0\n\t"      // aj, aj, ar, ar -> xmm0
//    "vpermilps $0x44, %xmm0, %xmm1\n\t"        // 10 10 = 0100 0100 = 44

//    "vshufps $0, %xmm3, %xmm2, %xmm1\n\t"
//    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1
    "vpshuflw $0xEB, %xmm1, %xmm1\n\t" //
    "vcvtdq2ps %xmm0, %xmm0\n\t"
    "vcvtdq2ps %xmm1, %xmm1\n\t"

    "vmulps %xmm0, %xmm1, %xmm0\n\t"        // aj*bj, ar*br, aj*br, ar*bj
    "vpermilps $0xB1, %xmm0, %xmm3\n\t"
    "vaddsubps %xmm0, %xmm3, %xmm0\n\t"     // ar*br - aj*bj, ... , ar*bj + aj*br
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // (ar*br - aj*bj)^2, ... , (ar*bj + aj*br)^2
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"     // 0123 = 00011011 = 1B
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...
    "vsqrtps %xmm1, %xmm1\n\t"              // Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vdivps %xmm1, %xmm0, %xmm0\n\t"        // (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    // push
    "sub $16, %rsp \n\t"
    "vextractps $0, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    // push
    "sub $16, %rsp \n\t"
    "vextractps $3, %xmm0, (%rsp) \n\t"
    "flds (%rsp) \n\t"
    "fpatan \n\t"
    "fstps (%rsp) \n\t"

    // B I G pop
    "vmovq (%rsp), %xmm0 \n\t"
    "add $32, %rsp \n\t"

    "movl $0x40490fdb, %edx\n\t" // Pi
    "vmovq %rdx, %xmm1\n\t"
    "vpbroadcastq %xmm1, %xmm1\n\t"
    "vdivps %xmm1, %xmm0, %xmm0\n\t"

    "vcvttps2dq %xmm0, %xmm0\n\t"
    "vpmulhrsw %xmm1, %xmm0, %xmm0\n\t"
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

static uint8_t sampleMax = 0;
static uint8_t isCheckADCMax;
static uint8_t isRdc;
static uint8_t isOffsetTuning;
static uint32_t samplePowSum = 0;
static uint32_t rdcBlockScalar = 9 + 1;
static uint32_t downsample;
static __m256i *buf8;
static __m256i *buf16x4;
static __m256i *lowPassed;
static __m256i dcAvgIq = {0,0,0,0};
static __m256i rdcBlockVector;
static __m256i rdcBlockRVector;
static __m256i rdcBlockVect1;
static int16_t previousR = 0;
static int16_t previousJ = 0;

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
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}
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


static struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {

    int16_t cosT = cos(theta) * (1 << 13);
    int16_t sinT = sin(phi) * (1 << 13);
    struct rotationMatrix result = {
            .a1 = {cosT, -sinT, cosT, -sinT},
            .a2 = {sinT, cosT, sinT, cosT}
    };

    return result;
}

static void filterSimpleLowPass(const uint32_t len) {
    int i;

    lowPassed = calloc(len << 1, sizeof(__m256i));

    for (i = 0; i < len; ++i) {
        lowPassed[i]
            = _mm256_add_epi16(buf16x4[i],
                               _mm256_shufflelo_epi16(buf16x4[i], _MM_SHUFFLE(1, 0, 3, 2)));
    }
}

static void filterDcBlockRaw(const uint32_t len) {

    const __m256i oneOverLen = _mm256_set1_epi16(16384/len);

    __m256i sumIq = {0,0,0,0};
    __m256i avgIq;

    int i;
    for (i = 0; i < len; ++i) {
        sumIq = _mm256_add_epi16(sumIq, _mm256_add_epi16(buf16x4[i],
                                                         _mm256_shufflelo_epi16(buf16x4[i], _MM_SHUFFLE(2, 3, 0, 1))));
    }

    avgIq = _mm256_mulhrs_epi16(sumIq, oneOverLen);
    avgIq = _mm256_add_epi16(avgIq, _mm256_mullo_epi16(dcAvgIq, rdcBlockVector));
    avgIq = _mm256_mulhrs_epi16(avgIq, rdcBlockRVector);
    avgIq = _mm256_mullo_epi16(avgIq, rdcBlockVect1);

    for (i = 0; i < len; ++i) {
        buf16x4[i] = _mm256_sub_epi16(buf16x4[i], avgIq);
    }

    dcAvgIq = avgIq;
}

static void rotateForNonOffsetTuning(const uint32_t len) {

    int i, j;

    for(i = 0, j = 0; i < len; ++i, j+=4) {
        buf16x4[i] = applyRotationMatrix(PI_OVER_TWO_ROTATION, buf16x4[i]);
    }
}

static int splitToQuads(const uint32_t len) {

    int i, j, k;

    for (j = 0, k = 0; j < len; ++j) {

        union m256_8 w = {.v = buf8[j]};

        for (i = 0; i < LENGTH; i += 4) {
            buf16x4[k++]
                = (union m256_16){w.buf16[i], w.buf16[i+1], w.buf16[i+2], w.buf16[i+3]}.v;
        }
    }

    return k;
}

static void findMaxSample(const uint32_t len) {

    int i;
    __m256i sm;

    for (i = 0; i < len; ++i) {
        sm = _mm256_max_epu8(sm, buf8[i]);
    }
    sampleMax = uCharMax(sampleMax, uCharMax(sm[0], sm[1]));
}

static void demodulateFmData(const uint32_t len) {
    union m256_16 temp;
    int i;
    for (i = 0; i < len; i+=2) {
        temp.v = lowPassed[i];
        lowPassed[i] =  argzB(lowPassed[i], lowPassed[i+1]);
    }

    temp.v = lowPassed[len-1];
    previousR = temp.buf[2];
    previousJ = temp.buf[3];
}

static uint32_t convertTo16BitNx4Matrix(const uint32_t len) {

    int i,depth;
    buf16x4 = calloc(len << LOG2_LENGTH, sizeof(__m256i));

    if (isCheckADCMax) {

        findMaxSample(len);
    }

    for (i = 0; i < len; i++) {
        buf8[i] = convert(_mm256_sub_epi8(buf8[i], Z));
    }

    depth = splitToQuads(len);

    if (isRdc) {
        filterDcBlockRaw(depth);
    }

    if (!isOffsetTuning) {
        rotateForNonOffsetTuning(depth);
    }

    return depth;
}

static void breakit(const uint8_t *buf, const uint32_t len, const uint32_t size) {

    int j = 0;
    int i;
    long leftToProcess = len;
    uint32_t step = INPUT_ELEMENT_BYTES << LOG2_LENGTH;
    uint32_t chunk = step * INPUT_ELEMENT_BYTES;
    union m256_8 z;

//    overRun = len - (size-1)*(step);
    buf8 = calloc(size * chunk, sizeof(__m256i));
//    memset(buf8, 0, size * chunk * sizeof(__m256i));

    for (i = 0; leftToProcess > 0; i += step, ++j) {

        memcpy(z.buf, buf + i, chunk);
        buf8[j] = z.v;
        leftToProcess -= step;
    }
}

static uint32_t initializeEnv(uint8_t *buf, uint32_t len) {

    srand( time(NULL));

    int i;
    uint8_t val;
    for (i = 0; i < len; ++i) {
        if (i % LENGTH == 0) printf("\n");
        buf[i] = (100 + rand()) % 255;
        printf("%hhu, ", buf[i]);
    }
    printf("\n\n");

    const int16_t scalarP1 = rdcBlockScalar + 1;
    rdcBlockVector = _mm256_set1_epi16(rdcBlockScalar);
    rdcBlockVect1 = (union m256_16){scalarP1, 0x1, scalarP1, 0x1}.v;
    rdcBlockRVector = (union m256_16){32768/scalarP1, 0x1, 32768/scalarP1, 0x1}.v;

    return i-1;
}

void multiply(int ar, int aj, int br, int bj, int *cr, int *cj)
{
    *cr = ar*br - aj*bj;
    *cj = aj*br + ar*bj;
}

int polar_discriminant(int ar, int aj, int br, int bj)
{
    int cr, cj;
    double angle;
    multiply(ar, aj, br, -bj, &cr, &cj);
    angle = atan2((double)cj, (double)cr);
    return (int)(angle / 3.14159 * (1<<14));
}

int fm_demod(int len)
{
    int i, j = 0, pcm;
    int16_t pre_r, pre_j, lp[len << 2], result[len << 1];

    for (i = 0; i < len; ++i, j+=4) {
        union m256_16 temp = {.v = lowPassed[i]};
        lp[j] = temp.buf[0];
        lp[j+1] = temp.buf[1];
        lp[j+2] = temp.buf[2];
        lp[j+3] = temp.buf[3];
    }

    pcm = polar_discriminant(lp[0], lp[1],
                             pre_r, pre_j);
    result[0] = (int16_t) pcm;
    for (i = 2; i < (len<<2)-1; i += 2) {

        pcm = polar_discriminant(lp[i], lp[i+1],
                                 lp[i-2], lp[i-1]);

        result[i/2] = (int16_t) pcm;
    }

    pre_r = lp[len - 2];
    pre_j = lp[len - 1];

    for (i = 0; i < len << 1; ++i) {
        if (i % LENGTH == 0) printf("\n");
        printf("%hd, ", result[i]);
    }
    printf("\n");

    return len << 1;
}

int main(int argc, char **argv) {

    static uint8_t *inBuf;
    static uint32_t len = ((1 << 6) + 27 + 2) / INPUT_ELEMENT_BYTES;

    uint8_t buf[len];
    uint32_t size = len / VECTOR_WIDTH + 1;
    int j, i;

    inBuf = calloc(len - 2, INPUT_ELEMENT_BYTES);

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

    initializeEnv(inBuf, len - 2);

    memcpy(buf + 2, inBuf, len - 2);
    buf[0] = previousR;
    buf[1] = previousJ;

    for (i = 0; i < len; ++i) { // TODO implement this with data parallelism
        samplePowSum += buf[i]*buf[i];
    }

    samplePowSum += samplePowSum / (LENGTH*len);

    breakit(buf, len, size);

    for (i = 0; i < size; ++i) {
        union m256_8 z = {.v = buf8[i]};
        for (j = 0; j < LENGTH; ++j) {
            printf("%hhu, ", z.buf[j]);
        }
        printf("\n");
    }

    int depth = convertTo16BitNx4Matrix(size);

    for (j = 0, i = 0; j < depth; ++j, i+=4) {
        if (i % LENGTH == 0) printf("\n");
        union m256_16 w = {.v = buf16x4[j]};

        printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
    }
    printf("\n");

    filterSimpleLowPass(depth);

    for (j = 0; j < depth; ++j) {
        if (j % (LENGTH >> 2) == 0) printf("\n");
        union m256_16 w = {.v = lowPassed[j]};

        printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
    }
    printf("\n");

    fm_demod(depth);
    demodulateFmData(depth);

    for (j = 0; j < depth; ++j) {
        if (j % (LENGTH >> 2) == 0) printf("\n");
        union m256_16 w = {.v = lowPassed[j]};

        printf("%hd, %hd, %hd, %hd, ", w.buf[0],w.buf[1], w.buf[2], w.buf[3]);
    }
    printf("\n");

    free(lowPassed);
    free(buf16x4);
    free(buf8);
    free(inBuf);
}