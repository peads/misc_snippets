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

#include <string.h>
#include "timed_functions.h"

static inline void timespec_diff(const struct timespec *a, const struct timespec *b,
                                 struct timespec *result) {

    result->tv_sec = a->tv_sec - b->tv_sec;
    result->tv_nsec = a->tv_nsec - b->tv_nsec;
    if (result->tv_nsec < 0) {
        --result->tv_sec;
        result->tv_nsec += 1000000000L;
    }
}

void findDeltaTime(const int idx, const struct timespec *tstart, const struct timespec *tend) {

    static uint64_t counts[TIMING_RUNS];
    struct timespec tau;

    timespec_diff(tend, tstart, &tau);

    long double temp = ((long double) tau.tv_nsec + (long double) counts[idx] * rollingTimeAvgs[idx]) / (long double) ++counts[idx];
    if (!isnan(temp)) {rollingTimeAvgs[idx] = temp;}
}

union unDouble timeFund(timedFunD fun, double s, int i) {

    double result;
    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    result = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);

    return (union unDouble) {result};

}

union unFloat timeFunf(timedFunF fun, float s, int i) {

    float result;
    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    result = fun(s);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);

    return (union unFloat) {result};
}

void timeFun(timedFun fun, void *s, void **result, int i) {

    struct timespec tstart, tend;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    fun(s, result != NULL ? result : NULL);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    findDeltaTime(i, &tstart, &tend);
}

void printTimedRuns(char **runNames, uint32_t length) {

    int j;
    for (j = 0; j < length; ++j) {
        const char *runName = runNames[j];
        printf("%sAverage time: %Lf ns\n", runName, rollingTimeAvgs[j]);
    }
}

int signumf(float y) {

    struct {
        union unFloat unf;
        uint8_t signBit;
        uint32_t abs;
    } s = {{y}, s.unf.i >> 31U, s.unf.i & FLOAT_S_FLIP_MASK};

    return s.abs & FLOAT_S_MASK ? NAN : s.abs ? s.signBit ? -1 : 1 : 0;
}

int signum(double y) {

    struct {
        union unDouble unf;
        uint8_t signBit;
        uint64_t abs;
    } s = {{y}, s.unf.i >> 63LU, s.unf.i & DOUBLE_S_FLIP_MASK};

    return s.abs & DOUBLE_S_MASK ? NAN : s.abs ? s.signBit ? -1 : 1 : 0;
}

int bitScanReverse(uint64_t bb) {

    union {
        double d;
        struct {
            uint32_t mantissal:32U;
            uint32_t mantissah:20U;
            uint32_t exponent:11U;
            uint32_t sign:1U;
        };
    } ud;
    ud.d = (double) (bb & ~(bb >> 32));  // avoid rounding error
    return ud.exponent - 1023;
}

uint64_t findMsb(uint64_t n) {

    if (n == 0) {
        return 0;
    }

    int msb;

    for (msb = 0, n >>= 1U; n != 0; n >>= 1U, ++msb) {}

    return (1U << msb);
}

float sqrtApproxf(const float z) {

    union unFloat un = {z};    /* Convert type, preserving bit pattern */
    /*
     * To justify the following code, prove that
     *
     * ((((unf.i / 2^m) - b) / 2) + b) * 2^m = ((unf.i - 2^m) / 2) + ((b + 1) / 2) * 2^m)
     *
     * where
     *
     * b = exponent bias
     * m = number of mantissa bits
     */
    un.i -= 1U << FLOAT_SHIFT;    /* Subtract 2^m. */
    un.i >>= 1U;        /* Divide by 2. */
    un.i += 1U << 29;    /* Add ((b + 1) / 2) * 2^m. */

    return un.f;        /* Interpret again as float */
}

double sqrtApprox(const double z) {

    union unDouble un = {z};         /* Convert type, preserving bit pattern */

    /*
     * To justify the following code, prove that
     *
     * ((((unf.i / 2^m) - b) / 2) + b) * 2^m = ((unf.i - 2^m) / 2) + ((b + 1) / 2) * 2^m)
     *
     * where
     *
     * m = number of mantissa bits => 52
     * b = exponent bias => 1023 => ((b + 1) / 2) * 2^m = (1024 * 2^(51)) = 2^(10+51) = 1 << 61
     */
    un.i -= 1LU << DOUBLE_SHIFT;                /* Subtract 2^m. */
    un.i >>= 1LU;                                  /* Divide by 2. */
    un.i += 1LU << 61LU;                /* Add ((b + 1) / 2) * 2^m. */

    return un.f;        /* Interpret again as float */
}

extern float ffabsf(float x);
inline float aatan(const float z) {
    // x / (1 + 9/32*x^2) == x / (1 + x^2/4 + x^2/32) == 32 x / (9x^2 + 32)
//    return z / (1 + NINE_THIRTY_SECONDS * z * z);
    const float zz = z*z;
#ifdef DEBUG
    float tmp = z / (1 + dividByPow2f(zz, 2) + dividByPow2f(zz, 5));
//    float tmp = dividByPow2f(z , -5) / (9*z*z + 32);
    float temp = z / (1 + NINE_THIRTY_SECONDS * z * z);

    double delta = fabs((double) tmp) - fabs((double) temp);

    if (delta >= 10e-7) {
        fprintf(stderr, "%f - %f = %f\n", tmp, temp, delta);
    }
    return tmp;
#else
    return z / (1 + dividByPow2f(zz, 2) + dividByPow2f(zz, 5));
#endif
}

float dividByPow2f(float x, int16_t n) {

    union unFloat unf = {x};

    uint32_t sign = unf.i & FLOAT_S_MASK;
    uint32_t e = unf.i & FLOAT_E_MASK;
    uint32_t m = unf.i & FLOAT_M_MASK;
//    uint32_t rb;

    e >>= FLOAT_SHIFT;
    if (e != FLOAT_E_MAX) {
        if (e > 1) {
            e -= n;
        } else {
            if (1 == e) {
                m |= FLOAT_IMPLIED_BIT;
            }
            e = 0;
//            rb = m & 1U;
            m >>= 1;
//            if (rb) {
                // do somehting with sticky bit?
//            }
        }
        e <<= FLOAT_SHIFT;
        unf.i = sign | e | m;
    }

    return unf.f;
}

uint64_t dividByPow2(double *x, int16_t n) {

    union unDouble unf = {*x};

    uint64_t sign = unf.i & DOUBLE_S_MASK;
    uint64_t e = unf.i & DOUBLE_E_MASK;
    uint64_t m = unf.i & DOUBLE_M_MASK;

    e >>= DOUBLE_SHIFT;
    if (e != DOUBLE_E_MAX) {
        if (e > 1) {
            e -= n;
        } else {
            if (1 == e) {
                m |= DOUBLE_IMPLIED_BIT;
            }
            e = 0;
            m >>= 1;
        }
        e <<= DOUBLE_SHIFT;
        unf.i = sign | e | m;
    }

    *x = *(float *) &unf.i;
    return unf.i;
}

void aatan2(const float y, const float x, float *__restrict__ result) {

    if (x == 0.f && y == 0.f) {
        *result = NAN;
    }

    if (0.0f == x) {
        *result =  y > 0.0f ? M_PI_2 : -M_PI_2;
    } else {
        if (ffabsf(x) > ffabsf(y)) {
            *result =  x > 0.f ? aatan(y / x) : y >= 0.f ? aatan(y / x) + M_PI : aatan(y / x) - M_PI;
        } else {
            *result =  y > 0.f ? -aatan(x / y) + M_PI_2 : -aatan(x / y) - M_PI_2;
        }
    }
}

float aatanTwo(float y, float x) {

    float t0, t1, t2, t3, t4;

    t3 = ffabsf(x);
    t1 = ffabsf(y);

//    t0 = max(t3, t1);
//    t1 = min(t3, t1);

    if (t3 > t1) {
        t0 = t3;
        t1 = t1;
    } else {
        t0 = t1;
        t1 = t3;
    }

    t3 = 1.f / t0;
    t3 = t1 * t3;

    t4 = t3 * t3;

    t0 =         - 0.013480470F;
    t0 = t0 * t4 + 0.057477314F;
    t0 = t0 * t4 - 0.121239071F;
    t0 = t0 * t4 + 0.195635925F;
    t0 = t0 * t4 - 0.332994597F;
    t0 = t0 * t4 + 0.999995630F;

    t3 = t0 * t3;

    t3 = (ffabsf(y) > ffabsf(x)) ? 1.570796327F - t3 : t3;
    t3 = (x < 0) ?  3.141592654F - t3 : t3;
    t3 = (y < 0) ? -t3 : t3;

    return t3;
}

void flipAbsMaxMin(float *g, float *f) {
    float x = fabsf(*g);
    float y = fabsf(*f);
    float temp;

    if (x <= y) {
        temp = x;
        x = y;
        y = temp;
//        *((uint32_t *) &x) ^= *((uint32_t *) &y);
//        *((uint32_t *) &y) ^= *((uint32_t *) &x);
//        *((uint32_t *) &x) ^= *((uint32_t *) &y);
    }

    *g = x;
    *f = y;
}
