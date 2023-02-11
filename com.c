#include <stdio.h>
#include <immintrin.h>
#include <time.h>
//#include <emmintrin.h>
#include <math.h>
#include <stdint.h>


#define TIMING_RUNS 4
typedef double (*timedFunD)(int ar, int aj, int br, int bj);

static time_t rollingTimeSums[TIMING_RUNS];
static long runCount = 0;

static void findDeltaTime(int idx, struct timespec tstart, struct timespec tend, char *timediff) {
    time_t deltaTsec = tend.tv_sec - tstart.tv_sec;
    time_t deltaTNanos = tend.tv_nsec - tstart.tv_nsec;

    if (deltaTsec > 0) {
        sprintf(timediff, "%lu.%lu s",deltaTsec, deltaTNanos);
    } else {
        sprintf(timediff, "%lu ns", deltaTNanos);
        rollingTimeSums[idx] += deltaTNanos;
    }
}

double timeFunD(timedFunD fun, int ar, int aj, int br, int bj, int i) {
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    double s = fun(ar,aj,br,bj);

    clock_gettime(CLOCK_MONOTONIC, &tend);
    char *timediff = malloc(256);
    findDeltaTime(i, tstart, tend, timediff);

    free(timediff);

    return s;

}

static double norm(int x, int y) {
    return sqrt(x*x+y*y);
}

static double cosTheta(int ar, int aj, int br, int bj) {

    double anorm = norm(ar, aj);//sqrt(ar*ar+aj*aj);
    double bnorm = norm(br, bj);//sqrt(br*br+bj*bj);
    return (ar*br+aj*bj)/(anorm*bnorm);
}

union vect {
    __m256d vect;
};


static double calcVectPd(int ar, int aj, int br, int bj) {

    union vect u = {ar, aj, ar, br};
    union vect v = {br, bj, bj, aj};

    double zj;
    __m256d z;

    z = _mm256_mul_pd(u.vect, v.vect);
    z = _mm256_addsub_pd(z, _mm256_permute_pd(z, 0b0101));
    zj = z[3];
    z = _mm256_mul_pd(z, z);
    z = _mm256_permute_pd(z, _MM_SHUFFLE(3,2,1,0));
    z = _mm256_add_pd(z, _mm256_permute2f128_pd(z, z, 1));
    z = _mm256_sqrt_pd(z);
    
    return -acos(zj/z[0]);
}

double calcVect(int ar, int aj, int br, int bj) {
    double zr;
    double zj;

    union vect {
        __m128i vect;
    };

    union vect u1 = {ar, ar};
    union vect v1 = {br, bj};
    union vect u0 = {aj, br};
    union vect v0 = {bj, aj};

    v1.vect = _mm_mul_epi32(u1.vect, v1.vect); // => {ar*br, ar*bj}
    v0.vect = _mm_mul_epi32(u0.vect, v0.vect); // => {aj*bj, br*aj}

    u1.vect = _mm_sub_epi64(v1.vect, v0.vect); // => {ar*br - aj*bj, ar*bj - br*aj} *we don't care about the second entry
    u0.vect = _mm_add_epi32(v1.vect, v0.vect); // => {ar*br + aj*bj, ar*bj + br*aj} *we don't care about the first entry
    
    zr = u1.vect[0];
    zj = u0.vect[1];
    
    u0.vect = _mm_mul_epi32(u0.vect, u0.vect);
    u1.vect = _mm_mul_epi32(u1.vect, u1.vect);
    u0.vect = _mm_add_epi32(u0.vect, _mm_shuffle_epi32(u1.vect, _MM_SHUFFLE(1,0,1,0)));

    return -acos(zj/sqrt(u0.vect[1]));
}



static double calcAcos(int ar, int aj, int br, int bj) {

    int vr = ar * br - aj * bj;
    int vj = aj * br + ar * bj;

    double vnorm = norm(vr, vj);

    return -acos(vj/vnorm);
}

static double calcAtan2(int ar, int aj, int br, int bj) {

    int vr = ar * br - aj * bj;
    int vj = aj * br + ar * bj;

    return atan2((double) vr, (double) vj);
}

static void mult(int ar, int aj, int br, int bj/*, int *vr, int *vj*/) {
    ++runCount;

    int j = 0;
    double results[TIMING_RUNS] = {
        timeFunD(calcAtan2,ar,aj,br,bj,0),
        timeFunD(calcAcos,ar,aj,br,bj,1),
        timeFunD(calcVect,ar,aj,br,bj,2),
        timeFunD(calcVectPd,ar,aj,br,bj,3)
    };
    char *runNames[TIMING_RUNS] 
        = {"calcAtan2 :: ", "calcAcos :: ", "calcVect :: ", "caclVectPd :: "};

    printf("(%d + %di) . (%d + %di) = (%d + %di) angle: %f, %f, %f, %f \n", 
        ar, aj, 
        br, bj,
        ar * br - aj * bj,
        aj * br + ar * bj,
        results[0], results[1], results[2], results[3]);

    for (; j < TIMING_RUNS; ++j){
        const char *runName = runNames[j];
        printf("%sAverage time: %ld ns\n", runName, rollingTimeSums[j]/runCount);
    }

   // printf("%f %f %f %f\n", u.vect[0], u.vect[1], u.vect[2], u.vect[3]);
    //printf("%f %f %f %f\n", v.vect[0], v.vect[1], v.vect[2], v.vect[3]);

   // __m256d v = _mm256_mul_pd(u.vect, v.vect);
   // v = _mm256_hadd_pd(v, v);
   // v = _mm256_mul_pd(v, v);
   // v = _mm256_add_pd(v, v);
    //v = _mm256_sqrt_pd(v);


    //__m128d zr = _mm_mul_pd(u.vect, v.vect);
    //__m128d zj = _mm_mul_pd(_mm_shuffle_pd(u.vect, u.vect, _MM_SHUFFLE2(0,1)), v.vect);
}

int main()
{

    int i[4];
    for (i[0] = 0; i[0] < 10; ++i[0]) {
        for (i[1] = 0; i[1] < 10; ++i[1]) {
            for (i[2] = 0; i[2] < 10; ++i[2]) {
                for (i[3] = 0; i[3] < 10; ++i[3]) {
                    mult(i[0],i[1],i[2],i[3]);
                }
            }
        }
    }

    mult(5,6,7,8);
    mult(9,10,11,12);

    return 0;
}
