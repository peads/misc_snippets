//
// Created by Patrick Eads on 2/15/23.
//
#include <stdint.h>
#define TIMING_RUNS 2
#include "timed_functions.h"

char *runNames[TIMING_RUNS] = {"bsr :: ", "bsrLzcnt :: " };

extern int bsr(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsr: "
#else
"bsr: "
#endif
    "movq $-1023, %rdx\n\t" // store 0 to compare against ZF later
    "bsr %rdi, %rax\n\t"
    "cmovzq %rdx, %rax\n\t"
    "ret"
);

extern int lzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_lzcnt: "
#else
"lzcnt: "
#endif
    "movq $-1023, %rdx\n\t" // store -1023 to return if CF set later
    "lzcnt %rdi, %rax\n\t"
    "cmovcq %rdx, %rax\n\t"
    "ret"
);

extern int bsrLzcnt(uint64_t i);
__asm__ (
#ifdef __APPLE_CC__
"_bsrLzcnt: "
#else
"bsrLzcnt: "
#endif
    "lzcnt %rdi, %rdx\n\t"
    "jc zero\n\t" // jump to return -1023 if input was 0
    "movq $63, %rax\n\t"
    "subq %rdx, %rax\n\t"
    "ret\n\t"
"zero:\n\t"
    "movq $-1023, %rax\n\t"
    "ret"
);

//void runTest(void *arg, int *result) {
//
//    struct bitArgs *bargs = arg;
//
//    *result = bargs->fun(bargs->i);
//}
//#define DEBUG
//void testIteration(struct bitArgs *args, void *results, int runIndex) {
//
//    int result;
//    int correct = bitScanReverse(args->i);
//    int delta;
//
//    timeFun((timedFun) runTest, args, results, runIndex);
//
//    result = ((int *) results)[runIndex];
//    delta = abs(abs(correct) - abs(result));
//    int isWrong = delta != 0;
//
//#ifdef DEBUG
//    if (isWrong) {
//        printf("%-25s Expected: %llu Got: %d\n",
//               runNames[runIndex], args->i, result);
//    }
//#endif
//    assert(!isWrong);
//}
int main(void) {

    uint64_t i;

    for (i = 0L; i < 100000000L; ++i) {
        int correct = bitScanReverse(i);
        int result = bsr(i);
        struct timespec tstart, tend;

        clock_gettime(CLOCK_MONOTONIC, &tstart);

        assert(result == correct);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(0, &tstart, &tend);

        clock_gettime(CLOCK_MONOTONIC, &tstart);

        result = bsrLzcnt(i);

        clock_gettime(CLOCK_MONOTONIC, &tend);
        findDeltaTime(1, &tstart, &tend);

        assert(result == correct);
    }

    printTimedRuns(runNames, TIMING_RUNS);

    return 0;
}
