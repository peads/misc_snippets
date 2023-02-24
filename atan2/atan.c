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

#define TIMING_RUNS 2
#define DEBUG
#include "timed_functions.h"

#ifdef DEBUG
    #define PRINTF printf
#else
    #define PRINTF //
#endif

#define MAX_ERROR 10e-7
#define MIN -20
#define MAX 20
#define STEP 1
//#define DEBUG

extern float  ffabsf(float f );
extern float  fsqrtf(float x );

/**
 * takes four float representing the complex numbers (ar + iaj) * (br + ibj),
 * s.t. z = {ar, aj, br, bj}
 **/
extern float argz2(float ar ,
                    float aj ,
                    float br ,
                    float bj );
__asm__(
#ifdef __clang__
"_argz2: "
#else
"argz2: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"
    "and $-16, %rsp\n\t"

//    "vshufps $0x0, %xmm1, %xmm0, %xmm0\n\t" // ar, ar, aj, aj
//    "vshufps $0x0, %xmm3, %xmm2, %xmm1\n\t" // br, br, bj, bj
//    "vpermilps $0x27, %xmm0, %xmm0\n\t"     // aj, ar, aj, ar -> xmm0
//    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1

//    "vshufps $17, %xmm2, %xmm3, %xmm1\n\t"
//    "vpermilps $221, %xmm1, %xmm0\n\t"

    "vshufps $0x11, %xmm0, %xmm1, %xmm0\n\t"
    "vpermilps $0xDD, %xmm0, %xmm0\n\t"      // aj, aj, ar, ar -> xmm0

    "vshufps $0, %xmm3, %xmm2, %xmm1\n\t"
    "vpermilps $0x87, %xmm1, %xmm1\n\t"     // bj, br, br, bj -> xmm1

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

    // B I G pop and return
    "vmovq (%rsp), %xmm0 \n\t"
    "add $32, %rsp \n\t"
    "ret"


//    "vextractps $0, %xmm0, %rcx\n\t"    // check zr == 0
//    "cmp $0, %rcx\n\t"
//    "vextractps $3, %xmm0, %rcx\n\t"    // check zj == 0
//    "jnz homestretch\n\t"
//
//    "cmp $0, %rcx\n\t"
//    "jz undefined\n\t"                  // return nan
//    "jnz zero\n\t"                      // return 0
//
//"homestretch: "
//    "movq $0x3F800000, %rdx\n\t"
//    "movq $0xBF800000, %rbx\n\t"
//    "shrq $31, %rcx\n\t"
//    "cmp $1, %rcx\n\t"
//    "cmove %rbx, %rdx\n\t"



//    "vmovq %rdx, %xmm2\n\t"
//    "vbroadcastss %xmm2, %xmm2\n\t"
//    "movq $0x3fc90fdb, %rcx\n\t"
//    "vmovq %rcx, %xmm3\n\t"
//    "vbroadcastss %xmm5, %xmm5\n\t"
//    "vmulps %xmm1, %xmm1, %xmm0\n\t"
//    "vmulps %xmm0, %xmm0, %xmm0\n\t"
//    "vmulps %xmm0, %xmm0, %xmm0\n\t"
//    "vmulps %xmm5, %xmm0, %xmm0\n\t"
//    "vaddps %xmm1, %xmm0, %xmm0\n\t"
//    "vsubps %xmm0, %xmm3, %xmm0\n\t"
//    "vmulps %xmm2, %xmm0, %xmm0\n\t"
//    "jmp return\n\t"
//
//"piovertwo: "
//    "movq $0x3fc90fdb, %rcx\n\t"
//    "vmovq %rcx, %xmm0\n\t"
//    "jmp return\n\t"
//
//"zero: "
//    "movq $0, %rcx\n\t"
//    "vmovq %rcx, %xmm0\n\t"
//    "jmp return\n\t"
//
//"undefined: "
//    "pcmpeqd   %xmm0, %xmm0\n\t"
//    "mulsd     %xmm0, %xmm0\n\t"

"return: "
    "movq  %rbp, %rsp\n\t"
    "popq %rbp\n\t"
    "ret\n\t"
);

int main(void) {

    float ar, aj, br, bj;
    float zr, zj, result, true;

    for (ar = MIN; ar < MAX; ar += STEP) {              // ar
        for (aj = MIN; aj < MAX; aj += STEP) {          // aj
            for (br = MIN; br < MAX; br += STEP) {      // br
                for (bj = MIN; bj < MAX; bj += STEP) {  // bj

                    zr = ar * br - aj * bj;
                    zj = ar * bj + aj * br;

                    result = argz2(ar, aj, br, bj);
                    true = atan2f(zr, zj);
                    
                    printf("(%f + %fi) * (%f + %fi) => Arg[(%f + %fi)] = %f, %f\n", ar, aj, br, bj, zr, zj, true, result);

                    if (!isnan(result)){
                        assert(ffabsf(ffabsf(true) - ffabsf(result)) < MAX_ERROR);
                    }
                }
            }
        }
    }

    return 0;
}
