//
// Created by Patrick Eads on 2/16/23.
//
#include <stdio.h>

#define MUL_PLUS_J_INT( X, J )	\
    tmp = X[J]; \
    X[J] = -X[J+1]; \
    X[J+1] = tmp

#define MUL_MINUS_J_INT( X, J ) \
    tmp = X[J]; \
    X[J] = X[J+1]; \
    X[J+1] = -tmp

extern void swapNegate(int *x, int *y);
__asm__ (
#ifdef __APPLE_CC__
"_swapNegate: "
#else
"swapNegate: "
#endif
    "movl (%rsi), %ebx\n\t"
    "movl (%rdi), %ecx\n\t"
    "movl %ebx, (%rdi)\n\t"
    "negl %ecx\n\t"
    "movl %ecx, (%rsi)\n\t"
    "ret"
);
int main(void) {
    int tmp;
    int ints[4] = {21, 7, 1,2};

    printf("i: %d, j: %d\n", ints[0], ints[1]);
    MUL_MINUS_J_INT(ints, 0);
    printf("i: %d, j: %d\n", ints[0], ints[1]);

    ints[0] = 21;
    ints[1] = 7;
    printf("i: %d, j: %d\n", ints[0], ints[1]);
    swapNegate(& ints[0], & ints[1]);
    printf("i: %d, j: %d\n\n", ints[0], ints[1]);

    ints[0] = 21;
    ints[1] = 7;
    printf("i: %d, j: %d\n", ints[0], ints[1]);
    MUL_PLUS_J_INT( ints, 0 );
    printf("i: %d, j: %d\n", ints[0], ints[1]);

    ints[0] = 21;
    ints[1] = 7;
    printf("i: %d, j: %d\n", ints[0], ints[1]);
    swapNegate(&ints[1], & ints[0]);
    printf("i: %d, j: %d\n\n", ints[0], ints[1]);
}