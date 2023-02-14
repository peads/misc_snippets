#include <stdint.h>
#include <stdio.h>

//static void swap1(double *x, double *y) {
//    uint64_t **i = (uint64_t**)&x;
//    uint64_t **j = (uint64_t**)&y;
//
//    **i ^= **j;
//    **j ^= **i;
//    **i ^= **j;
//}

//static void swap(double *x, double *y) {
//
//    __asm__ (
//        "xorq   %0, %1\n\t"
//        "xorq   %1, %0\n\t"
//        "xorq   %0, %1\n\t"
//        "movq   (%0), %0\n\t"
//        "movq   (%1), %1\n\t"
//        : "=r" (*x), "=r" (*y) : "0" (x), "1" (y)
//    );
//}
static void swap1(void *x, void *y) {
    uintptr_t **i = (uintptr_t**)&x;
    uintptr_t **j = (uintptr_t**)&y;

    **i ^= **j;
    **j ^= **i;
    **i ^= **j;
}

extern void swap2(void *x, void *y);
__asm__(
"swap2: "
    "movq (%rsi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "xorq (%rdi), %rax\n\t"
    "xorq %rax, (%rdi)\n\t"
    "movq %rax, (%rsi)\n\t"
    "ret"
);


static void swap(void *x, void *y) {
    __asm__ (
        "movq (%%rsi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "xorq (%%rdi), %%rax\n\t"
        "xorq %%rax, (%%rdi)\n\t"
        "movq %%rax, (%%rsi)\n\t"
        ://:::"rax"
    );
}

int main(void) {
    double x = 1.0;
    double y = 2.0;

    printf("%f %f\n", x, y);

    swap1(&x, &y);

    printf("%f %f\n", x, y);

    swap(&x, &y);

    printf("%f %f\n", x, y);
    swap2(&x, &y);

    printf("%f %f\n", x, y);
    return 0;
}

