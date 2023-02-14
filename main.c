#include <stdint.h>
#include <stdio.h>

static void swap1(double *x, double *y) {
    uint64_t **i = (uint64_t**)&x;
    uint64_t **j = (uint64_t**)&y;

    **i ^= **j;
    **j ^= **i;
    **i ^= **j;
}

static void swap(double *x, double *y) {

    printf("0x%lX 0x%lX %f %f\n", &x, &y, *x, *y);

    __asm__ __volatile__(
        "xorq   %0, %1\n\t"
        "xorq   %1, %0\n\t"
        "xorq   %0, %1\n\t"
        "movq   (%0), %0\n\t"
        "movq   (%1), %1\n\t"
        : "=r" (*x), "=r" (*y) : "0" (x), "1" (y)
    );

    printf("0x%lX 0x%lX %f %f\n", &x, &y, *x, *y);
}

int main(void) {
    double x = 1.0;
    double y = 2.0;

    printf("%f %f\n", x, y);

    swap1(&x, &y);

    printf("%f %f\n", x, y);

    swap(&x, &y);

    printf("%f %f\n", x, y);
    return 0;
}

