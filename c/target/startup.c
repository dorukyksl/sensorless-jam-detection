/* Bare-metal start-up for the instruction-count benchmark. No libc. */
#include <stddef.h>
#include <stdint.h>

#include "semihost.h"

extern uint32_t _etext, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);
void Reset_Handler(void);

static void Default_Handler(void)
{
    semihost_puts("fault\n");
    semihost_exit(1);
}

__attribute__((section(".vectors"), used))
static void (*const vectors[16])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, 0, 0, 0, 0,
    Default_Handler, Default_Handler, 0,
    Default_Handler, Default_Handler,
};

/* the compiler may emit calls to these for struct copies and clears */
void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;
    while (n-- > 0u) {
        *d++ = *s++;
    }
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    uint8_t *d = dst;
    while (n-- > 0u) {
        *d++ = (uint8_t)c;
    }
    return dst;
}

void Reset_Handler(void)
{
    uint32_t *src = &_etext;
    for (uint32_t *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0u;
    }
    /* full access to CP10 and CP11 (the FPU) */
    *(volatile uint32_t *)0xE000ED88u |= (0xFu << 20);
    __asm__ volatile("dsb\n isb" ::: "memory");
    semihost_exit(main());
}
