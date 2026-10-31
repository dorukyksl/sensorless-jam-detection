/* Arm semihosting, enough to print and to stop QEMU. */
#ifndef SEMIHOST_H
#define SEMIHOST_H

#define SYS_WRITE0 0x04
#define SYS_EXIT   0x18
#define ADP_STOPPED_APPLICATION_EXIT 0x20026
#define ADP_STOPPED_RUNTIME_ERROR    0x20023

static inline int semihost(int op, const void *arg)
{
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static inline void semihost_puts(const char *s)
{
    (void)semihost(SYS_WRITE0, s);
}

static inline void semihost_exit(int rc)
{
    (void)semihost(SYS_EXIT, (const void *)(rc == 0 ? ADP_STOPPED_APPLICATION_EXIT
                                                    : ADP_STOPPED_RUNTIME_ERROR));
    for (;;) {
    }
}

#endif
