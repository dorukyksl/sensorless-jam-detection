/* Minimal test harness: each check prints on failure and counts. */
#ifndef JD_TEST_H
#define JD_TEST_H

#include <stdio.h>

static int jd_test_failures = 0;
static int jd_test_checks = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        jd_test_checks++;                                                  \
        if (!(cond)) {                                                     \
            jd_test_failures++;                                            \
            printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);         \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
        }                                                                  \
    } while (0)

#define RUN(fn)                                                            \
    do {                                                                   \
        const int before = jd_test_failures;                               \
        fn();                                                              \
        printf("%-44s %s\n", #fn, (jd_test_failures == before) ? "ok" : "FAILED"); \
    } while (0)

#define TEST_EXIT()                                                        \
    (printf("%d checks, %d failures\n", jd_test_checks, jd_test_failures), \
     (jd_test_failures == 0) ? 0 : 1)

#endif
