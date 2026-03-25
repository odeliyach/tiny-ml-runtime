#include "unity.h"
#include <math.h>
#include <setjmp.h>

static jmp_buf UnityJumpBuffer;

static int UnityTestsRun    = 0;
static int UnityTestsFailed = 0;

void UnityBegin(const char *file) {
    (void)file;
    UnityTestsRun = 0;
    UnityTestsFailed = 0;
    printf("Unity test run start: %s\n", file);
}

int UnityEnd(void) {
    printf("Unity results: %d run, %d failed\n", UnityTestsRun, UnityTestsFailed);
    return UnityTestsFailed;
}

void UnityAssert(int condition, int line, const char *msg) {
    if (!condition) {
        UnityTestsFailed++;
        printf("FAIL: line %d: %s\n", line, msg);
        longjmp(UnityJumpBuffer, 1);
    }
}

void UnityAssertEqualInt(int expected, int actual, int line, const char *msg) {
    UnityAssert(expected == actual, line, msg);
}

void UnityAssertFloatWithin(float delta, float expected, float actual, int line, const char *msg) {
    float diff = fabsf(expected - actual);
    if (!isfinite(delta) || !isfinite(diff)) {
        UnityAssert(0, line, "non-finite float comparison");
    }
    UnityAssert(diff <= delta, line, msg);
}

void UnityDefaultTestRun(UnityTestFunction func, const char *name, int line) {
    printf("RUN_TEST(%s)\n", name);
    if (setjmp(UnityJumpBuffer) == 0) {
        UnityTestsRun++;
        func();
    }
    (void)line;
}
