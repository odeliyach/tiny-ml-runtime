#ifndef UNITY_FRAMEWORK_H
#define UNITY_FRAMEWORK_H

#include <stdio.h>

typedef void (*UnityTestFunction)(void);

void UnityBegin(const char *file);
int  UnityEnd(void);
void UnityDefaultTestRun(UnityTestFunction func, const char *name, int line);
void UnityAssert(int condition, int line, const char *msg);
void UnityAssertEqualInt(int expected, int actual, int line, const char *msg);
void UnityAssertFloatWithin(float delta, float expected, float actual, int line, const char *msg);

#define UNITY_FLOAT_TOLERANCE 1e-6f

#define UNITY_BEGIN()           UnityBegin(__FILE__)
#define UNITY_END()             UnityEnd()
#define RUN_TEST(func)          UnityDefaultTestRun(func, #func, __LINE__)

#define TEST_ASSERT(condition)              UnityAssert((condition), __LINE__, #condition)
#define TEST_ASSERT_TRUE(condition)         TEST_ASSERT((condition))
#define TEST_ASSERT_FALSE(condition)        TEST_ASSERT(!(condition))
#define TEST_ASSERT_NOT_NULL(ptr)           UnityAssert((ptr) != NULL, __LINE__, #ptr " != NULL")
#define TEST_ASSERT_EQUAL_INT(exp, act)     UnityAssertEqualInt((exp), (act), __LINE__, #act)
#define TEST_ASSERT_FLOAT_WITHIN(d, e, a)   UnityAssertFloatWithin((d), (e), (a), __LINE__, #a)
#define TEST_ASSERT_EQUAL_FLOAT(exp, act)   UnityAssertFloatWithin(UNITY_FLOAT_TOLERANCE, (exp), (act), __LINE__, #act)

#endif
