#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>

#include <pmmintrin.h>

#pragma STDC FENV_ACCESS ON

#define FTZ_BIT 15
#define DAZ_BIT 6

#define FTZ_MASK (1 << FTZ_BIT)
#define DAZ_MASK (1 << DAZ_BIT)
#define FTZ_DAZ_MASK (FTZ_MASK | DAZ_MASK)

//warning these macros has to be used in the same scope
#define MXCSR_SET_DAZ_AND_FTZ \
    int oldMXCSR__ = _mm_getcsr(); /*read the old MXCSR setting */  \
    int newMXCSR__ = oldMXCSR__ | FTZ_DAZ_MASK; /* set DAZ and FZ bits */ \
    _mm_setcsr( newMXCSR__ ); /*write the new MXCSR setting to the MXCSR */


#define MXCSR_RESET_DAZ_AND_FTZ \
    /*restore old MXCSR settings to turn denormals back on if they were on*/ \
    _mm_setcsr( oldMXCSR__ );


__attribute__((noinline))
void fp32_denorm_test() {
    volatile float subnormal = 0x1p-127;
    volatile float neg_subnormal = -0x1p-127;
    assert(std::fpclassify(subnormal) == FP_SUBNORMAL);
    assert(std::fpclassify(neg_subnormal) == FP_SUBNORMAL);

    volatile float zero = 0.0f;
    volatile float neg_zero = -0.0f;

    assert(zero + neg_zero == 0.0f);
    assert(neg_zero + zero == 0.0f);
    assert(neg_zero + neg_zero == -0.0f);
    printf("neg_subnormal + neg_subnormal: %a\n", neg_subnormal + neg_subnormal);
    printf("neg_subnormal + neg_zero: %a\n", neg_subnormal + neg_zero);
    printf("sqrtf subnormal: %a\n", sqrtf(subnormal));
    printf("sqrtf neg_subnormal: %a\n", sqrtf(neg_subnormal));
    printf("sqrtf neg_zero: %a\n", sqrtf(neg_zero));
}


__attribute__((noinline))
void test_with_denormals_disabled() {
    int oldMXCSR__ = _mm_getcsr(); /*read the old MXCSR setting */  \
    int newMXCSR__ = oldMXCSR__ | FTZ_DAZ_MASK; /* set DAZ and FZ bits */     \
    _mm_setcsr( newMXCSR__ ); /*write the new MXCSR setting to the MXCSR */
    fp32_denorm_test();
    _mm_setcsr( oldMXCSR__ );
}

__attribute__((noinline))
void test_with_denormals_enabled() {
    int oldMXCSR__ = _mm_getcsr(); /*read the old MXCSR setting */  \
    int newMXCSR__ = oldMXCSR__ &= ~FTZ_DAZ_MASK; /* clear DAZ and FZ bits */     \
    _mm_setcsr( newMXCSR__ ); /*write the new MXCSR setting to the MXCSR */
    fp32_denorm_test();
    _mm_setcsr( oldMXCSR__ );
}

__attribute__((noinline))
void test_daz_only() {
    int oldMXCSR__ = _mm_getcsr(); /*read the old MXCSR setting */  \
    int newMXCSR__ = (oldMXCSR__ &= ~FTZ_DAZ_MASK) | DAZ_MASK;
    _mm_setcsr( newMXCSR__ ); /*write the new MXCSR setting to the MXCSR */
    fp32_denorm_test();
    _mm_setcsr( oldMXCSR__ );
}

__attribute__((noinline))
void test_ftz_only() {
    int oldMXCSR__ = _mm_getcsr(); /*read the old MXCSR setting */  \
    int newMXCSR__ = (oldMXCSR__ &= ~FTZ_DAZ_MASK) | FTZ_MASK;
    _mm_setcsr( newMXCSR__ ); /*write the new MXCSR setting to the MXCSR */
    fp32_denorm_test();
    _mm_setcsr( oldMXCSR__ );
}

int main() {
    volatile float zero = 0.0f;
    volatile float neg_zero = -0.0f;
    volatile float one = 1.0f;
    volatile float neg_one = -1.0f;

    printf("In default FP mode\n");
    fp32_denorm_test();

    printf("\nWith denormals disabled\n");
    test_with_denormals_disabled();

    printf("\nWith denormals enabled\n");
    test_with_denormals_enabled();

    printf("\nWith daz only\n");
    test_daz_only();

    printf("\nWith ftz only\n");
    test_ftz_only();

    return 0;
}
