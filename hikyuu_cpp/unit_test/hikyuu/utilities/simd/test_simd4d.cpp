/*
 *  Copyright (c) 2019 hikyuu.org
 *
 *  Created on: 2022-02-07
 *      Author: fasiondog
 */

#include "doctest/doctest.h"
#include <hikyuu/utilities/simd/simd4d.h>

using namespace hku;

/**
 * @defgroup test_hikyuu_Parameter test_hikyuu_Parameter
 * @ingroup test_hikyuu_utilities
 * @{
 */

/** @par 检测点 */
TEST_CASE("test_simd4d") {
    double a[4] = {1, 2, 3, 5};
    simd4d x;
    x.load(a);
    std::cout << "x: " << sum(x) << std::endl;

    simd4d temp = _mm256_hadd_pd(x, x);
    simd4d temp2 = _mm256_hadd_pd(temp, temp);
    std::cout << "temp: " << temp << std::endl;
    std::cout << "temp2: " << temp2 << std::endl;
    std::cout << "min: " << min(x, simd4d(2, 3, 4, 5)) << std::endl;
    std::cout << x + simd4d(2, 3, 4, 5) << std::endl;
}

/** @} */