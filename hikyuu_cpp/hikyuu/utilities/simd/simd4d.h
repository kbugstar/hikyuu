/*
 *  Copyright (c) 2019 hikyuu.org
 *
 *  Created on: 2022-02-07
 *      Author: fasiondog
 */

#pragma once

#include "simd_check.h"

namespace hku {

#ifdef HKU_HAVE_AVX
class simd4d {
public:
    typedef double type;

    inline simd4d() {}
    inline simd4d(double f) {
        x = _mm256_set1_pd(f);
    }
    inline simd4d(double r0, double r1, double r2, double r3) {
        x = _mm256_setr_pd(r0, r1, r2, r3);
    }

    inline simd4d(const __m256d& val) : x(val) {}
    inline simd4d& operator=(const __m256d& val) {
        x = val;
        return *this;
    }
    inline operator __m256d() const {
        return x;
    }

    inline void load_aligned(const type* ptr) {
        x = _mm256_load_pd(ptr);
    }
    inline void store_aligned(type* ptr) const {
        _mm256_store_pd(ptr, x);
    }
    inline void load(const type* ptr) {
        x = _mm256_loadu_pd(ptr);
    }
    inline void store(type* ptr) const {
        _mm256_storeu_pd(ptr, x);
    }

    inline simd4d& operator=(const double& val) {
        x = simd4d(val);
        return *this;
    }

    inline size_t size() const {
        return 4;
    }
    inline double operator[](size_t idx) const {
        double temp[4];
        store(temp);
        return temp[idx];
    }

private:
    __m256d x;
};

class simd4d_bool {
public:
    typedef double type;

    inline simd4d_bool() {}
    inline simd4d_bool(const __m256d& val) : x(val) {}
    // inline simd4d_bool(const simd4d_bool& low, const simd4d_bool& high) {
    //     x = _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
    // }

    inline simd4d_bool& operator=(const __m256d& val) {
        x = val;
        return *this;
    }

    inline operator __m256d() const {
        return x;
    }

private:
    __m256d x;
};

#else
class simd4d {
public:
    typedef float type;

    inline simd4d() {}
    inline simd4d(const simd4d& low_, const simd4d& high_) : _low(low_), _high(high_) {}
    inline simd4d(float f) : _low(f), _high(f) {}
    inline simd4d(float r0, float r1, float r2, float r3, float r4, float r5, float r6, float r7)
    : _low(r0, r1, r2, r3), _high(r4, r5, r6, r7) {}
    inline simd4d(const simd8i& val) : _low(val.low()), _high(val.high()) {}

    // truncate to 32bit integers
    inline operator simd8i::rawarray() const {
        simd8i::rawarray temp;
        temp.low = simd4i(_low);
        temp.high = simd4i(_high);
        return temp;
    }

    inline void load_aligned(const type* ptr) {
        _low.load_aligned(ptr);
        _high.load_aligned(ptr + 4);
    }
    inline void store_aligned(type* ptr) const {
        _low.store_aligned(ptr);
        _high.store_aligned(ptr + 4);
    }
    inline void load(const type* ptr) {
        _low.load(ptr);
        _high.load(ptr + 4);
    }
    inline void store(type* ptr) const {
        _low.store(ptr);
        _high.store(ptr + 4);
    }

    inline unsigned int size() const {
        return 8;
    }
    inline float operator[](unsigned int idx) const {
        if (idx < 4)
            return _low[idx];
        else
            return _high[idx - 4];
    }

    inline const simd4d& low() const {
        return _low;
    }
    inline const simd4d& high() const {
        return _high;
    }

private:
    simd4d _low, _high;
};

class simd4d_bool {
public:
    typedef float type;

    inline simd4d_bool() {}
    inline simd4d_bool(const simd4d_bool& low_, const simd4d_bool& high_)
    : _low(low_), _high(high_) {}

    inline const simd4d_bool& low() const {
        return _low;
    }
    inline const simd4d_bool& high() const {
        return _high;
    }

private:
    simd4d_bool _low, _high;
};
#endif

// ----------------------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& out, const simd4d& item) {
    double temp[4];
    item.store(temp);
    out << "(" << temp[0] << ", " << temp[1] << ", " << temp[2] << ", " << temp[3] << ")";
    return out;
}

// ----------------------------------------------------------------------------------------

inline simd4d operator+(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_add_pd(lhs, rhs);
#else
    return simd4d(lhs.low() + rhs.low(), lhs.high() + rhs.high());
#endif
}
inline simd4d& operator+=(simd4d& lhs, const simd4d& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d operator-(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_sub_pd(lhs, rhs);
#else
    return simd4d(lhs.low() - rhs.low(), lhs.high() - rhs.high());
#endif
}
inline simd4d& operator-=(simd4d& lhs, const simd4d& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d operator*(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_mul_pd(lhs, rhs);
#else
    return simd4d(lhs.low() * rhs.low(), lhs.high() * rhs.high());
#endif
}
inline simd4d& operator*=(simd4d& lhs, const simd4d& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d operator/(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_div_pd(lhs, rhs);
#else
    return simd4d(lhs.low() / rhs.low(), lhs.high() / rhs.high());
#endif
}

inline simd4d& operator/=(simd4d& lhs, const simd4d& rhs) {
    lhs = lhs / rhs;
    return lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator==(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_cmp_pd(lhs, rhs, 0);
#else
    return simd4d_bool(lhs.low() == rhs.low(), lhs.high() == rhs.high());
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator!=(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_cmp_pd(lhs, rhs, 4);
#else
    return simd4d_bool(lhs.low() != rhs.low(), lhs.high() != rhs.high());
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator<(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_cmp_pd(lhs, rhs, 1);
#else
    return simd4d_bool(lhs.low() < rhs.low(), lhs.high() < rhs.high());
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator>(const simd4d& lhs, const simd4d& rhs) {
    return rhs < lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator<=(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_cmp_pd(lhs, rhs, 2);
#else
    return simd4d_bool(lhs.low() <= rhs.low(), lhs.high() <= rhs.high());
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d_bool operator>=(const simd4d& lhs, const simd4d& rhs) {
    return rhs <= lhs;
}

// ----------------------------------------------------------------------------------------

inline simd4d min(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_min_pd(lhs, rhs);
#else
    return simd4d(min(lhs.low(), rhs.low()), min(lhs.high(), rhs.high()));
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d max(const simd4d& lhs, const simd4d& rhs) {
#ifdef HKU_HAVE_AVX
    return _mm256_max_pd(lhs, rhs);
#else
    return simd4d(max(lhs.low(), rhs.low()), max(lhs.high(), rhs.high()));
#endif
}

// ----------------------------------------------------------------------------------------

// inline simd4d reciprocal(const simd4d& item) {
// #ifdef HKU_HAVE_AVX
//     return _mm256_rcp_ps(item);
// #else
//     return simd4d(reciprocal(item.low()), reciprocal(item.high()));
// #endif
// }

// ----------------------------------------------------------------------------------------

inline simd4d sqrt(const simd4d& item) {
#ifdef HKU_HAVE_AVX
    return _mm256_sqrt_pd(item);
#else
    return simd4d(reciprocal_sqrt(item.low()), reciprocal_sqrt(item.high()));
#endif
}

// ----------------------------------------------------------------------------------------

inline double sum(const simd4d& item) {
#ifdef HKU_HAVE_AVX
    simd4d temp = _mm256_hadd_pd(item, item);
    return temp[0] + temp[3];
#else
    return sum(item[0] + item[1] + item[2] + item[3]);
#endif
}

// ----------------------------------------------------------------------------------------

inline double dot(const simd4d& lhs, const simd4d& rhs) {
    return sum(lhs * rhs);
}

// ----------------------------------------------------------------------------------------

inline simd4d ceil(const simd4d& item) {
#ifdef HKU_HAVE_AVX
    return _mm256_ceil_pd(item);
#else
    return simd4d(ceil(item.low()), ceil(item.high()));
#endif
}

// ----------------------------------------------------------------------------------------

inline simd4d floor(const simd4d& item) {
#ifdef HKU_HAVE_AVX
    return _mm256_floor_pd(item);
#else
    return simd4d(floor(item.low()), floor(item.high()));
#endif
}

// ----------------------------------------------------------------------------------------

// perform cmp ? a : b
inline simd4d select(const simd4d_bool& cmp, const simd4d& a, const simd4d& b) {
#ifdef HKU_HAVE_AVX
    return _mm256_blendv_pd(b, a, cmp);
#else
    return simd4d(select(cmp.low(), a.low(), b.low()), select(cmp.high(), a.high(), b.high()));
#endif
}

// ----------------------------------------------------------------------------------------

}  // namespace hku
