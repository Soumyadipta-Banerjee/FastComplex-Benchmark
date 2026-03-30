#pragma once

#include <vector>
#include <cstddef>
#include <immintrin.h>
#include "aligned_allocator.hpp"

// Structure of Arrays (SoA) layout for Complex numbers
struct ComplexVectorSoA {
    std::vector<double, AlignedAllocator<double, 64>> real;
    std::vector<double, AlignedAllocator<double, 64>> imag;

    explicit ComplexVectorSoA(size_t size) : real(size, 0.0), imag(size, 0.0) {}

    size_t size() const {
        return real.size();
    }

    void resize(size_t new_size) {
        real.resize(new_size);
        imag.resize(new_size);
    }
};

// --- ADDITION ---

// Raw loop addition (Scalar SoA)
inline void AddSoA(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        result.real[i] = a.real[i] + b.real[i];
        result.imag[i] = a.imag[i] + b.imag[i];
    }
}

// AVX2 vectorized addition (Vector SoA)
inline void AddSoA_AVX2(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d a_real = _mm256_load_pd(&a.real[i]);
        __m256d a_imag = _mm256_load_pd(&a.imag[i]);
        __m256d b_real = _mm256_load_pd(&b.real[i]);
        __m256d b_imag = _mm256_load_pd(&b.imag[i]);

        _mm256_store_pd(&result.real[i], _mm256_add_pd(a_real, b_real));
        _mm256_store_pd(&result.imag[i], _mm256_add_pd(a_imag, b_imag));
    }

    for (; i < n; ++i) {
        result.real[i] = a.real[i] + b.real[i];
        result.imag[i] = a.imag[i] + b.imag[i];
    }
}

// --- MULTIPLICATION ---

// Raw loop multiplication (Scalar SoA)
inline void MultiplySoA(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double ar = a.real[i];
        double ai = a.imag[i];
        double br = b.real[i];
        double bi = b.imag[i];

        result.real[i] = ar * br - ai * bi;
        result.imag[i] = ar * bi + ai * br;
    }
}

// AVX2 vectorized multiplication (Vector SoA)
inline void MultiplySoA_AVX2(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d a_real = _mm256_load_pd(&a.real[i]);
        __m256d a_imag = _mm256_load_pd(&a.imag[i]);
        __m256d b_real = _mm256_load_pd(&b.real[i]);
        __m256d b_imag = _mm256_load_pd(&b.imag[i]);

        __m256d rr1 = _mm256_mul_pd(a_real, b_real);
        __m256d rr2 = _mm256_mul_pd(a_imag, b_imag);
        __m256d out_real = _mm256_sub_pd(rr1, rr2);

        __m256d ii1 = _mm256_mul_pd(a_real, b_imag);
        __m256d ii2 = _mm256_mul_pd(a_imag, b_real);
        __m256d out_imag = _mm256_add_pd(ii1, ii2);

        _mm256_store_pd(&result.real[i], out_real);
        _mm256_store_pd(&result.imag[i], out_imag);
    }

    for (; i < n; ++i) {
        double ar = a.real[i];
        double ai = a.imag[i];
        double br = b.real[i];
        double bi = b.imag[i];

        result.real[i] = ar * br - ai * bi;
        result.imag[i] = ar * bi + ai * br;
    }
}

// --- DAXPY (y = a*x + y) ---

// Raw loop DAXPY (Scalar SoA)
inline void DaxpySoA(const ComplexVectorSoA& x, ComplexVectorSoA& y, double alpha_r, double alpha_i) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double xr = x.real[i];
        double xi = x.imag[i];
        
        y.real[i] += alpha_r * xr - alpha_i * xi;
        y.imag[i] += alpha_r * xi + alpha_i * xr;
    }
}

// AVX2 vectorized DAXPY (Vector SoA)
inline void DaxpySoA_AVX2(const ComplexVectorSoA& x, ComplexVectorSoA& y, double alpha_r, double alpha_i) {
    const size_t n = x.size();
    size_t i = 0;

    __m256d v_alpha_r = _mm256_set1_pd(alpha_r);
    __m256d v_alpha_i = _mm256_set1_pd(alpha_i);

    for (; i + 3 < n; i += 4) {
        __m256d x_r = _mm256_load_pd(&x.real[i]);
        __m256d x_i = _mm256_load_pd(&x.imag[i]);
        __m256d y_r = _mm256_load_pd(&y.real[i]);
        __m256d y_i = _mm256_load_pd(&y.imag[i]);

        // a_r * x_r - a_i * x_i
        __m256d rr1 = _mm256_mul_pd(v_alpha_r, x_r);
        __m256d rr2 = _mm256_mul_pd(v_alpha_i, x_i);
        __m256d px_r = _mm256_sub_pd(rr1, rr2);
        
        // a_r * x_i + a_i * x_r
        __m256d ii1 = _mm256_mul_pd(v_alpha_r, x_i);
        __m256d ii2 = _mm256_mul_pd(v_alpha_i, x_r);
        __m256d px_i = _mm256_add_pd(ii1, ii2);

        // y += (ax)
        y_r = _mm256_add_pd(y_r, px_r);
        y_i = _mm256_add_pd(y_i, px_i);

        _mm256_store_pd(&y.real[i], y_r);
        _mm256_store_pd(&y.imag[i], y_i);
    }

    for (; i < n; ++i) {
        double xr = x.real[i];
        double xi = x.imag[i];
        
        y.real[i] += alpha_r * xr - alpha_i * xi;
        y.imag[i] += alpha_r * xi + alpha_i * xr;
    }
}

// --- DIVISION ---

// Raw loop division (Scalar SoA)
inline void DivideSoA(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double ar = a.real[i];
        double ai = a.imag[i];
        double br = b.real[i];
        double bi = b.imag[i];
        double denom = br * br + bi * bi;

        result.real[i] = (ar * br + ai * bi) / denom;
        result.imag[i] = (ai * br - ar * bi) / denom;
    }
}

// AVX2 vectorized division (Vector SoA)
inline void DivideSoA_AVX2(const ComplexVectorSoA& a, const ComplexVectorSoA& b, ComplexVectorSoA& result) {
    const size_t n = a.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d a_r = _mm256_load_pd(&a.real[i]);
        __m256d a_i = _mm256_load_pd(&a.imag[i]);
        __m256d b_r = _mm256_load_pd(&b.real[i]);
        __m256d b_i = _mm256_load_pd(&b.imag[i]);

        // denom = b_r^2 + b_i^2
        __m256d br2 = _mm256_mul_pd(b_r, b_r);
        __m256d bi2 = _mm256_mul_pd(b_i, b_i);
        __m256d denom = _mm256_add_pd(br2, bi2);

        // real = (a_r*b_r + a_i*b_i) / denom
        __m256d r1 = _mm256_mul_pd(a_r, b_r);
        __m256d r2 = _mm256_mul_pd(a_i, b_i);
        __m256d res_r = _mm256_div_pd(_mm256_add_pd(r1, r2), denom);

        // imag = (a_i*b_r - a_r*b_i) / denom
        __m256d i1 = _mm256_mul_pd(a_i, b_r);
        __m256d i2 = _mm256_mul_pd(a_r, b_i);
        __m256d res_i = _mm256_div_pd(_mm256_sub_pd(i1, i2), denom);

        _mm256_store_pd(&result.real[i], res_r);
        _mm256_store_pd(&result.imag[i], res_i);
    }

    for (; i < n; ++i) {
        double ar = a.real[i];
        double ai = a.imag[i];
        double br = b.real[i];
        double bi = b.imag[i];
        double denom = br * br + bi * bi;
        result.real[i] = (ar * br + ai * bi) / denom;
        result.imag[i] = (ai * br - ar * bi) / denom;
    }
}

// --- DOT PRODUCT ---

// Vectorized Dot Product using AVX2
inline void DotProductSoA_AVX2(const ComplexVectorSoA& a, const ComplexVectorSoA& b, double& res_r, double& res_i) {
    const size_t n = a.size();
    size_t i = 0;

    __m256d sum_r = _mm256_setzero_pd();
    __m256d sum_i = _mm256_setzero_pd();

    for (; i + 3 < n; i += 4) {
        __m256d a_r = _mm256_load_pd(&a.real[i]);
        __m256d a_i = _mm256_load_pd(&a.imag[i]);
        __m256d b_r = _mm256_load_pd(&b.real[i]);
        __m256d b_i = _mm256_load_pd(&b.imag[i]);

        // (a_r + a_i i) * (b_r - b_i i) = (a_r*b_r + a_i*b_i) + (a_i*b_r - a_r*b_i)i
        sum_r = _mm256_add_pd(sum_r, _mm256_add_pd(_mm256_mul_pd(a_r, b_r), _mm256_mul_pd(a_i, b_i)));
        sum_i = _mm256_add_pd(sum_i, _mm256_sub_pd(_mm256_mul_pd(a_i, b_r), _mm256_mul_pd(a_r, b_i)));
    }

    // Horizontal sum
    alignas(32) double r_tmp[4], i_tmp[4];
    _mm256_store_pd(r_tmp, sum_r);
    _mm256_store_pd(i_tmp, sum_i);
    
    res_r = r_tmp[0] + r_tmp[1] + r_tmp[2] + r_tmp[3];
    res_i = i_tmp[0] + i_tmp[1] + i_tmp[2] + i_tmp[3];

    for (; i < n; ++i) {
        res_r += a.real[i] * b.real[i] + a.imag[i] * b.imag[i];
        res_i += a.imag[i] * b.real[i] - a.real[i] * b.imag[i];
    }
}

// --- HELPERS ---

#include <complex>

inline ComplexVectorSoA AoS_to_SoA(const std::vector<std::complex<double>>& aos) {
    ComplexVectorSoA soa(aos.size());
    for (size_t i = 0; i < aos.size(); ++i) {
        soa.real[i] = aos[i].real();
        soa.imag[i] = aos[i].imag();
    }
    return soa;
}

inline std::vector<std::complex<double>> SoA_to_AoS(const ComplexVectorSoA& soa) {
    std::vector<std::complex<double>> aos(soa.size());
    for (size_t i = 0; i < soa.size(); ++i) {
        aos[i] = {soa.real[i], soa.imag[i]};
    }
    return aos;
}
