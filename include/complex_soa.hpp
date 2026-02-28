#pragma once

#include <vector>
#include <cstddef>
#include <immintrin.h>

// Structure of Arrays (SoA) layout for Complex numbers
struct ComplexVectorSoA {
    std::vector<double> real;
    std::vector<double> imag;

    explicit ComplexVectorSoA(size_t size) : real(size, 0.0), imag(size, 0.0) {}

    size_t size() const {
        return real.size();
    }

    void resize(size_t new_size) {
        real.resize(new_size);
        imag.resize(new_size);
    }
};

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
        __m256d a_real = _mm256_loadu_pd(&a.real[i]);
        __m256d a_imag = _mm256_loadu_pd(&a.imag[i]);
        __m256d b_real = _mm256_loadu_pd(&b.real[i]);
        __m256d b_imag = _mm256_loadu_pd(&b.imag[i]);

        __m256d rr1 = _mm256_mul_pd(a_real, b_real);
        __m256d rr2 = _mm256_mul_pd(a_imag, b_imag);
        __m256d out_real = _mm256_sub_pd(rr1, rr2);

        __m256d ii1 = _mm256_mul_pd(a_real, b_imag);
        __m256d ii2 = _mm256_mul_pd(a_imag, b_real);
        __m256d out_imag = _mm256_add_pd(ii1, ii2);

        _mm256_storeu_pd(&result.real[i], out_real);
        _mm256_storeu_pd(&result.imag[i], out_imag);
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
        __m256d x_r = _mm256_loadu_pd(&x.real[i]);
        __m256d x_i = _mm256_loadu_pd(&x.imag[i]);
        __m256d y_r = _mm256_loadu_pd(&y.real[i]);
        __m256d y_i = _mm256_loadu_pd(&y.imag[i]);

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

        _mm256_storeu_pd(&y.real[i], y_r);
        _mm256_storeu_pd(&y.imag[i], y_i);
    }

    for (; i < n; ++i) {
        double xr = x.real[i];
        double xi = x.imag[i];
        
        y.real[i] += alpha_r * xr - alpha_i * xi;
        y.imag[i] += alpha_r * xi + alpha_i * xr;
    }
}
