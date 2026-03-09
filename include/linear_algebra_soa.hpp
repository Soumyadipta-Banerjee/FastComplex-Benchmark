#pragma once

#include "complex_soa.hpp"
#include <vector>

struct MatrixSoA {
    std::vector<double, AlignedAllocator<double, 64>> real;
    std::vector<double, AlignedAllocator<double, 64>> imag;
    size_t rows;
    size_t cols;
    size_t stride_cols;

    MatrixSoA(size_t r, size_t c) : rows(r), cols(c) {
        stride_cols = ((c + 3) / 4) * 4;
        real.resize(r * stride_cols, 0.0);
        imag.resize(r * stride_cols, 0.0);
    }
};

// Vectorized Matrix-Vector Multiplication: y = A * x
inline void GemvSoA_AVX2(const MatrixSoA& A, const ComplexVectorSoA& x, ComplexVectorSoA& y) {
    const size_t rows = A.rows;
    const size_t cols = A.cols;

    for (size_t i = 0; i < rows; ++i) {
        double res_r = 0.0;
        double res_i = 0.0;

        size_t row_offset = i * A.stride_cols;
        size_t j = 0;

        __m256d sum_r = _mm256_setzero_pd();
        __m256d sum_i = _mm256_setzero_pd();

        for (; j + 3 < cols; j += 4) {
            __m256d a_r = _mm256_load_pd(&A.real[row_offset + j]);
            __m256d a_i = _mm256_load_pd(&A.imag[row_offset + j]);
            __m256d x_r = _mm256_load_pd(&x.real[j]);
            __m256d x_i = _mm256_load_pd(&x.imag[j]);

            // (a_r + a_i i) * (x_r + x_i i) = (a_r*x_r - a_i*x_i) + (a_r*x_i + a_i*x_r)i
            __m256d r1 = _mm256_mul_pd(a_r, x_r);
            __m256d r2 = _mm256_mul_pd(a_i, x_i);
            sum_r = _mm256_add_pd(sum_r, _mm256_sub_pd(r1, r2));

            __m256d i1 = _mm256_mul_pd(a_r, x_i);
            __m256d i2 = _mm256_mul_pd(a_i, x_r);
            sum_i = _mm256_add_pd(sum_i, _mm256_add_pd(i1, i2));
        }

        // Horizontal sum
        alignas(32) double r_tmp[4], i_tmp[4];
        _mm256_store_pd(r_tmp, sum_r);
        _mm256_store_pd(i_tmp, sum_i);
        
        res_r = r_tmp[0] + r_tmp[1] + r_tmp[2] + r_tmp[3];
        res_i = i_tmp[0] + i_tmp[1] + i_tmp[2] + i_tmp[3];

        // Tail cases
        for (; j < cols; ++j) {
            double ar = A.real[row_offset + j];
            double ai = A.imag[row_offset + j];
            double xr = x.real[j];
            double xi = x.imag[j];

            res_r += ar * xr - ai * xi;
            res_i += ar * xi + ai * xr;
        }

        y.real[i] = res_r;
        y.imag[i] = res_i;
    }
}
