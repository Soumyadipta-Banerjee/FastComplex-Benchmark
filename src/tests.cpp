#include "complex_soa.hpp"
#include "linear_algebra_soa.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

bool near(double a, double b) {
    return std::abs(a - b) < 1e-12;
}

int main() {
    const size_t n = 17; // Use a size that is not a multiple of 4 to test tail handling
    ComplexVectorSoA a(n), b(n), res_scalar(n), res_vector(n);

    for (size_t i = 0; i < n; ++i) {
        a.real[i] = i + 1.0;
        a.imag[i] = (i + 1.0) * 0.5;
        b.real[i] = (i + 1.0) * 2.0;
        b.imag[i] = (i + 1.0) * 1.5;
    }

    // Test Multiplication
    MultiplySoA(a, b, res_scalar);
    MultiplySoA_AVX2(a, b, res_vector);
    for (size_t i = 0; i < n; ++i) {
        assert(near(res_scalar.real[i], res_vector.real[i]));
        assert(near(res_scalar.imag[i], res_vector.imag[i]));
    }
    std::cout << "Multiplication: PASSED\n";

    // Test DAXPY
    ComplexVectorSoA y_scalar = b;
    ComplexVectorSoA y_vector = b;
    double alpha_r = 2.5, alpha_i = -1.5;
    DaxpySoA(a, y_scalar, alpha_r, alpha_i);
    DaxpySoA_AVX2(a, y_vector, alpha_r, alpha_i);
    for (size_t i = 0; i < n; ++i) {
        assert(near(y_scalar.real[i], y_vector.real[i]));
        assert(near(y_scalar.imag[i], y_vector.imag[i]));
    }
    std::cout << "DAXPY: PASSED\n";

    // Test Division
    DivideSoA(a, b, res_scalar);
    DivideSoA_AVX2(a, b, res_vector);
    for (size_t i = 0; i < n; ++i) {
        assert(near(res_scalar.real[i], res_vector.real[i]));
        assert(near(res_scalar.imag[i], res_vector.imag[i]));
    }
    std::cout << "Division: PASSED\n";

    // Test Dot Product
    double dr_s = 0, di_s = 0, dr_v, di_v;
    for (size_t i = 0; i < n; ++i) {
        dr_s += a.real[i] * b.real[i] + a.imag[i] * b.imag[i];
        di_s += a.imag[i] * b.real[i] - a.real[i] * b.imag[i];
    }
    DotProductSoA_AVX2(a, b, dr_v, di_v);
    assert(near(dr_s, dr_v));
    assert(near(di_s, di_v));
    std::cout << "Dot Product: PASSED\n";

    // Test GEMV
    MatrixSoA mat(4, n);
    for (size_t i = 0; i < 4 * n; ++i) {
        mat.real[i] = i * 0.1;
        mat.imag[i] = i * 0.2;
    }
    ComplexVectorSoA gemv_res_v(4), gemv_res_s(4);
    GemvSoA_AVX2(mat, a, gemv_res_v);
    
    // Scalar implementation of GEMV for verification
    for (size_t i = 0; i < 4; ++i) {
        double r = 0, im = 0;
        for (size_t j = 0; j < n; ++j) {
            double ar = mat.real[i * n + j];
            double ai = mat.imag[i * n + j];
            double xr = a.real[j];
            double xi = a.imag[j];
            r += ar * xr - ai * xi;
            im += ar * xi + ai * xr;
        }
        gemv_res_s.real[i] = r;
        gemv_res_s.imag[i] = im;
    }

    for (size_t i = 0; i < 4; ++i) {
        assert(near(gemv_res_s.real[i], gemv_res_v.real[i]));
        assert(near(gemv_res_s.imag[i], gemv_res_v.imag[i]));
    }
    std::cout << "GEMV: PASSED\n";

    std::cout << "\nAll sanity checks passed!\n";
    return 0;
}
