#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "complex_soa.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper for bit-reversal
inline uint32_t reverse_bits(uint32_t n, int bits) {
    uint32_t reversed = 0;
    for (int i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    return reversed;
}

// In-place iterative Radix-2 FFT for AoS (std::complex)
inline void fft_iterative_aos(std::vector<std::complex<double>>& a) {
    int n = a.size();
    if (n <= 1) return;

    int log2n = 0;
    while ((1 << log2n) < n) log2n++;

    for (int i = 0; i < n; ++i) {
        int rev = reverse_bits(i, log2n);
        if (i < rev) {
            std::swap(a[i], a[rev]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// In-place iterative Radix-2 FFT for SoA
inline void fft_iterative_soa(ComplexVectorSoA& a) {
    int n = a.size();
    if (n <= 1) return;

    int log2n = 0;
    while ((1 << log2n) < n) log2n++;

    for (int i = 0; i < n; ++i) {
        int rev = reverse_bits(i, log2n);
        if (i < rev) {
            std::swap(a.real[i], a.real[rev]);
            std::swap(a.imag[i], a.imag[rev]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        double wlen_r = std::cos(angle);
        double wlen_i = std::sin(angle);

        for (int i = 0; i < n; i += len) {
            double w_r = 1.0;
            double w_i = 0.0;
            for (int j = 0; j < len / 2; ++j) {
                int idx1 = i + j;
                int idx2 = i + j + len / 2;

                double u_r = a.real[idx1];
                double u_i = a.imag[idx1];

                double a2_r = a.real[idx2];
                double a2_i = a.imag[idx2];

                double v_r = a2_r * w_r - a2_i * w_i;
                double v_i = a2_r * w_i + a2_i * w_r;

                a.real[idx1] = u_r + v_r;
                a.imag[idx1] = u_i + v_i;

                a.real[idx2] = u_r - v_r;
                a.imag[idx2] = u_i - v_i;

                double next_w_r = w_r * wlen_r - w_i * wlen_i;
                double next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}
