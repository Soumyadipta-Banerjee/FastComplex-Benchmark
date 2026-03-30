#include <benchmark/benchmark.h>
#include <vector>
#include <complex>
#include <random>

#include "complex_soa.hpp"
#include "fft_stress_test.hpp"
#include "linear_algebra_soa.hpp"

struct BenchData {
    std::vector<std::complex<double>> aos_a;
    std::vector<std::complex<double>> aos_b;
    std::vector<std::complex<double>> aos_res;

    ComplexVectorSoA soa_a;
    ComplexVectorSoA soa_b;
    ComplexVectorSoA soa_res;

    MatrixSoA mat_soa;
    std::vector<std::vector<std::complex<double>>> mat_aos;

    std::complex<double> alpha;

    BenchData(size_t size) 
        : aos_a(size), aos_b(size), aos_res(size),
          soa_a(size), soa_b(size), soa_res(size),
          mat_soa(64, size), mat_aos(64, std::vector<std::complex<double>>(size))
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        alpha = {dist(gen), dist(gen)};

        for (size_t i = 0; i < size; ++i) {
            aos_a[i] = {dist(gen), dist(gen)};
            aos_b[i] = {dist(gen), dist(gen)};

            soa_a.real[i] = aos_a[i].real();
            soa_a.imag[i] = aos_a[i].imag();

            soa_b.real[i] = aos_b[i].real();
            soa_b.imag[i] = aos_b[i].imag();
        }

        // Initialize matrix for GEMV
        for (size_t i = 0; i < 64; ++i) {
            for (size_t j = 0; j < size; ++j) {
                std::complex<double> val = {dist(gen), dist(gen)};
                mat_aos[i][j] = val;
                mat_soa.real[i * mat_soa.stride_cols + j] = val.real();
                mat_soa.imag[i * mat_soa.stride_cols + j] = val.imag();
            }
        }
    }
};

// --- Addition ---

static void BM_Addition_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            data.aos_res[i] = data.aos_a[i] + data.aos_b[i];
        }
        benchmark::DoNotOptimize(data.aos_res.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_Addition_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        AddSoA_AVX2(data.soa_a, data.soa_b, data.soa_res);
        benchmark::DoNotOptimize(data.soa_res.real.data());
        benchmark::DoNotOptimize(data.soa_res.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

// --- Multiplication ---

static void BM_Multiplication_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            data.aos_res[i] = data.aos_a[i] * data.aos_b[i];
        }
        benchmark::DoNotOptimize(data.aos_res.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_Multiplication_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        MultiplySoA_AVX2(data.soa_a, data.soa_b, data.soa_res);
        benchmark::DoNotOptimize(data.soa_res.real.data());
        benchmark::DoNotOptimize(data.soa_res.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}


// --- DAXPY ---

static void BM_Daxpy_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);
    std::complex<double> alpha = data.alpha;

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            data.aos_b[i] += alpha * data.aos_a[i];
        }
        benchmark::DoNotOptimize(data.aos_b.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_Daxpy_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);
    double a_r = data.alpha.real();
    double a_i = data.alpha.imag();

    for (auto _ : state) {
        DaxpySoA_AVX2(data.soa_a, data.soa_b, a_r, a_i);
        benchmark::DoNotOptimize(data.soa_b.real.data());
        benchmark::DoNotOptimize(data.soa_b.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

// --- Division ---

static void BM_Division_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            data.aos_res[i] = data.aos_a[i] / data.aos_b[i];
        }
        benchmark::DoNotOptimize(data.aos_res.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_Division_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        DivideSoA_AVX2(data.soa_a, data.soa_b, data.soa_res);
        benchmark::DoNotOptimize(data.soa_res.real.data());
        benchmark::DoNotOptimize(data.soa_res.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

// --- Dot Product ---

static void BM_DotProduct_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        std::complex<double> res(0, 0);
        for (size_t i = 0; i < size; ++i) {
            res += data.aos_a[i] * std::conj(data.aos_b[i]);
        }
        benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_DotProduct_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        double r, i;
        DotProductSoA_AVX2(data.soa_a, data.soa_b, r, i);
        benchmark::DoNotOptimize(r);
        benchmark::DoNotOptimize(i);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

// --- GEMV (Matrix-Vector) ---

static void BM_GEMV_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);
    std::vector<std::complex<double>> res(64);

    for (auto _ : state) {
        for (size_t i = 0; i < 64; ++i) {
            std::complex<double> row_res(0, 0);
            for (size_t j = 0; j < size; ++j) {
                row_res += data.mat_aos[i][j] * data.aos_a[j];
            }
            res[i] = row_res;
        }
        benchmark::DoNotOptimize(res.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size * 64);
}

static void BM_GEMV_SoA_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);
    ComplexVectorSoA res(64);

    for (auto _ : state) {
        GemvSoA_AVX2(data.mat_soa, data.soa_a, res);
        benchmark::DoNotOptimize(res.real.data());
        benchmark::DoNotOptimize(res.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size * 64);
}

// --- Conversion Overhead ---

static void BM_Conversion_AoS2SoA(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<std::complex<double>> aos(size, {1.0, 1.0});

    for (auto _ : state) {
        ComplexVectorSoA soa = AoS_to_SoA(aos);
        benchmark::DoNotOptimize(soa.real.data());
        benchmark::ClobberMemory();
    }
}

static void BM_Conversion_SoA2AoS(benchmark::State& state) {
    size_t size = state.range(0);
    ComplexVectorSoA soa(size);

    for (auto _ : state) {
        std::vector<std::complex<double>> aos = SoA_to_AoS(soa);
        benchmark::DoNotOptimize(aos.data());
        benchmark::ClobberMemory();
    }
}


// --- FFT ---

static void BM_FFT_AoS(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<std::complex<double>> copy = data.aos_a;
        state.ResumeTiming();

        fft_iterative_aos(copy);

        benchmark::DoNotOptimize(copy.data());
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(size * std::log2(size));
}

static void BM_FFT_SoA(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        state.PauseTiming();
        ComplexVectorSoA copy = data.soa_a; 
        state.ResumeTiming();

        fft_iterative_soa(copy);

        benchmark::DoNotOptimize(copy.real.data());
        benchmark::DoNotOptimize(copy.imag.data());
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(size * std::log2(size));
}

const int MIN_RANGE = 1 << 10;
const int MAX_RANGE = 1 << 14; // Further reduced range to keep GEMV benchmarks within reasonable time

BENCHMARK(BM_Addition_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Addition_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_Multiplication_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Multiplication_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_Daxpy_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Daxpy_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_Division_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Division_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_DotProduct_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DotProduct_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_GEMV_AoS)->Range(256, 1024);
BENCHMARK(BM_GEMV_SoA_AVX2)->Range(256, 1024);

BENCHMARK(BM_Conversion_AoS2SoA)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Conversion_SoA2AoS)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_FFT_AoS)->Range(MIN_RANGE, MAX_RANGE)->Complexity(benchmark::oNLogN);
BENCHMARK(BM_FFT_SoA)->Range(MIN_RANGE, MAX_RANGE)->Complexity(benchmark::oNLogN);

BENCHMARK_MAIN();
