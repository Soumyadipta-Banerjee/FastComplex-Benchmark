#include <benchmark/benchmark.h>
#include <vector>
#include <complex>
#include <random>

#include "complex_soa.hpp"
#include "fft_stress_test.hpp"

struct BenchData {
    std::vector<std::complex<double>> aos_a;
    std::vector<std::complex<double>> aos_b;
    std::vector<std::complex<double>> aos_res;

    ComplexVectorSoA soa_a;
    ComplexVectorSoA soa_b;
    ComplexVectorSoA soa_res;

    std::complex<double> alpha;

    BenchData(size_t size) 
        : aos_a(size), aos_b(size), aos_res(size),
          soa_a(size), soa_b(size), soa_res(size) 
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
    }
};

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

static void BM_Multiplication_SoA(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);

    for (auto _ : state) {
        MultiplySoA(data.soa_a, data.soa_b, data.soa_res);
        benchmark::DoNotOptimize(data.soa_res.real.data());
        benchmark::DoNotOptimize(data.soa_res.imag.data());
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
            // y = a*x + y
            data.aos_b[i] += alpha * data.aos_a[i];
        }
        benchmark::DoNotOptimize(data.aos_b.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
}

static void BM_Daxpy_SoA(benchmark::State& state) {
    size_t size = state.range(0);
    BenchData data(size);
    double a_r = data.alpha.real();
    double a_i = data.alpha.imag();

    for (auto _ : state) {
        DaxpySoA(data.soa_a, data.soa_b, a_r, a_i);
        benchmark::DoNotOptimize(data.soa_b.real.data());
        benchmark::DoNotOptimize(data.soa_b.imag.data());
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
const int MAX_RANGE = 1 << 16;  // Reduced upper range slightly to iterate faster on typical caches

BENCHMARK(BM_Multiplication_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Multiplication_SoA)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Multiplication_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_Daxpy_AoS)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Daxpy_SoA)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_Daxpy_SoA_AVX2)->Range(MIN_RANGE, MAX_RANGE);

BENCHMARK(BM_FFT_AoS)->Range(MIN_RANGE, MAX_RANGE)->Complexity(benchmark::oNLogN);
BENCHMARK(BM_FFT_SoA)->Range(MIN_RANGE, MAX_RANGE)->Complexity(benchmark::oNLogN);

BENCHMARK_MAIN();
