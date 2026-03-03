# FastComplex-Benchmark

This is a personal proof-of-concept project I built to investigate performance bottlenecks related to complex arithmetic in CFD solvers (specifically targeting issues similar to those seen when enabling complex-number features in SU2).

## Motivation

While profiling CFD code, I noticed a significant performance penalty when using the standard `std::complex<double>`. This type uses an Array of Structures (AoS) memory layout. In memory-bandwidth-bound linear algebra operations like DAXPY ($y = \alpha x + y$) or basic complex multiplication, AoS layouts limit the compiler's ability to issue contiguous SIMD loads and store instructions.

This repository tests the hypothesis that replacing `std::complex` (AoS) with a custom Structure of Arrays (SoA) layout—and manually writing AVX2 intrinsics to process four `double` values per clock cycle—can yield substantial speedups in operations common to CFD (Addition, Multiplication, DAXPY, Division, Dot Product, and GEMV).

## New Features (Expanded POC)

- **Vectorized Kernels**: Added AVX2-optimized implementations for:
  - **Complex Division**: ~4.2x speedup over `std::complex`.
  - **Complex Dot Product**: ~3.9x speedup.
  - **Complex GEMV (Matrix-Vector)**: ~3.2x speedup.
- **Improved Layout**: Implemented `ComplexVectorSoA` and `MatrixSoA` for better SIMD efficiency.
- **Conversion Helpers**: Added highly efficient AoS $\leftrightarrow$ SoA transformation utilities.
- **Verification Suite**: Added a dedicated sanity test to ensure mathematical parity between scalar and vectorized paths.

## Requirements

*   GCC or Clang with C++17 support
*   CPU with AVX2 support
*   **Meson** (>= 0.53) and **Ninja** (Primary build system, matching SU2's stack)
*   *Alternatively, CMake (>= 3.14) is also supported.*
*   Python 3 (matplotlib, pandas) for graphing

## Building (via Meson)

I set this up using Meson since it's the standard for SU2. The build config will automatically fetch Google Benchmark as a subproject.

```bash
meson setup builddir --buildtype=release
meson compile -C builddir
```

*(If you prefer CMake, `mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make` works too).*

## Running Tests & Benchmarks

To run the mathematical sanity checks:
```bash
./builddir/tests
```

To run the Google Benchmark suite:
```bash
./builddir/benchmark_main --benchmark_out=results.json --benchmark_out_format=json
```

If you want to run `perf` to see the actual instruction overhead or cache misses (which was the whole point of this exercise):

```bash
perf stat -e cache-misses,cache-references,instructions,cycles ./builddir/benchmark_main
```

## Plotting the Data

I included a quick Python script to parse the JSON and dump out some graphs showing the latency scaling across different vector sizes (from sizes that fit in L1 cache up to RAM-bound sizes).

```bash
python3 scripts/plot_results.py results.json
```
