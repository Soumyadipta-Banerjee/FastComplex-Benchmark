cmake_minimum_required(VERSION 3.14)
project(FastComplex-Benchmark CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable required optimization and architecture flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
# Disable benchmark testing to avoid building their tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Find Threads (required for Google Benchmark)
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(benchmark_main src/benchmark_main.cpp)
target_link_libraries(benchmark_main PRIVATE benchmark::benchmark benchmark::benchmark_main Threads::Threads)
