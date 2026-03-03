project('FastComplex-Benchmark', 'cpp',
  version : '1.0',
  default_options : ['cpp_std=c++17', 'warning_level=2', 'optimization=3']
)

# Add AVX2 support
add_project_arguments('-march=native', '-mavx2', language : 'cpp')

# Dependencies: using cmake module to import google benchmark
cmake = import('cmake')
benchmark_opts = cmake.subproject_options()
benchmark_opts.add_cmake_defines({'BENCHMARK_ENABLE_TESTING': 'OFF'})

benchmark_subproj = cmake.subproject('google-benchmark', options: benchmark_opts)
benchmark_dep = benchmark_subproj.dependency('benchmark')

thread_dep = dependency('threads')

# Include directories
inc = include_directories('include')

# Executable
executable('benchmark_main',
  'src/benchmark_main.cpp',
  include_directories : inc,
  dependencies : [benchmark_dep, thread_dep],
  install : false
)

executable('tests',
  'src/tests.cpp',
  include_directories : inc,
  dependencies : [thread_dep], # Google benchmark not needed for sanity tests
  install : false
)
