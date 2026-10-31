# Build all C++ benchmarks:
# cmake -S ~/feather-ops/cpp_benchmarks -B ~/feather-ops/cpp_benchmarks/build -G Ninja
# cmake --build ~/feather-ops/cpp_benchmarks/build
#
# The libtorch benchmark is skipped if Torch cannot be found.
# Torch is auto-detected from the active Python environment when possible.

cmake_minimum_required(VERSION 3.21)
project(feather_ops_cpp_benchmarks LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architectures")
endif()

find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

find_package(Torch QUIET)
if(NOT Torch_FOUND)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
      RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(TORCH_CMAKE_PREFIX_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
      find_package(Torch QUIET)
    endif()
  endif()
endif()

function(add_hip_kernel_benchmark target bench_source kernel_source)
  add_executable(${target} ${bench_source} ${kernel_source})
  set_source_files_properties(${kernel_source} PROPERTIES LANGUAGE HIP COMPILE_DEFINITIONS NO_PYTORCH)
  target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(${target} PRIVATE hip::host)
endfunction()

add_hip_kernel_benchmark(
  benchmark_mm_hip_fp16
  benchmark_mm_hip_fp16.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/hip_kernel_fp16.cu
)

add_hip_kernel_benchmark(
  benchmark_scaled_mm_hip
  benchmark_scaled_mm_hip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/hip_kernel.cu
)

add_executable(benchmark_mm_hipblaslt_fp16 benchmark_mm_hipblaslt_fp16.cpp)
target_compile_definitions(benchmark_mm_hipblaslt_fp16 PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(benchmark_mm_hipblaslt_fp16 PRIVATE hip::host roc::hipblaslt)

if(Torch_FOUND)
  add_hip_kernel_benchmark(
    benchmark_scaled_mm_hip_libtorch
    benchmark_scaled_mm_hip_libtorch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/hip_kernel.cu
  )
  target_link_libraries(benchmark_scaled_mm_hip_libtorch PRIVATE ${TORCH_LIBRARIES})

  # libtorch_cpu has a runtime dependency on the ROCm host BLAS that its
  # exported CMake targets do not list. Link it explicitly so the benchmark
  # does not need a custom LD_LIBRARY_PATH at runtime.
  find_library(
    ROCM_OPENBLAS_LIBRARY
    NAMES rocm-openblas
    HINTS
      "${ROCM_PATH}/lib/host-math/lib"
      "$ENV{ROCM_PATH}/lib/host-math/lib"
      "${ROCM_PATH}/lib"
      "$ENV{ROCM_PATH}/lib"
  )
  if(ROCM_OPENBLAS_LIBRARY)
    target_link_libraries(benchmark_scaled_mm_hip_libtorch PRIVATE "${ROCM_OPENBLAS_LIBRARY}")
  endif()
else()
  message(STATUS "Torch not found. Skipping benchmark_scaled_mm_hip_libtorch. Set CMAKE_PREFIX_PATH to torch.utils.cmake_prefix_path or Python3_EXECUTABLE to enable it.")
endif()
