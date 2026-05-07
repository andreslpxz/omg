cmake_minimum_required(VERSION 3.18)
project(NeuralFractalAnchoring CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture targets (adjust for your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect source files
set(KERNEL_SOURCES
    src/kernels/fractal_kernels.cu
    src/kernels/hilbert_kernels.cu
    src/kernels/engram_kernels.cu
)

set(LIB_SOURCES
    src/fractal_anchor.cu
    src/topological_quantizer.cu
    src/dna_engram.cu
    src/nfa_engine.cu
)

# NFA static library
add_library(nfa_lib STATIC ${KERNEL_SOURCES} ${LIB_SOURCES})
target_include_directories(nfa_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(nfa_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Demo / benchmark executable
add_executable(nfa_demo demo/main.cu)
target_link_libraries(nfa_demo PRIVATE nfa_lib)
set_target_properties(nfa_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS nfa_lib ARCHIVE DESTINATION lib)
install(TARGETS nfa_demo RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Print build info
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
