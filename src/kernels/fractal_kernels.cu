#include "nfa_types.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

namespace nfa {
namespace kernels {

// IFS (Iterated Function System) fractal expansion kernel
// Each thread generates one weight by running the chaos game from fractal seeds
__global__ void fractal_expand_kernel(
    const FractalSeed* __restrict__ seeds,
    uint32_t num_seeds,
    float* __restrict__ output,
    uint32_t rows, uint32_t cols,
    float global_scale, float global_bias,
    uint32_t iterations)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = rows * cols;
    if (idx >= total) return;

    // Deterministic seed per output element for reproducibility
    curandState rng;
    curand_init(idx * 6364136223846793005ULL + 1442695040888963407ULL, 0, 0, &rng);

    // Target position in normalized [0,1] space
    float target_x = static_cast<float>(idx % cols) / static_cast<float>(cols);
    float target_y = static_cast<float>(idx / cols) / static_cast<float>(rows);

    // Run chaos game: iterate the IFS
    float x = 0.5f, y = 0.5f;
    float accumulated = 0.0f;
    float weight_sum = 0.0f;

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        // Select a transform based on cumulative probability
        float r = curand_uniform(&rng);
        float cumulative = 0.0f;
        uint32_t selected = 0;

        for (uint32_t s = 0; s < num_seeds; ++s) {
            cumulative += seeds[s].probability;
            if (r <= cumulative) {
                selected = s;
                break;
            }
        }

        const FractalSeed& seed = seeds[selected];

        // Apply 2D affine transform: [a b c; d e f] * [x; y; 1]
        float nx = seed.affine[0] * x + seed.affine[1] * y + seed.affine[2];
        float ny = seed.affine[3] * x + seed.affine[4] * y + seed.affine[5];
        x = nx;
        y = ny;

        // Accumulate weighted contribution based on proximity to target
        float dx = x - target_x;
        float dy = y - target_y;
        float dist_sq = dx * dx + dy * dy;
        float kernel_val = expf(-dist_sq * static_cast<float>(rows * cols) * 0.5f);

        accumulated += kernel_val * seed.scale + seed.bias;
        weight_sum += kernel_val;
    }

    float value = (weight_sum > 1e-8f)
        ? (accumulated / weight_sum) * global_scale + global_bias
        : global_bias;

    output[idx] = value;
}

// Tensor product kernel: outer product of two vectors to form weight submatrix
// Used for low-rank reconstruction in engram system
__global__ void tensor_product_kernel(
    const float* __restrict__ row_basis,
    const float* __restrict__ col_basis,
    const float* __restrict__ coefficients,
    float* __restrict__ output,
    uint32_t rows, uint32_t cols,
    uint32_t rank)
{
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    float sum = 0.0f;
    for (uint32_t r = 0; r < rank; ++r) {
        sum += coefficients[r] * row_basis[r * rows + row] * col_basis[r * cols + col];
    }

    output[row * cols + col] = sum;
}

// Accumulate (add) one matrix into another
__global__ void accumulate_kernel(
    float* __restrict__ base,
    const float* __restrict__ addition,
    uint32_t count,
    float scale)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    base[idx] += addition[idx] * scale;
}

// Element-wise add with scaling
__global__ void add_scaled_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float scale_a, float scale_b,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    output[idx] = a[idx] * scale_a + b[idx] * scale_b;
}

} // namespace kernels

// Host wrapper for fractal expansion
void launch_fractal_expand(
    const FractalSeed* d_seeds, uint32_t num_seeds,
    float* d_output, uint32_t rows, uint32_t cols,
    float global_scale, float global_bias, uint32_t iterations)
{
    uint32_t total = rows * cols;
    uint32_t block_size = 256;
    uint32_t grid_size = (total + block_size - 1) / block_size;

    kernels::fractal_expand_kernel<<<grid_size, block_size>>>(
        d_seeds, num_seeds, d_output, rows, cols,
        global_scale, global_bias, iterations);

    cudaDeviceSynchronize();
}

// Host wrapper for tensor product
void launch_tensor_product(
    const float* d_row_basis, const float* d_col_basis,
    const float* d_coefficients, float* d_output,
    uint32_t rows, uint32_t cols, uint32_t rank)
{
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    kernels::tensor_product_kernel<<<grid, block>>>(
        d_row_basis, d_col_basis, d_coefficients,
        d_output, rows, cols, rank);

    cudaDeviceSynchronize();
}

// Host wrapper for accumulate
void launch_accumulate(float* d_base, const float* d_addition,
                       uint32_t count, float scale)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (count + block_size - 1) / block_size;

    kernels::accumulate_kernel<<<grid_size, block_size>>>(
        d_base, d_addition, count, scale);

    cudaDeviceSynchronize();
}

// Host wrapper for scaled add
void launch_add_scaled(float* d_output, const float* d_a, const float* d_b,
                       float scale_a, float scale_b, uint32_t count)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (count + block_size - 1) / block_size;

    kernels::add_scaled_kernel<<<grid_size, block_size>>>(
        d_output, d_a, d_b, scale_a, scale_b, count);

    cudaDeviceSynchronize();
}

} // namespace nfa
