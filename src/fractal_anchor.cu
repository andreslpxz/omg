#include "fractal_anchor.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

namespace nfa {

// Forward declarations of kernel launchers
void launch_fractal_expand(
    const FractalSeed* d_seeds, uint32_t num_seeds,
    float* d_output, uint32_t rows, uint32_t cols,
    float global_scale, float global_bias, uint32_t iterations);

std::vector<FractalSeed> FractalAnchorSystem::compress(
    const float* weights, uint32_t rows, uint32_t cols,
    uint32_t max_seeds, uint32_t iterations)
{
    std::vector<FractalSeed> seeds;
    seeds.reserve(max_seeds);

    // Compute weight statistics for normalization
    double mean = 0.0, var = 0.0;
    uint32_t total = rows * cols;
    for (uint32_t i = 0; i < total; ++i) mean += weights[i];
    mean /= total;
    for (uint32_t i = 0; i < total; ++i) {
        double d = weights[i] - mean;
        var += d * d;
    }
    var /= total;
    float stddev = static_cast<float>(std::sqrt(var));

    // IFS fitting: use a combination of contractional affine transforms
    // that, when iterated, approximate the weight distribution
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 0.3f);

    float total_prob = 0.0f;

    for (uint32_t s = 0; s < max_seeds; ++s) {
        FractalSeed seed{};

        // Generate contractional affine transforms
        // Ensure |det| < 1 for convergence (IFS contractivity)
        float scale_factor = 0.3f + 0.4f * static_cast<float>(s) / max_seeds;

        seed.affine[0] = scale_factor * (0.5f + 0.5f * uniform(rng));  // a
        seed.affine[1] = normal(rng) * 0.3f;                            // b
        seed.affine[2] = uniform(rng) * 0.5f;                           // c (translation x)
        seed.affine[3] = normal(rng) * 0.3f;                            // d
        seed.affine[4] = scale_factor * (0.5f + 0.5f * uniform(rng));  // e
        seed.affine[5] = uniform(rng) * 0.5f;                           // f (translation y)

        // Probability proportional to area covered
        float det = std::abs(seed.affine[0] * seed.affine[4] -
                             seed.affine[1] * seed.affine[3]);
        seed.probability = std::max(det, 0.01f);
        total_prob += seed.probability;

        // Scale and bias fitted to weight statistics
        seed.scale = stddev * (1.0f + normal(rng) * 0.2f);
        seed.bias = static_cast<float>(mean) * normal(rng) * 0.1f;
        seed.iterations = iterations;

        seeds.push_back(seed);
    }

    // Normalize probabilities
    for (auto& seed : seeds) {
        seed.probability /= total_prob;
    }

    // Iterative refinement: compare generated weights with originals
    // and adjust seed parameters to minimize MSE
    // (Simplified version - production would use gradient-free optimization)
    float best_mse = 1e30f;
    std::vector<FractalSeed> best_seeds = seeds;

    for (int refine = 0; refine < 5; ++refine) {
        // Expand on GPU to test quality
        float* d_seeds_ptr = nullptr;
        float* d_output = nullptr;
        size_t seeds_bytes = seeds.size() * sizeof(FractalSeed);
        size_t output_bytes = total * sizeof(float);

        cudaMalloc(&d_seeds_ptr, seeds_bytes);
        cudaMalloc(&d_output, output_bytes);
        cudaMemcpy(d_seeds_ptr, seeds.data(), seeds_bytes, cudaMemcpyHostToDevice);

        launch_fractal_expand(
            reinterpret_cast<FractalSeed*>(d_seeds_ptr), seeds.size(),
            d_output, rows, cols, 1.0f, 0.0f, iterations);

        std::vector<float> reconstructed(total);
        cudaMemcpy(reconstructed.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_seeds_ptr);
        cudaFree(d_output);

        // Compute MSE
        float mse = 0.0f;
        for (uint32_t i = 0; i < total; ++i) {
            float d = weights[i] - reconstructed[i];
            mse += d * d;
        }
        mse /= total;

        if (mse < best_mse) {
            best_mse = mse;
            best_seeds = seeds;
        }

        // Perturb seeds for next iteration
        for (auto& seed : seeds) {
            for (int j = 0; j < 6; ++j) {
                seed.affine[j] += normal(rng) * 0.01f;
            }
            seed.scale += normal(rng) * stddev * 0.01f;
            seed.bias += normal(rng) * 0.001f;
        }
    }

    return best_seeds;
}

float* FractalAnchorSystem::expand_gpu(
    const FractalSeed* seeds, uint32_t num_seeds,
    uint32_t rows, uint32_t cols,
    float global_scale, float global_bias)
{
    // Upload seeds to GPU
    FractalSeed* d_seeds = nullptr;
    float* d_output = nullptr;
    size_t seeds_bytes = num_seeds * sizeof(FractalSeed);
    size_t output_bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    cudaMalloc(&d_seeds, seeds_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemcpy(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice);

    uint32_t iterations = (num_seeds > 0) ? seeds[0].iterations : 1024;

    launch_fractal_expand(d_seeds, num_seeds, d_output, rows, cols,
                          global_scale, global_bias, iterations);

    cudaFree(d_seeds);
    return d_output;
}

void FractalAnchorSystem::free_gpu(float* d_weights)
{
    if (d_weights) cudaFree(d_weights);
}

float FractalAnchorSystem::compression_ratio(
    uint32_t num_seeds, uint32_t rows, uint32_t cols)
{
    size_t original = static_cast<size_t>(rows) * cols * sizeof(float);
    size_t compressed = num_seeds * sizeof(FractalSeed);
    return static_cast<float>(original) / static_cast<float>(compressed);
}

} // namespace nfa
