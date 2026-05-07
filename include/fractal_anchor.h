#pragma once

#include "nfa_types.h"
#include <vector>
#include <cstdint>

namespace nfa {

// Host-side fractal anchor manager
class FractalAnchorSystem {
public:
    // Compress a weight matrix into fractal seeds using IFS fitting
    static std::vector<FractalSeed> compress(
        const float* weights, uint32_t rows, uint32_t cols,
        uint32_t max_seeds, uint32_t iterations);

    // Expand fractal seeds into a full weight matrix on GPU
    // Returns device pointer to reconstructed weights
    static float* expand_gpu(
        const FractalSeed* seeds, uint32_t num_seeds,
        uint32_t rows, uint32_t cols,
        float global_scale, float global_bias);

    // Free GPU weight buffer
    static void free_gpu(float* d_weights);

    // Compute compression ratio
    static float compression_ratio(
        uint32_t num_seeds, uint32_t rows, uint32_t cols);
};

} // namespace nfa
