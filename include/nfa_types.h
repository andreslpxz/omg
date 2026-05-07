#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace nfa {

// Fractal seed: encodes an IFS (Iterated Function System) attractor
struct FractalSeed {
    float affine[6];       // 2D affine transform coefficients [a,b,c,d,e,f]
    float probability;     // selection probability for this transform
    float scale;           // amplitude scaling for weight reconstruction
    float bias;            // bias offset
    uint32_t iterations;   // recursion depth
};

// A fractal anchor replaces a weight matrix
struct FractalAnchor {
    uint32_t rows;
    uint32_t cols;
    uint32_t num_seeds;
    float global_scale;
    float global_bias;
    // seeds stored contiguously after this header
};

// Hilbert curve quantization entry
struct TopologicalIndex {
    uint32_t curve_order;      // Hilbert curve order (resolution)
    uint32_t dimensions;       // dimensionality of the curve
    uint32_t num_entries;      // number of indexed weight groups
    float value_range_min;
    float value_range_max;
};

// DNA engram: activation dictionary entry for JIT weight reconstruction
struct EngramEntry {
    uint32_t expert_id;
    uint32_t input_dim;
    uint32_t output_dim;
    float activation_threshold;
    // Compact representation: basis vectors + coefficients
    uint32_t num_basis;
};

// Full model descriptor
struct NFAModelDescriptor {
    uint32_t magic;             // 0x4E464121 = "NFA!"
    uint32_t version;
    uint32_t num_layers;
    uint64_t compressed_size;   // total file size
    uint64_t expanded_size;     // total weight bytes when expanded
    float compression_ratio;
};

// GPU-side buffer handle
struct GPUWeightBuffer {
    float* data;
    size_t rows;
    size_t cols;
    size_t bytes;
};

// Configuration for the NFA engine
struct NFAConfig {
    uint32_t fractal_iterations;    // default recursion depth
    uint32_t hilbert_order;         // Hilbert curve order
    uint32_t num_micro_experts;     // MoE expert count
    uint32_t basis_rank;            // low-rank basis dimension
    float sparsity_threshold;       // activation sparsity for engrams
    bool use_tensor_cores;          // prefer tensor core operations
    size_t max_vram_bytes;          // VRAM budget
};

inline NFAConfig default_config() {
    return NFAConfig{
        .fractal_iterations = 1024,
        .hilbert_order = 8,
        .num_micro_experts = 4096,
        .basis_rank = 64,
        .sparsity_threshold = 0.01f,
        .use_tensor_cores = true,
        .max_vram_bytes = 8ULL * 1024 * 1024 * 1024  // 8 GB
    };
}

} // namespace nfa
