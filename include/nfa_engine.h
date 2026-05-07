#pragma once

#include "nfa_types.h"
#include "fractal_anchor.h"
#include "topological_quantizer.h"
#include "dna_engram.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace nfa {

// Layer descriptor: how a single layer's weights are encoded
struct LayerEncoding {
    uint32_t layer_id;
    uint32_t rows;
    uint32_t cols;

    // Fractal anchors for this layer
    std::vector<FractalSeed> fractal_seeds;
    float fractal_scale;
    float fractal_bias;

    // Topological indices for residual correction
    std::vector<uint32_t> topo_indices;
    uint32_t topo_group_size;
    uint32_t topo_curve_order;
    float topo_range_min;
    float topo_range_max;

    // DNA engram experts for dynamic reconstruction
    std::vector<DNAEngramSystem::MicroExpert> engram_experts;
};

// The unified NFA engine
class NFAEngine {
public:
    explicit NFAEngine(const NFAConfig& config = default_config());
    ~NFAEngine();

    // Compress a full model (layer-by-layer)
    void compress_model(
        const std::vector<std::pair<const float*, std::pair<uint32_t, uint32_t>>>& layers);

    // Save compressed model to file (~200MB target)
    void save(const std::string& path) const;

    // Load compressed model from file
    void load(const std::string& path);

    // Expand all layers into VRAM (full model materialization)
    std::vector<GPUWeightBuffer> expand_all_gpu();

    // Expand a single layer on-demand
    GPUWeightBuffer expand_layer_gpu(uint32_t layer_id);

    // JIT forward pass: reconstruct weights for a specific input
    GPUWeightBuffer jit_forward_gpu(
        uint32_t layer_id, const float* d_input, uint32_t input_dim);

    // Free all GPU buffers
    void free_all_gpu();

    // Stats
    size_t compressed_bytes() const;
    size_t expanded_bytes() const;
    float compression_ratio() const;
    uint32_t num_layers() const;

private:
    NFAConfig config_;
    NFAModelDescriptor descriptor_;
    std::vector<LayerEncoding> layers_;
    std::vector<GPUWeightBuffer> gpu_buffers_;

    // Internal: three-stage expansion pipeline
    float* stage1_fractal_expand(const LayerEncoding& layer);
    float* stage2_topo_correct(const LayerEncoding& layer, float* d_base);
    float* stage3_engram_refine(const LayerEncoding& layer,
                                 float* d_base, const float* d_input);
};

} // namespace nfa
