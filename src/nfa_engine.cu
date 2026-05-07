#include "nfa_engine.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace nfa {

// Forward declarations from kernel files
void launch_accumulate(float* d_base, const float* d_addition,
                       uint32_t count, float scale);
void launch_add_scaled(float* d_output, const float* d_a, const float* d_b,
                       float scale_a, float scale_b, uint32_t count);

NFAEngine::NFAEngine(const NFAConfig& config)
    : config_(config)
{
    descriptor_.magic = 0x4E464121;  // "NFA!"
    descriptor_.version = 1;
    descriptor_.num_layers = 0;
    descriptor_.compressed_size = 0;
    descriptor_.expanded_size = 0;
    descriptor_.compression_ratio = 0.0f;
}

NFAEngine::~NFAEngine()
{
    free_all_gpu();
}

void NFAEngine::compress_model(
    const std::vector<std::pair<const float*, std::pair<uint32_t, uint32_t>>>& layers)
{
    layers_.clear();
    layers_.reserve(layers.size());

    for (uint32_t l = 0; l < layers.size(); ++l) {
        const float* weights = layers[l].first;
        uint32_t rows = layers[l].second.first;
        uint32_t cols = layers[l].second.second;
        uint32_t total = rows * cols;

        LayerEncoding encoding;
        encoding.layer_id = l;
        encoding.rows = rows;
        encoding.cols = cols;

        // --- Stage 1: Fractal Anchor Compression ---
        // Number of seeds scales with sqrt of matrix size for good coverage
        uint32_t num_seeds = std::max(8u,
            static_cast<uint32_t>(std::sqrt(static_cast<float>(total)) * 0.1f));
        num_seeds = std::min(num_seeds, 256u);  // cap for memory

        encoding.fractal_seeds = FractalAnchorSystem::compress(
            weights, rows, cols, num_seeds, config_.fractal_iterations);

        // Compute global scale/bias from weight statistics
        double mean = 0.0, var = 0.0;
        for (uint32_t i = 0; i < total; ++i) mean += weights[i];
        mean /= total;
        for (uint32_t i = 0; i < total; ++i) {
            double d = weights[i] - mean;
            var += d * d;
        }
        var /= total;
        encoding.fractal_scale = static_cast<float>(std::sqrt(var));
        encoding.fractal_bias = static_cast<float>(mean);

        // --- Stage 2: Topological Quantization of Residuals ---
        // First, get fractal reconstruction to compute residuals
        float* d_fractal = FractalAnchorSystem::expand_gpu(
            encoding.fractal_seeds.data(), encoding.fractal_seeds.size(),
            rows, cols, encoding.fractal_scale, encoding.fractal_bias);

        std::vector<float> fractal_weights(total);
        cudaMemcpy(fractal_weights.data(), d_fractal, total * sizeof(float),
                   cudaMemcpyDeviceToHost);
        FractalAnchorSystem::free_gpu(d_fractal);

        // Compute residuals
        std::vector<float> residuals(total);
        float res_min = 1e30f, res_max = -1e30f;
        for (uint32_t i = 0; i < total; ++i) {
            residuals[i] = weights[i] - fractal_weights[i];
            res_min = std::min(res_min, residuals[i]);
            res_max = std::max(res_max, residuals[i]);
        }

        // Topological encoding of residuals (grouped for compression)
        encoding.topo_group_size = std::max(4u, std::min(64u,
            static_cast<uint32_t>(std::sqrt(static_cast<float>(total)) * 0.01f)));
        encoding.topo_curve_order = config_.hilbert_order;
        encoding.topo_range_min = res_min;
        encoding.topo_range_max = res_max;

        encoding.topo_indices = TopologicalQuantizer::encode_grouped(
            residuals.data(), total, encoding.topo_group_size,
            encoding.topo_curve_order, res_min, res_max);

        // --- Stage 3: DNA Engram Dictionary ---
        // Build micro-expert dictionary from remaining error
        float* d_topo = TopologicalQuantizer::decode_grouped_gpu(
            encoding.topo_indices.data(),
            static_cast<uint32_t>(encoding.topo_indices.size()),
            encoding.topo_group_size, encoding.topo_curve_order,
            encoding.topo_range_min, encoding.topo_range_max);

        std::vector<float> topo_residuals(total);
        cudaMemcpy(topo_residuals.data(), d_topo, total * sizeof(float),
                   cudaMemcpyDeviceToHost);
        TopologicalQuantizer::free_gpu(d_topo);

        // Second-order residuals for engram refinement
        std::vector<float> residual2(total);
        for (uint32_t i = 0; i < total; ++i) {
            residual2[i] = residuals[i] - topo_residuals[i];
        }

        uint32_t num_experts = std::min(config_.num_micro_experts,
            std::max(4u, total / 256));
        encoding.engram_experts = DNAEngramSystem::build_dictionary(
            residual2.data(), rows, cols, num_experts, config_.basis_rank);

        layers_.push_back(std::move(encoding));
    }

    descriptor_.num_layers = static_cast<uint32_t>(layers_.size());
    descriptor_.compressed_size = compressed_bytes();
    descriptor_.expanded_size = expanded_bytes();
    descriptor_.compression_ratio = compression_ratio();
}

void NFAEngine::save(const std::string& path) const
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&descriptor_), sizeof(descriptor_));

    // Write each layer
    for (const auto& layer : layers_) {
        // Layer header
        file.write(reinterpret_cast<const char*>(&layer.layer_id), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&layer.rows), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&layer.cols), sizeof(uint32_t));

        // Fractal seeds
        uint32_t num_seeds = static_cast<uint32_t>(layer.fractal_seeds.size());
        file.write(reinterpret_cast<const char*>(&num_seeds), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(layer.fractal_seeds.data()),
                   num_seeds * sizeof(FractalSeed));
        file.write(reinterpret_cast<const char*>(&layer.fractal_scale), sizeof(float));
        file.write(reinterpret_cast<const char*>(&layer.fractal_bias), sizeof(float));

        // Topological indices
        uint32_t num_indices = static_cast<uint32_t>(layer.topo_indices.size());
        file.write(reinterpret_cast<const char*>(&num_indices), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(layer.topo_indices.data()),
                   num_indices * sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&layer.topo_group_size), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&layer.topo_curve_order), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&layer.topo_range_min), sizeof(float));
        file.write(reinterpret_cast<const char*>(&layer.topo_range_max), sizeof(float));

        // DNA engram dictionary
        auto engram_data = DNAEngramSystem::serialize(layer.engram_experts);
        uint32_t engram_size = static_cast<uint32_t>(engram_data.size());
        file.write(reinterpret_cast<const char*>(&engram_size), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(engram_data.data()), engram_size);
    }
}

void NFAEngine::load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    file.read(reinterpret_cast<char*>(&descriptor_), sizeof(descriptor_));

    if (descriptor_.magic != 0x4E464121) {
        throw std::runtime_error("Invalid NFA file magic number");
    }

    layers_.clear();
    layers_.resize(descriptor_.num_layers);

    for (uint32_t l = 0; l < descriptor_.num_layers; ++l) {
        auto& layer = layers_[l];

        file.read(reinterpret_cast<char*>(&layer.layer_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&layer.rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&layer.cols), sizeof(uint32_t));

        uint32_t num_seeds;
        file.read(reinterpret_cast<char*>(&num_seeds), sizeof(uint32_t));
        layer.fractal_seeds.resize(num_seeds);
        file.read(reinterpret_cast<char*>(layer.fractal_seeds.data()),
                  num_seeds * sizeof(FractalSeed));
        file.read(reinterpret_cast<char*>(&layer.fractal_scale), sizeof(float));
        file.read(reinterpret_cast<char*>(&layer.fractal_bias), sizeof(float));

        uint32_t num_indices;
        file.read(reinterpret_cast<char*>(&num_indices), sizeof(uint32_t));
        layer.topo_indices.resize(num_indices);
        file.read(reinterpret_cast<char*>(layer.topo_indices.data()),
                  num_indices * sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&layer.topo_group_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&layer.topo_curve_order), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&layer.topo_range_min), sizeof(float));
        file.read(reinterpret_cast<char*>(&layer.topo_range_max), sizeof(float));

        uint32_t engram_size;
        file.read(reinterpret_cast<char*>(&engram_size), sizeof(uint32_t));
        std::vector<uint8_t> engram_data(engram_size);
        file.read(reinterpret_cast<char*>(engram_data.data()), engram_size);
        layer.engram_experts = DNAEngramSystem::deserialize(
            engram_data.data(), engram_size);
    }
}

std::vector<GPUWeightBuffer> NFAEngine::expand_all_gpu()
{
    free_all_gpu();
    gpu_buffers_.resize(layers_.size());

    for (uint32_t l = 0; l < layers_.size(); ++l) {
        gpu_buffers_[l] = expand_layer_gpu(l);
    }

    return gpu_buffers_;
}

GPUWeightBuffer NFAEngine::expand_layer_gpu(uint32_t layer_id)
{
    if (layer_id >= layers_.size()) {
        throw std::runtime_error("Invalid layer ID");
    }

    const auto& layer = layers_[layer_id];
    uint32_t total = layer.rows * layer.cols;

    // Stage 1: Fractal base expansion
    float* d_base = stage1_fractal_expand(layer);

    // Stage 2: Topological residual correction
    float* d_corrected = stage2_topo_correct(layer, d_base);

    GPUWeightBuffer buf;
    buf.data = d_corrected;
    buf.rows = layer.rows;
    buf.cols = layer.cols;
    buf.bytes = static_cast<size_t>(total) * sizeof(float);

    return buf;
}

GPUWeightBuffer NFAEngine::jit_forward_gpu(
    uint32_t layer_id, const float* d_input, uint32_t input_dim)
{
    if (layer_id >= layers_.size()) {
        throw std::runtime_error("Invalid layer ID");
    }

    const auto& layer = layers_[layer_id];
    uint32_t total = layer.rows * layer.cols;

    // Stage 1 + 2: Get base weights
    float* d_base = stage1_fractal_expand(layer);
    float* d_corrected = stage2_topo_correct(layer, d_base);

    // Stage 3: Engram refinement based on input
    float* d_refined = stage3_engram_refine(layer, d_corrected, d_input);

    GPUWeightBuffer buf;
    buf.data = d_refined;
    buf.rows = layer.rows;
    buf.cols = layer.cols;
    buf.bytes = static_cast<size_t>(total) * sizeof(float);

    return buf;
}

void NFAEngine::free_all_gpu()
{
    for (auto& buf : gpu_buffers_) {
        if (buf.data) {
            cudaFree(buf.data);
            buf.data = nullptr;
        }
    }
    gpu_buffers_.clear();
}

size_t NFAEngine::compressed_bytes() const
{
    size_t total = sizeof(NFAModelDescriptor);

    for (const auto& layer : layers_) {
        total += 3 * sizeof(uint32_t); // layer header
        total += sizeof(uint32_t) + layer.fractal_seeds.size() * sizeof(FractalSeed);
        total += 2 * sizeof(float); // scale, bias
        total += sizeof(uint32_t) + layer.topo_indices.size() * sizeof(uint32_t);
        total += 2 * sizeof(uint32_t) + 2 * sizeof(float); // topo params

        auto engram_data = DNAEngramSystem::serialize(layer.engram_experts);
        total += sizeof(uint32_t) + engram_data.size();
    }

    return total;
}

size_t NFAEngine::expanded_bytes() const
{
    size_t total = 0;
    for (const auto& layer : layers_) {
        total += static_cast<size_t>(layer.rows) * layer.cols * sizeof(float);
    }
    return total;
}

float NFAEngine::compression_ratio() const
{
    size_t comp = compressed_bytes();
    size_t exp = expanded_bytes();
    return (comp > 0) ? static_cast<float>(exp) / static_cast<float>(comp) : 0.0f;
}

uint32_t NFAEngine::num_layers() const
{
    return static_cast<uint32_t>(layers_.size());
}

// --- Internal pipeline stages ---

float* NFAEngine::stage1_fractal_expand(const LayerEncoding& layer)
{
    return FractalAnchorSystem::expand_gpu(
        layer.fractal_seeds.data(),
        static_cast<uint32_t>(layer.fractal_seeds.size()),
        layer.rows, layer.cols,
        layer.fractal_scale, layer.fractal_bias);
}

float* NFAEngine::stage2_topo_correct(const LayerEncoding& layer, float* d_base)
{
    if (layer.topo_indices.empty()) return d_base;

    uint32_t total = layer.rows * layer.cols;

    // Decode topological residuals
    float* d_residuals = TopologicalQuantizer::decode_grouped_gpu(
        layer.topo_indices.data(),
        static_cast<uint32_t>(layer.topo_indices.size()),
        layer.topo_group_size, layer.topo_curve_order,
        layer.topo_range_min, layer.topo_range_max);

    // Add residuals to base: output = base + residuals
    float* d_output = nullptr;
    cudaMalloc(&d_output, total * sizeof(float));

    launch_add_scaled(d_output, d_base, d_residuals, 1.0f, 1.0f, total);

    cudaFree(d_base);
    cudaFree(d_residuals);

    return d_output;
}

float* NFAEngine::stage3_engram_refine(
    const LayerEncoding& layer, float* d_base, const float* d_input)
{
    if (layer.engram_experts.empty() || d_input == nullptr) return d_base;

    uint32_t total = layer.rows * layer.cols;

    // Copy input to host for expert retrieval
    std::vector<float> h_input(layer.cols);
    cudaMemcpy(h_input.data(), d_input,
               std::min(static_cast<size_t>(layer.cols), static_cast<size_t>(layer.cols)) * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Select relevant experts
    uint32_t max_active = std::min(16u,
        static_cast<uint32_t>(layer.engram_experts.size()));

    auto active_ids = DNAEngramSystem::retrieve_experts(
        h_input.data(), layer.cols,
        layer.engram_experts.data(),
        static_cast<uint32_t>(layer.engram_experts.size()),
        config_.sparsity_threshold, max_active);

    if (active_ids.empty()) return d_base;

    // Reconstruct refinement from active experts
    float* d_refinement = DNAEngramSystem::reconstruct_gpu(
        layer.engram_experts.data(), active_ids.data(),
        static_cast<uint32_t>(active_ids.size()),
        layer.rows, layer.cols);

    // Add refinement to corrected base
    launch_accumulate(d_base, d_refinement, total, 1.0f);

    cudaFree(d_refinement);
    return d_base;
}

} // namespace nfa
