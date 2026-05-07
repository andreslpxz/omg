#include "nfa_engine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

using namespace nfa;

static void print_separator() {
    printf("================================================================\n");
}

static void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("[!] No CUDA devices found. Running in CPU-fallback mode.\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);
    printf("VRAM: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Memory Bus: %d-bit\n", prop.memoryBusWidth);
    printf("Peak Bandwidth: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

static float compute_mse(const float* a, const float* b, uint32_t n) {
    double mse = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        mse += d * d;
    }
    return static_cast<float>(mse / n);
}

static float compute_max_error(const float* a, const float* b, uint32_t n) {
    float max_err = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Benchmark: Fractal Anchor compression/expansion
static void bench_fractal_anchor(uint32_t rows, uint32_t cols) {
    print_separator();
    printf("[Benchmark] Fractal Anchor System (%u x %u)\n", rows, cols);

    uint32_t total = rows * cols;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // Generate synthetic weights (simulating transformer layer)
    std::vector<float> weights(total);
    for (auto& w : weights) w = dist(rng);

    printf("  Original size: %.2f MB\n", total * sizeof(float) / (1024.0 * 1024.0));

    // Compress
    auto t0 = std::chrono::high_resolution_clock::now();
    auto seeds = FractalAnchorSystem::compress(weights.data(), rows, cols, 64, 512);
    auto t1 = std::chrono::high_resolution_clock::now();

    double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t compressed_size = seeds.size() * sizeof(FractalSeed);
    float ratio = FractalAnchorSystem::compression_ratio(
        static_cast<uint32_t>(seeds.size()), rows, cols);

    printf("  Compressed size: %.2f KB (%zu seeds)\n",
           compressed_size / 1024.0, seeds.size());
    printf("  Compression ratio: %.1fx\n", ratio);
    printf("  Compression time: %.1f ms\n", compress_ms);

    // Expand on GPU
    t0 = std::chrono::high_resolution_clock::now();
    float* d_expanded = FractalAnchorSystem::expand_gpu(
        seeds.data(), static_cast<uint32_t>(seeds.size()),
        rows, cols, 1.0f, 0.0f);
    t1 = std::chrono::high_resolution_clock::now();

    double expand_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Copy back and measure quality
    std::vector<float> expanded(total);
    cudaMemcpy(expanded.data(), d_expanded, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    FractalAnchorSystem::free_gpu(d_expanded);

    float mse = compute_mse(weights.data(), expanded.data(), total);
    float max_err = compute_max_error(weights.data(), expanded.data(), total);

    printf("  Expansion time: %.1f ms\n", expand_ms);
    printf("  Expansion throughput: %.2f GB/s\n",
           (total * sizeof(float)) / (expand_ms * 1e6));
    printf("  MSE: %.6e\n", mse);
    printf("  Max Error: %.6f\n", max_err);
}

// Benchmark: Topological Quantization
static void bench_topological(uint32_t count, uint32_t group_size) {
    print_separator();
    printf("[Benchmark] Topological Quantizer (%u weights, group=%u)\n",
           count, group_size);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> weights(count);
    for (auto& w : weights) w = dist(rng);

    float wmin = *std::min_element(weights.begin(), weights.end());
    float wmax = *std::max_element(weights.begin(), weights.end());

    printf("  Original size: %.2f MB\n", count * sizeof(float) / (1024.0 * 1024.0));
    printf("  Value range: [%.4f, %.4f]\n", wmin, wmax);

    // Grouped encode
    auto t0 = std::chrono::high_resolution_clock::now();
    auto indices = TopologicalQuantizer::encode_grouped(
        weights.data(), count, group_size, 8, wmin, wmax);
    auto t1 = std::chrono::high_resolution_clock::now();

    double encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t index_size = indices.size() * sizeof(uint32_t);
    float ratio = static_cast<float>(count * sizeof(float)) / index_size;

    printf("  Index count: %zu (%.2f KB)\n", indices.size(), index_size / 1024.0);
    printf("  Compression ratio: %.1fx\n", ratio);
    printf("  Encode time: %.1f ms\n", encode_ms);

    // Decode on GPU
    t0 = std::chrono::high_resolution_clock::now();
    float* d_decoded = TopologicalQuantizer::decode_grouped_gpu(
        indices.data(), static_cast<uint32_t>(indices.size()),
        group_size, 8, wmin, wmax);
    t1 = std::chrono::high_resolution_clock::now();

    double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint32_t decoded_count = static_cast<uint32_t>(indices.size()) * group_size;
    std::vector<float> decoded(decoded_count);
    cudaMemcpy(decoded.data(), d_decoded, decoded_count * sizeof(float),
               cudaMemcpyDeviceToHost);
    TopologicalQuantizer::free_gpu(d_decoded);

    uint32_t cmp_count = std::min(count, decoded_count);
    float mse = compute_mse(weights.data(), decoded.data(), cmp_count);

    printf("  Decode time: %.1f ms\n", decode_ms);
    printf("  MSE: %.6e\n", mse);
}

// Benchmark: DNA Engram / MoE system
static void bench_dna_engram(uint32_t rows, uint32_t cols, uint32_t num_experts) {
    print_separator();
    printf("[Benchmark] DNA Engram System (%u x %u, %u experts)\n",
           rows, cols, num_experts);

    uint32_t total = rows * cols;
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> weights(total);
    for (auto& w : weights) w = dist(rng);

    printf("  Original size: %.2f MB\n", total * sizeof(float) / (1024.0 * 1024.0));

    // Build dictionary
    auto t0 = std::chrono::high_resolution_clock::now();
    auto experts = DNAEngramSystem::build_dictionary(
        weights.data(), rows, cols, num_experts, 16);
    auto t1 = std::chrono::high_resolution_clock::now();

    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto serialized = DNAEngramSystem::serialize(experts);
    printf("  Dictionary size: %.2f KB (%zu experts)\n",
           serialized.size() / 1024.0, experts.size());
    printf("  Build time: %.1f ms\n", build_ms);

    // Simulate input activation and retrieve experts
    std::vector<float> input(cols);
    for (auto& v : input) v = dist(rng);

    t0 = std::chrono::high_resolution_clock::now();
    auto active = DNAEngramSystem::retrieve_experts(
        input.data(), cols, experts.data(),
        static_cast<uint32_t>(experts.size()), 0.01f, 8);
    t1 = std::chrono::high_resolution_clock::now();

    double retrieve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Active experts: %zu / %zu\n", active.size(), experts.size());
    printf("  Retrieval time: %.1f ms\n", retrieve_ms);

    // JIT reconstruct
    if (!active.empty()) {
        t0 = std::chrono::high_resolution_clock::now();
        float* d_weights = DNAEngramSystem::reconstruct_gpu(
            experts.data(), active.data(),
            static_cast<uint32_t>(active.size()), rows, cols);
        t1 = std::chrono::high_resolution_clock::now();

        double recon_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  JIT reconstruction time: %.1f ms\n", recon_ms);
        printf("  Reconstruction throughput: %.2f GB/s\n",
               (total * sizeof(float)) / (recon_ms * 1e6));

        DNAEngramSystem::free_gpu(d_weights);
    }
}

// Full pipeline benchmark
static void bench_full_pipeline(uint32_t rows, uint32_t cols, uint32_t num_layers) {
    print_separator();
    printf("[Benchmark] FULL NFA PIPELINE (%u layers of %u x %u)\n",
           num_layers, rows, cols);

    uint32_t total_per_layer = rows * cols;
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // Generate synthetic model
    std::vector<std::vector<float>> all_weights(num_layers);
    std::vector<std::pair<const float*, std::pair<uint32_t, uint32_t>>> layer_ptrs;

    for (uint32_t l = 0; l < num_layers; ++l) {
        all_weights[l].resize(total_per_layer);
        for (auto& w : all_weights[l]) w = dist(rng);
        layer_ptrs.push_back({all_weights[l].data(), {rows, cols}});
    }

    size_t total_bytes = static_cast<size_t>(num_layers) * total_per_layer * sizeof(float);
    printf("  Original model size: %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    NFAConfig config = default_config();
    config.fractal_iterations = 512;
    config.hilbert_order = 8;
    config.num_micro_experts = 64;
    config.basis_rank = 16;

    NFAEngine engine(config);

    // Compress
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.compress_model(layer_ptrs);
    auto t1 = std::chrono::high_resolution_clock::now();

    double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Compressed size: %.2f MB\n", engine.compressed_bytes() / (1024.0 * 1024.0));
    printf("  Compression ratio: %.1fx\n", engine.compression_ratio());
    printf("  Compression time: %.1f ms\n", compress_ms);

    // Save to disk
    const char* model_path = "/tmp/nfa_model.bin";
    t0 = std::chrono::high_resolution_clock::now();
    engine.save(model_path);
    t1 = std::chrono::high_resolution_clock::now();
    printf("  Save time: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Load from disk
    NFAEngine engine2(config);
    t0 = std::chrono::high_resolution_clock::now();
    engine2.load(model_path);
    t1 = std::chrono::high_resolution_clock::now();
    printf("  Load time: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Expand all layers to VRAM
    t0 = std::chrono::high_resolution_clock::now();
    auto gpu_buffers = engine2.expand_all_gpu();
    t1 = std::chrono::high_resolution_clock::now();

    double expand_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  VRAM expansion time: %.1f ms\n", expand_ms);
    printf("  Expansion throughput: %.2f GB/s\n",
           total_bytes / (expand_ms * 1e6));

    // Verify reconstruction quality
    for (uint32_t l = 0; l < num_layers; ++l) {
        std::vector<float> expanded(total_per_layer);
        cudaMemcpy(expanded.data(), gpu_buffers[l].data,
                   total_per_layer * sizeof(float), cudaMemcpyDeviceToHost);

        float mse = compute_mse(all_weights[l].data(), expanded.data(), total_per_layer);
        printf("  Layer %u MSE: %.6e\n", l, mse);
    }

    engine2.free_all_gpu();
    printf("\n");
}

int main(int argc, char** argv) {
    printf("\n");
    print_separator();
    printf("  NEURAL FRACTAL ANCHORING (NFA) ENGINE\n");
    printf("  Fractal Compression | Topological Quantization | DNA Engrams\n");
    print_separator();
    printf("\n");

    print_gpu_info();
    printf("\n");

    // Individual subsystem benchmarks
    bench_fractal_anchor(512, 512);     // ~1 MB layer
    bench_topological(262144, 16);      // 1M floats grouped by 16
    bench_dna_engram(256, 256, 16);     // Small MoE test

    // Full pipeline: simulate a small model
    // 4 layers of 1024x1024 (~16 MB original)
    bench_full_pipeline(1024, 1024, 4);

    // Larger test: 8 layers of 2048x2048 (~128 MB original)
    bench_full_pipeline(2048, 2048, 8);

    print_separator();
    printf("[Done] All benchmarks completed.\n");
    print_separator();

    return 0;
}
