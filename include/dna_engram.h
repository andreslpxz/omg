#pragma once

#include "nfa_types.h"
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace nfa {

// Activation dictionary for JIT weight reconstruction (DNA Engram / Dynamic MoE)
class DNAEngramSystem {
public:
    struct BasisVector {
        std::vector<float> data;
        uint32_t dim;
    };

    struct MicroExpert {
        uint32_t expert_id;
        uint32_t input_dim;
        uint32_t output_dim;
        std::vector<BasisVector> row_basis;   // low-rank row basis
        std::vector<BasisVector> col_basis;   // low-rank column basis
        std::vector<float> coefficients;      // mixing coefficients
        float activation_threshold;
    };

    // Build activation dictionary from a full weight set
    static std::vector<MicroExpert> build_dictionary(
        const float* weights, uint32_t rows, uint32_t cols,
        uint32_t num_experts, uint32_t basis_rank);

    // Semantic retrieval: select relevant experts based on input activation
    static std::vector<uint32_t> retrieve_experts(
        const float* input_activation, uint32_t input_dim,
        const MicroExpert* experts, uint32_t num_experts,
        float sparsity_threshold, uint32_t max_active);

    // JIT weight reconstruction on GPU: tensor product of selected expert bases
    static float* reconstruct_gpu(
        const MicroExpert* experts, const uint32_t* active_ids,
        uint32_t num_active, uint32_t rows, uint32_t cols);

    // Serialize dictionary to compact binary format
    static std::vector<uint8_t> serialize(
        const std::vector<MicroExpert>& experts);

    // Deserialize from binary
    static std::vector<MicroExpert> deserialize(
        const uint8_t* data, size_t size);

    static void free_gpu(float* d_weights);
};

} // namespace nfa
