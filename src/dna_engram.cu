#include "dna_engram.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

namespace nfa {

// Forward declarations of kernel launchers
void launch_expert_activation(
    const float* d_input, const float* d_signatures, float* d_scores,
    uint32_t input_dim, uint32_t num_experts, uint32_t signature_dim);

void launch_lowrank_reconstruct(
    const float* d_row_basis, const float* d_col_basis,
    const float* d_coefficients, float* d_output,
    uint32_t rows, uint32_t cols, uint32_t rank);

void launch_weighted_accumulate(
    float* d_output, const float* d_expert, float weight, uint32_t count);

void launch_topk_indices(
    const float* d_scores, uint32_t* d_top_indices, float* d_top_scores,
    uint32_t num_experts, uint32_t k, float threshold);

// Simple SVD-like decomposition for low-rank approximation
// Uses randomized power iteration method
static void randomized_lowrank(
    const float* matrix, uint32_t rows, uint32_t cols, uint32_t rank,
    std::vector<float>& U, std::vector<float>& V, std::vector<float>& sigma)
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    rank = std::min(rank, std::min(rows, cols));
    U.resize(rank * rows, 0.0f);
    V.resize(rank * cols, 0.0f);
    sigma.resize(rank, 0.0f);

    // Randomized range finder
    std::vector<float> omega(cols * rank);
    for (auto& v : omega) v = normal(rng);

    // Y = A * Omega  (rows x rank)
    std::vector<float> Y(rows * rank, 0.0f);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < rank; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < cols; ++k) {
                sum += matrix[i * cols + k] * omega[k * rank + j];
            }
            Y[i * rank + j] = sum;
        }
    }

    // Power iteration for better approximation
    std::vector<float> Z(cols * rank, 0.0f);
    for (int power = 0; power < 2; ++power) {
        // Z = A^T * Y
        std::fill(Z.begin(), Z.end(), 0.0f);
        for (uint32_t i = 0; i < cols; ++i) {
            for (uint32_t j = 0; j < rank; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < rows; ++k) {
                    sum += matrix[k * cols + i] * Y[k * rank + j];
                }
                Z[i * rank + j] = sum;
            }
        }
        // Y = A * Z
        std::fill(Y.begin(), Y.end(), 0.0f);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < rank; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < cols; ++k) {
                    sum += matrix[i * cols + k] * Z[k * rank + j];
                }
                Y[i * rank + j] = sum;
            }
        }
    }

    // QR-like orthogonalization of Y (simplified Gram-Schmidt)
    for (uint32_t j = 0; j < rank; ++j) {
        // Orthogonalize against previous columns
        for (uint32_t k = 0; k < j; ++k) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < rows; ++i) {
                dot += Y[i * rank + j] * U[k * rows + i];
            }
            for (uint32_t i = 0; i < rows; ++i) {
                Y[i * rank + j] -= dot * U[k * rows + i];
            }
        }
        // Normalize
        float norm = 0.0f;
        for (uint32_t i = 0; i < rows; ++i) {
            norm += Y[i * rank + j] * Y[i * rank + j];
        }
        norm = std::sqrt(norm);
        sigma[j] = norm;
        if (norm > 1e-8f) {
            for (uint32_t i = 0; i < rows; ++i) {
                U[j * rows + i] = Y[i * rank + j] / norm;
            }
        }
    }

    // Compute V = A^T * U / sigma
    for (uint32_t j = 0; j < rank; ++j) {
        if (sigma[j] < 1e-8f) continue;
        for (uint32_t i = 0; i < cols; ++i) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < rows; ++k) {
                sum += matrix[k * cols + i] * U[j * rows + k];
            }
            V[j * cols + i] = sum / sigma[j];
        }
    }
}

std::vector<DNAEngramSystem::MicroExpert> DNAEngramSystem::build_dictionary(
    const float* weights, uint32_t rows, uint32_t cols,
    uint32_t num_experts, uint32_t basis_rank)
{
    std::vector<MicroExpert> experts;
    experts.reserve(num_experts);

    // Partition the weight matrix into blocks assigned to experts
    uint32_t block_rows = std::max(1u, rows / static_cast<uint32_t>(std::sqrt(num_experts)));
    uint32_t block_cols = std::max(1u, cols / static_cast<uint32_t>(std::sqrt(num_experts)));

    uint32_t expert_id = 0;
    for (uint32_t br = 0; br < rows && expert_id < num_experts; br += block_rows) {
        for (uint32_t bc = 0; bc < cols && expert_id < num_experts; bc += block_cols) {
            uint32_t er = std::min(block_rows, rows - br);
            uint32_t ec = std::min(block_cols, cols - bc);

            // Extract sub-matrix
            std::vector<float> sub(er * ec);
            for (uint32_t i = 0; i < er; ++i) {
                for (uint32_t j = 0; j < ec; ++j) {
                    sub[i * ec + j] = weights[(br + i) * cols + (bc + j)];
                }
            }

            // Low-rank decomposition of this block
            uint32_t actual_rank = std::min(basis_rank, std::min(er, ec));
            std::vector<float> U, V, sigma;
            randomized_lowrank(sub.data(), er, ec, actual_rank, U, V, sigma);

            MicroExpert expert;
            expert.expert_id = expert_id;
            expert.input_dim = ec;
            expert.output_dim = er;
            expert.activation_threshold = 0.01f;
            expert.num_basis = actual_rank;

            // Store basis vectors
            expert.row_basis.resize(actual_rank);
            expert.col_basis.resize(actual_rank);
            expert.coefficients = sigma;

            for (uint32_t r = 0; r < actual_rank; ++r) {
                expert.row_basis[r].dim = er;
                expert.row_basis[r].data.resize(er);
                std::memcpy(expert.row_basis[r].data.data(),
                           &U[r * er], er * sizeof(float));

                expert.col_basis[r].dim = ec;
                expert.col_basis[r].data.resize(ec);
                std::memcpy(expert.col_basis[r].data.data(),
                           &V[r * ec], ec * sizeof(float));
            }

            experts.push_back(std::move(expert));
            ++expert_id;
        }
    }

    return experts;
}

std::vector<uint32_t> DNAEngramSystem::retrieve_experts(
    const float* input_activation, uint32_t input_dim,
    const MicroExpert* experts, uint32_t num_experts,
    float sparsity_threshold, uint32_t max_active)
{
    // Compute activation scores on GPU
    // First, build signature matrix from first basis of each expert
    uint32_t sig_dim = 0;
    for (uint32_t i = 0; i < num_experts; ++i) {
        if (!experts[i].col_basis.empty()) {
            sig_dim = std::max(sig_dim, experts[i].col_basis[0].dim);
        }
    }
    if (sig_dim == 0) sig_dim = input_dim;

    std::vector<float> signatures(num_experts * sig_dim, 0.0f);
    for (uint32_t i = 0; i < num_experts; ++i) {
        if (!experts[i].col_basis.empty()) {
            uint32_t dim = std::min(sig_dim, experts[i].col_basis[0].dim);
            std::memcpy(&signatures[i * sig_dim],
                       experts[i].col_basis[0].data.data(),
                       dim * sizeof(float));
        }
    }

    // GPU-accelerated scoring
    float* d_input = nullptr;
    float* d_sigs = nullptr;
    float* d_scores = nullptr;
    uint32_t* d_top_ids = nullptr;
    float* d_top_scores = nullptr;

    cudaMalloc(&d_input, input_dim * sizeof(float));
    cudaMalloc(&d_sigs, signatures.size() * sizeof(float));
    cudaMalloc(&d_scores, num_experts * sizeof(float));
    cudaMalloc(&d_top_ids, max_active * sizeof(uint32_t));
    cudaMalloc(&d_top_scores, max_active * sizeof(float));

    cudaMemcpy(d_input, input_activation, input_dim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigs, signatures.data(), signatures.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    launch_expert_activation(d_input, d_sigs, d_scores,
                             input_dim, num_experts, sig_dim);

    launch_topk_indices(d_scores, d_top_ids, d_top_scores,
                        num_experts, max_active, sparsity_threshold);

    std::vector<uint32_t> top_ids(max_active);
    cudaMemcpy(top_ids.data(), d_top_ids, max_active * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_sigs);
    cudaFree(d_scores);
    cudaFree(d_top_ids);
    cudaFree(d_top_scores);

    // Filter invalid indices
    std::vector<uint32_t> active;
    for (auto id : top_ids) {
        if (id < num_experts) active.push_back(id);
    }
    return active;
}

float* DNAEngramSystem::reconstruct_gpu(
    const MicroExpert* experts, const uint32_t* active_ids,
    uint32_t num_active, uint32_t rows, uint32_t cols)
{
    float* d_output = nullptr;
    size_t total = static_cast<size_t>(rows) * cols;
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMemset(d_output, 0, total * sizeof(float));

    float* d_expert_out = nullptr;
    cudaMalloc(&d_expert_out, total * sizeof(float));

    for (uint32_t a = 0; a < num_active; ++a) {
        const MicroExpert& exp = experts[active_ids[a]];
        if (exp.row_basis.empty() || exp.col_basis.empty()) continue;

        uint32_t rank = std::min(static_cast<uint32_t>(exp.coefficients.size()),
                                  exp.num_basis);
        if (rank == 0) continue;

        uint32_t er = exp.output_dim;
        uint32_t ec = exp.input_dim;

        // Flatten basis vectors for GPU
        std::vector<float> flat_row(rank * er, 0.0f);
        std::vector<float> flat_col(rank * ec, 0.0f);

        for (uint32_t r = 0; r < rank; ++r) {
            if (r < exp.row_basis.size()) {
                uint32_t dim = std::min(er, exp.row_basis[r].dim);
                std::memcpy(&flat_row[r * er],
                           exp.row_basis[r].data.data(),
                           dim * sizeof(float));
            }
            if (r < exp.col_basis.size()) {
                uint32_t dim = std::min(ec, exp.col_basis[r].dim);
                std::memcpy(&flat_col[r * ec],
                           exp.col_basis[r].data.data(),
                           dim * sizeof(float));
            }
        }

        float* d_row = nullptr;
        float* d_col = nullptr;
        float* d_coeff = nullptr;

        cudaMalloc(&d_row, flat_row.size() * sizeof(float));
        cudaMalloc(&d_col, flat_col.size() * sizeof(float));
        cudaMalloc(&d_coeff, rank * sizeof(float));

        cudaMemcpy(d_row, flat_row.data(), flat_row.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, flat_col.data(), flat_col.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_coeff, exp.coefficients.data(), rank * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemset(d_expert_out, 0, total * sizeof(float));

        // Reconstruct this expert's contribution via tensor product
        launch_lowrank_reconstruct(d_row, d_col, d_coeff,
                                    d_expert_out, er, ec, rank);

        // Accumulate into output (weight = 1/num_active for averaging)
        float weight = 1.0f / static_cast<float>(num_active);
        launch_weighted_accumulate(d_output, d_expert_out, weight, total);

        cudaFree(d_row);
        cudaFree(d_col);
        cudaFree(d_coeff);
    }

    cudaFree(d_expert_out);
    return d_output;
}

std::vector<uint8_t> DNAEngramSystem::serialize(
    const std::vector<MicroExpert>& experts)
{
    std::vector<uint8_t> buffer;

    // Header: number of experts
    uint32_t num = static_cast<uint32_t>(experts.size());
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&num),
                  reinterpret_cast<uint8_t*>(&num) + sizeof(num));

    for (const auto& exp : experts) {
        // Expert header
        buffer.insert(buffer.end(),
                      reinterpret_cast<const uint8_t*>(&exp.expert_id),
                      reinterpret_cast<const uint8_t*>(&exp.expert_id) + sizeof(uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const uint8_t*>(&exp.input_dim),
                      reinterpret_cast<const uint8_t*>(&exp.input_dim) + sizeof(uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const uint8_t*>(&exp.output_dim),
                      reinterpret_cast<const uint8_t*>(&exp.output_dim) + sizeof(uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const uint8_t*>(&exp.activation_threshold),
                      reinterpret_cast<const uint8_t*>(&exp.activation_threshold) + sizeof(float));

        uint32_t num_basis = exp.num_basis;
        buffer.insert(buffer.end(),
                      reinterpret_cast<uint8_t*>(&num_basis),
                      reinterpret_cast<uint8_t*>(&num_basis) + sizeof(uint32_t));

        // Coefficients
        uint32_t nc = static_cast<uint32_t>(exp.coefficients.size());
        buffer.insert(buffer.end(),
                      reinterpret_cast<uint8_t*>(&nc),
                      reinterpret_cast<uint8_t*>(&nc) + sizeof(uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const uint8_t*>(exp.coefficients.data()),
                      reinterpret_cast<const uint8_t*>(exp.coefficients.data()) + nc * sizeof(float));

        // Row basis
        for (uint32_t r = 0; r < num_basis && r < exp.row_basis.size(); ++r) {
            uint32_t dim = exp.row_basis[r].dim;
            buffer.insert(buffer.end(),
                          reinterpret_cast<uint8_t*>(&dim),
                          reinterpret_cast<uint8_t*>(&dim) + sizeof(uint32_t));
            buffer.insert(buffer.end(),
                          reinterpret_cast<const uint8_t*>(exp.row_basis[r].data.data()),
                          reinterpret_cast<const uint8_t*>(exp.row_basis[r].data.data()) +
                          dim * sizeof(float));
        }

        // Col basis
        for (uint32_t r = 0; r < num_basis && r < exp.col_basis.size(); ++r) {
            uint32_t dim = exp.col_basis[r].dim;
            buffer.insert(buffer.end(),
                          reinterpret_cast<uint8_t*>(&dim),
                          reinterpret_cast<uint8_t*>(&dim) + sizeof(uint32_t));
            buffer.insert(buffer.end(),
                          reinterpret_cast<const uint8_t*>(exp.col_basis[r].data.data()),
                          reinterpret_cast<const uint8_t*>(exp.col_basis[r].data.data()) +
                          dim * sizeof(float));
        }
    }

    return buffer;
}

std::vector<DNAEngramSystem::MicroExpert> DNAEngramSystem::deserialize(
    const uint8_t* data, size_t size)
{
    std::vector<MicroExpert> experts;
    size_t offset = 0;

    if (offset + sizeof(uint32_t) > size) return experts;
    uint32_t num;
    std::memcpy(&num, data + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    for (uint32_t e = 0; e < num && offset < size; ++e) {
        MicroExpert exp;

        std::memcpy(&exp.expert_id, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&exp.input_dim, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&exp.output_dim, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&exp.activation_threshold, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&exp.num_basis, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        uint32_t nc;
        std::memcpy(&nc, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        exp.coefficients.resize(nc);
        std::memcpy(exp.coefficients.data(), data + offset, nc * sizeof(float));
        offset += nc * sizeof(float);

        exp.row_basis.resize(exp.num_basis);
        for (uint32_t r = 0; r < exp.num_basis; ++r) {
            uint32_t dim;
            std::memcpy(&dim, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            exp.row_basis[r].dim = dim;
            exp.row_basis[r].data.resize(dim);
            std::memcpy(exp.row_basis[r].data.data(), data + offset, dim * sizeof(float));
            offset += dim * sizeof(float);
        }

        exp.col_basis.resize(exp.num_basis);
        for (uint32_t r = 0; r < exp.num_basis; ++r) {
            uint32_t dim;
            std::memcpy(&dim, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            exp.col_basis[r].dim = dim;
            exp.col_basis[r].data.resize(dim);
            std::memcpy(exp.col_basis[r].data.data(), data + offset, dim * sizeof(float));
            offset += dim * sizeof(float);
        }

        experts.push_back(std::move(exp));
    }

    return experts;
}

void DNAEngramSystem::free_gpu(float* d_weights)
{
    if (d_weights) cudaFree(d_weights);
}

} // namespace nfa
