#include "nfa_types.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace nfa {
namespace kernels {

// Compute activation scores for micro-experts based on input
// Each expert has a "signature" (first basis vector) used for matching
__global__ void expert_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ expert_signatures,
    float* __restrict__ scores,
    uint32_t input_dim,
    uint32_t num_experts,
    uint32_t signature_dim)
{
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_experts) return;

    // Dot product between input and expert signature (cosine similarity)
    float dot = 0.0f;
    float norm_input = 0.0f;
    float norm_sig = 0.0f;

    uint32_t dim = min(input_dim, signature_dim);
    const float* sig = expert_signatures + eid * signature_dim;

    for (uint32_t d = 0; d < dim; ++d) {
        dot += input[d] * sig[d];
        norm_input += input[d] * input[d];
        norm_sig += sig[d] * sig[d];
    }

    float denom = sqrtf(norm_input) * sqrtf(norm_sig);
    scores[eid] = (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

// Low-rank reconstruction kernel: reconstruct weight sub-matrix from basis vectors
// W_expert = sum_r coeff[r] * row_basis[r] (x) col_basis[r]  (tensor/outer product)
__global__ void lowrank_reconstruct_kernel(
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

// Weighted accumulation of multiple expert reconstructions
// output += scale * expert_output  (for each active expert)
__global__ void weighted_expert_accumulate_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_output,
    float weight,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    output[idx] += weight * expert_output[idx];
}

// Normalize output by total expert contribution
__global__ void normalize_kernel(
    float* __restrict__ data,
    float normalizer,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    data[idx] *= normalizer;
}

// Top-K selection: find the K highest activation scores
// Simple parallel reduction for small K
__global__ void topk_indices_kernel(
    const float* __restrict__ scores,
    uint32_t* __restrict__ top_indices,
    float* __restrict__ top_scores,
    uint32_t num_experts,
    uint32_t k,
    float threshold)
{
    // Single-block kernel for simplicity (K is typically small)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Initialize
    for (uint32_t i = 0; i < k; ++i) {
        top_indices[i] = 0xFFFFFFFF;
        top_scores[i] = -1e30f;
    }

    // Linear scan with insertion
    for (uint32_t e = 0; e < num_experts; ++e) {
        float s = scores[e];
        if (s < threshold) continue;

        // Find insertion point
        for (uint32_t i = 0; i < k; ++i) {
            if (s > top_scores[i]) {
                // Shift down
                for (uint32_t j = k - 1; j > i; --j) {
                    top_scores[j] = top_scores[j - 1];
                    top_indices[j] = top_indices[j - 1];
                }
                top_scores[i] = s;
                top_indices[i] = e;
                break;
            }
        }
    }
}

} // namespace kernels

// Host wrappers
void launch_expert_activation(
    const float* d_input, const float* d_signatures, float* d_scores,
    uint32_t input_dim, uint32_t num_experts, uint32_t signature_dim)
{
    uint32_t block = 256;
    uint32_t grid = (num_experts + block - 1) / block;
    kernels::expert_activation_kernel<<<grid, block>>>(
        d_input, d_signatures, d_scores, input_dim, num_experts, signature_dim);
    cudaDeviceSynchronize();
}

void launch_lowrank_reconstruct(
    const float* d_row_basis, const float* d_col_basis,
    const float* d_coefficients, float* d_output,
    uint32_t rows, uint32_t cols, uint32_t rank)
{
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    kernels::lowrank_reconstruct_kernel<<<grid, block>>>(
        d_row_basis, d_col_basis, d_coefficients, d_output, rows, cols, rank);
    cudaDeviceSynchronize();
}

void launch_weighted_accumulate(
    float* d_output, const float* d_expert, float weight, uint32_t count)
{
    uint32_t block = 256;
    uint32_t grid = (count + block - 1) / block;
    kernels::weighted_expert_accumulate_kernel<<<grid, block>>>(
        d_output, d_expert, weight, count);
    cudaDeviceSynchronize();
}

void launch_topk_indices(
    const float* d_scores, uint32_t* d_top_indices, float* d_top_scores,
    uint32_t num_experts, uint32_t k, float threshold)
{
    kernels::topk_indices_kernel<<<1, 1>>>(
        d_scores, d_top_indices, d_top_scores, num_experts, k, threshold);
    cudaDeviceSynchronize();
}

} // namespace nfa
