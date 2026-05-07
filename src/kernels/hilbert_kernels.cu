#include "nfa_types.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace nfa {
namespace kernels {

// Device-side Hilbert curve: convert 2D coordinates to Hilbert index
__device__ uint64_t xy_to_hilbert(uint32_t x, uint32_t y, uint32_t order)
{
    uint64_t d = 0;
    for (int32_t s = static_cast<int32_t>(order) / 2; s > 0; s /= 2) {
        uint32_t rx = (x & s) > 0 ? 1 : 0;
        uint32_t ry = (y & s) > 0 ? 1 : 0;
        d += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);

        // Rotate quadrant
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            uint32_t tmp = x;
            x = y;
            y = tmp;
        }
    }
    return d;
}

// Device-side Hilbert curve: convert Hilbert index to 2D coordinates
__device__ void hilbert_to_xy(uint64_t d, uint32_t order, uint32_t* x, uint32_t* y)
{
    *x = 0;
    *y = 0;
    for (uint32_t s = 1; s < order; s *= 2) {
        uint32_t rx = 1 & (static_cast<uint32_t>(d / 2));
        uint32_t ry = 1 & (static_cast<uint32_t>(d) ^ rx);

        // Rotate
        if (ry == 0) {
            if (rx == 1) {
                *x = s - 1 - *x;
                *y = s - 1 - *y;
            }
            uint32_t tmp = *x;
            *x = *y;
            *y = tmp;
        }

        *x += s * rx;
        *y += s * ry;
        d /= 4;
    }
}

// Encode weights to Hilbert indices
// Maps each weight value to a position on the Hilbert curve
__global__ void hilbert_encode_kernel(
    const float* __restrict__ weights,
    uint32_t* __restrict__ indices,
    uint32_t count,
    uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float w = weights[idx];
    float normalized = (w - range_min) / (range_max - range_min);
    normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);

    uint32_t resolution = 1u << curve_order;
    uint32_t x = static_cast<uint32_t>(normalized * (resolution - 1));
    // Use element position as second coordinate
    uint32_t y = idx % resolution;

    uint64_t hilbert_idx = xy_to_hilbert(x, y, resolution);
    indices[idx] = static_cast<uint32_t>(hilbert_idx & 0xFFFFFFFF);
}

// Decode Hilbert indices back to weight values
__global__ void hilbert_decode_kernel(
    const uint32_t* __restrict__ indices,
    float* __restrict__ weights,
    uint32_t count,
    uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t resolution = 1u << curve_order;
    uint32_t x, y;
    hilbert_to_xy(static_cast<uint64_t>(indices[idx]), resolution, &x, &y);

    float normalized = static_cast<float>(x) / static_cast<float>(resolution - 1);
    weights[idx] = normalized * (range_max - range_min) + range_min;
}

// Group encode: map a group of weights to a single representative Hilbert index
// Uses centroid of the group
__global__ void hilbert_group_encode_kernel(
    const float* __restrict__ weights,
    uint32_t* __restrict__ group_indices,
    uint32_t total_count,
    uint32_t group_size,
    uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_groups = (total_count + group_size - 1) / group_size;
    if (gid >= num_groups) return;

    // Compute centroid of this group
    float centroid = 0.0f;
    uint32_t start = gid * group_size;
    uint32_t end = min(start + group_size, total_count);
    uint32_t actual_size = end - start;

    for (uint32_t i = start; i < end; ++i) {
        centroid += weights[i];
    }
    centroid /= static_cast<float>(actual_size);

    // Compute variance as second coordinate
    float variance = 0.0f;
    for (uint32_t i = start; i < end; ++i) {
        float diff = weights[i] - centroid;
        variance += diff * diff;
    }
    variance /= static_cast<float>(actual_size);

    float norm_centroid = (centroid - range_min) / (range_max - range_min);
    norm_centroid = fminf(fmaxf(norm_centroid, 0.0f), 1.0f);

    float max_var = (range_max - range_min) * (range_max - range_min) * 0.25f;
    float norm_var = fminf(variance / (max_var + 1e-8f), 1.0f);

    uint32_t resolution = 1u << curve_order;
    uint32_t x = static_cast<uint32_t>(norm_centroid * (resolution - 1));
    uint32_t y = static_cast<uint32_t>(norm_var * (resolution - 1));

    group_indices[gid] = static_cast<uint32_t>(
        xy_to_hilbert(x, y, resolution) & 0xFFFFFFFF);
}

// Group decode: reconstruct weight group from a single Hilbert index
// Produces group_size weights by interpolation along the curve
__global__ void hilbert_group_decode_kernel(
    const uint32_t* __restrict__ group_indices,
    float* __restrict__ weights,
    uint32_t num_groups,
    uint32_t group_size,
    uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    uint32_t resolution = 1u << curve_order;
    uint32_t x, y;
    hilbert_to_xy(static_cast<uint64_t>(group_indices[gid]), resolution, &x, &y);

    float centroid = static_cast<float>(x) / static_cast<float>(resolution - 1);
    centroid = centroid * (range_max - range_min) + range_min;

    float norm_var = static_cast<float>(y) / static_cast<float>(resolution - 1);
    float stddev = sqrtf(norm_var * (range_max - range_min) * (range_max - range_min) * 0.25f);

    // Reconstruct group using Gaussian spread around centroid
    uint32_t base = gid * group_size;
    for (uint32_t i = 0; i < group_size; ++i) {
        // Deterministic spread using golden ratio
        float t = static_cast<float>(i) / static_cast<float>(group_size);
        float offset = (t - 0.5f) * 2.0f * stddev;
        weights[base + i] = centroid + offset;
    }
}

} // namespace kernels

// Host wrappers
void launch_hilbert_encode(
    const float* d_weights, uint32_t* d_indices, uint32_t count,
    uint32_t curve_order, float range_min, float range_max)
{
    uint32_t block = 256;
    uint32_t grid = (count + block - 1) / block;
    kernels::hilbert_encode_kernel<<<grid, block>>>(
        d_weights, d_indices, count, curve_order, range_min, range_max);
    cudaDeviceSynchronize();
}

void launch_hilbert_decode(
    const uint32_t* d_indices, float* d_weights, uint32_t count,
    uint32_t curve_order, float range_min, float range_max)
{
    uint32_t block = 256;
    uint32_t grid = (count + block - 1) / block;
    kernels::hilbert_decode_kernel<<<grid, block>>>(
        d_indices, d_weights, count, curve_order, range_min, range_max);
    cudaDeviceSynchronize();
}

void launch_hilbert_group_encode(
    const float* d_weights, uint32_t* d_indices, uint32_t count,
    uint32_t group_size, uint32_t curve_order, float range_min, float range_max)
{
    uint32_t num_groups = (count + group_size - 1) / group_size;
    uint32_t block = 256;
    uint32_t grid = (num_groups + block - 1) / block;
    kernels::hilbert_group_encode_kernel<<<grid, block>>>(
        d_weights, d_indices, count, group_size, curve_order, range_min, range_max);
    cudaDeviceSynchronize();
}

void launch_hilbert_group_decode(
    const uint32_t* d_indices, float* d_weights, uint32_t num_groups,
    uint32_t group_size, uint32_t curve_order, float range_min, float range_max)
{
    uint32_t block = 256;
    uint32_t grid = (num_groups + block - 1) / block;
    kernels::hilbert_group_decode_kernel<<<grid, block>>>(
        d_indices, d_weights, num_groups, group_size, curve_order, range_min, range_max);
    cudaDeviceSynchronize();
}

} // namespace nfa
