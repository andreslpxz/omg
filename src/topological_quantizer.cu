#include "topological_quantizer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace nfa {

// Forward declarations of kernel launchers
void launch_hilbert_encode(
    const float* d_weights, uint32_t* d_indices, uint32_t count,
    uint32_t curve_order, float range_min, float range_max);

void launch_hilbert_decode(
    const uint32_t* d_indices, float* d_weights, uint32_t count,
    uint32_t curve_order, float range_min, float range_max);

void launch_hilbert_group_encode(
    const float* d_weights, uint32_t* d_indices, uint32_t count,
    uint32_t group_size, uint32_t curve_order, float range_min, float range_max);

void launch_hilbert_group_decode(
    const uint32_t* d_indices, float* d_weights, uint32_t num_groups,
    uint32_t group_size, uint32_t curve_order, float range_min, float range_max);

// Host-side Hilbert curve utilities
namespace hilbert {

void rotate(uint32_t n, uint32_t* x, uint32_t* y, uint32_t rx, uint32_t ry)
{
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        uint32_t tmp = *x;
        *x = *y;
        *y = tmp;
    }
}

uint64_t coords_to_index(const uint32_t* coords, uint32_t dims, uint32_t order)
{
    // 2D Hilbert curve implementation
    if (dims < 2) return coords[0];

    uint32_t x = coords[0], y = coords[1];
    uint32_t n = 1u << order;
    uint64_t d = 0;

    for (int32_t s = static_cast<int32_t>(n) / 2; s > 0; s /= 2) {
        uint32_t rx = (x & s) > 0 ? 1 : 0;
        uint32_t ry = (y & s) > 0 ? 1 : 0;
        d += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);
        rotate(s, &x, &y, rx, ry);
    }

    return d;
}

void index_to_coords(uint64_t index, uint32_t dims, uint32_t order, uint32_t* coords)
{
    if (dims < 2) {
        coords[0] = static_cast<uint32_t>(index);
        return;
    }

    uint32_t n = 1u << order;
    uint32_t x = 0, y = 0;

    for (uint32_t s = 1; s < n; s *= 2) {
        uint32_t rx = 1 & (static_cast<uint32_t>(index / 2));
        uint32_t ry = 1 & (static_cast<uint32_t>(index) ^ rx);

        rotate(s, &x, &y, rx, ry);

        x += s * rx;
        y += s * ry;
        index /= 4;
    }

    coords[0] = x;
    coords[1] = y;
}

} // namespace hilbert

std::vector<uint32_t> TopologicalQuantizer::encode(
    const float* weights, uint32_t count,
    uint32_t curve_order, float range_min, float range_max)
{
    float* d_weights = nullptr;
    uint32_t* d_indices = nullptr;

    cudaMalloc(&d_weights, count * sizeof(float));
    cudaMalloc(&d_indices, count * sizeof(uint32_t));
    cudaMemcpy(d_weights, weights, count * sizeof(float), cudaMemcpyHostToDevice);

    launch_hilbert_encode(d_weights, d_indices, count,
                          curve_order, range_min, range_max);

    std::vector<uint32_t> indices(count);
    cudaMemcpy(indices.data(), d_indices, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_indices);
    return indices;
}

float* TopologicalQuantizer::decode_gpu(
    const uint32_t* indices, uint32_t count,
    uint32_t curve_order, float range_min, float range_max)
{
    uint32_t* d_indices = nullptr;
    float* d_weights = nullptr;

    cudaMalloc(&d_indices, count * sizeof(uint32_t));
    cudaMalloc(&d_weights, count * sizeof(float));
    cudaMemcpy(d_indices, indices, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    launch_hilbert_decode(d_indices, d_weights, count,
                          curve_order, range_min, range_max);

    cudaFree(d_indices);
    return d_weights;
}

std::vector<uint32_t> TopologicalQuantizer::encode_grouped(
    const float* weights, uint32_t count,
    uint32_t group_size, uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t num_groups = (count + group_size - 1) / group_size;

    float* d_weights = nullptr;
    uint32_t* d_indices = nullptr;

    cudaMalloc(&d_weights, count * sizeof(float));
    cudaMalloc(&d_indices, num_groups * sizeof(uint32_t));
    cudaMemcpy(d_weights, weights, count * sizeof(float), cudaMemcpyHostToDevice);

    launch_hilbert_group_encode(d_weights, d_indices, count,
                                group_size, curve_order, range_min, range_max);

    std::vector<uint32_t> indices(num_groups);
    cudaMemcpy(indices.data(), d_indices, num_groups * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_indices);
    return indices;
}

float* TopologicalQuantizer::decode_grouped_gpu(
    const uint32_t* indices, uint32_t num_groups,
    uint32_t group_size, uint32_t curve_order,
    float range_min, float range_max)
{
    uint32_t* d_indices = nullptr;
    float* d_weights = nullptr;
    uint32_t total = num_groups * group_size;

    cudaMalloc(&d_indices, num_groups * sizeof(uint32_t));
    cudaMalloc(&d_weights, total * sizeof(float));
    cudaMemcpy(d_indices, indices, num_groups * sizeof(uint32_t), cudaMemcpyHostToDevice);

    launch_hilbert_group_decode(d_indices, d_weights, num_groups,
                                group_size, curve_order, range_min, range_max);

    cudaFree(d_indices);
    return d_weights;
}

void TopologicalQuantizer::free_gpu(float* d_weights)
{
    if (d_weights) cudaFree(d_weights);
}

} // namespace nfa
