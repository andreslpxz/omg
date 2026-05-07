#pragma once

#include "nfa_types.h"
#include <vector>
#include <cstdint>

namespace nfa {

// Hilbert curve coordinate-based topological quantization
class TopologicalQuantizer {
public:
    // Encode weights as Hilbert curve indices
    static std::vector<uint32_t> encode(
        const float* weights, uint32_t count,
        uint32_t curve_order, float range_min, float range_max);

    // Decode Hilbert indices back to weight values on GPU
    static float* decode_gpu(
        const uint32_t* indices, uint32_t count,
        uint32_t curve_order, float range_min, float range_max);

    // Group-encode: map groups of weights to single Hilbert index
    static std::vector<uint32_t> encode_grouped(
        const float* weights, uint32_t count,
        uint32_t group_size, uint32_t curve_order,
        float range_min, float range_max);

    // Group-decode on GPU: reconstruct weight groups from indices
    static float* decode_grouped_gpu(
        const uint32_t* indices, uint32_t num_groups,
        uint32_t group_size, uint32_t curve_order,
        float range_min, float range_max);

    static void free_gpu(float* d_weights);
};

// Hilbert curve utilities (host-side)
namespace hilbert {
    // Convert N-D coordinates to Hilbert index
    uint64_t coords_to_index(const uint32_t* coords, uint32_t dims, uint32_t order);
    // Convert Hilbert index to N-D coordinates
    void index_to_coords(uint64_t index, uint32_t dims, uint32_t order, uint32_t* coords);
    // Rotate/flip operations for Hilbert curve construction
    void rotate(uint32_t n, uint32_t* coords, uint32_t rx, uint32_t ry);
}

} // namespace nfa
