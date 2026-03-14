#include <metal_stdlib>
using namespace metal;

// Q4_0 block: 32 elements in 18 bytes (2 byte scale + 16 bytes data)
struct block_q4_0 {
    half d;
    uchar qs[16];
};

kernel void dequantize_q4_0(
    device const block_q4_0* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    const uint block_idx = tpig / 32;
    const uint element_in_block = tpig % 32;
    const uint byte_idx = element_in_block / 2;
    const uint nibble_idx = element_in_block % 2;

    device const block_q4_0& block = in[block_idx];
    const uchar byte = block.qs[byte_idx];
    const int nibble = (nibble_idx == 0) ? (byte & 0x0F) : (byte >> 4);
    
    out[tpig] = (float)block.d * (nibble - 8);
}

// IQ4_XS block: 256 elements in 136 bytes
struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

constant float kvalues_iq4nl_f[16] = {
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0
};

kernel void dequantize_iq4_xs(
    device const block_iq4_xs* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]
) {
    const uint block_idx = tpig / 256;
    const uint element_in_block = tpig % 256;
    const uint subblock_idx = element_in_block / 32;
    const uint element_in_subblock = element_in_block % 32;
    
    device const block_iq4_xs& block = in[block_idx];
    
    // Decode 6-bit scale for this subblock
    const uint low_bits = (block.scales_l[subblock_idx / 2] >> (4 * (subblock_idx % 2))) & 0x0F;
    const uint high_bits = (block.scales_h >> (2 * subblock_idx)) & 0x03;
    const int ls = (int)(low_bits | (high_bits << 4));
    
    const float scale = (float)block.d * (ls - 32.0);
    
    const uint qs_offset = subblock_idx * 16 + element_in_subblock / 2;
    const uint nibble_idx = element_in_subblock % 2;
    const uchar byte = block.qs[qs_offset];
    const uint index = (nibble_idx == 0) ? (byte & 0x0F) : (byte >> 4);
    
    out[tpig] = scale * kvalues_iq4nl_f[index];
}

