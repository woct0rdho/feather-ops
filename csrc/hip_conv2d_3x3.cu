// 3x3, stride 1, padding 1 conv2d for bf16/fp16 NCHW activations: implicit GEMM over a halo tile.
//
// GEMM view: M = C_out, N = H*W (NCHW pixel order), K = 9 * C_in, with the contraction ordered
// k = group * 144 + tap * 16 + (c % 16), group = c / 16, tap = kh * 3 + kw
// so that one 16-channel group's nine taps are the contiguous k range [g*144, g*144 + 144). The
// weight is prepacked to [ceil(9*C_in/8), C_out, 8] in that order (see feather_ops/conv2d.py), which
// makes the per-group weight slice 18 contiguous rows of 16-byte vectors.
//
// The activation operand is not prepacked. Per 16-channel group a block stages a 3 x (kBlockN + 2)
// halo tile as 8-channel planes, and all nine taps read it at a constant 16-byte-granule offset, so
// the input is fetched once per block instead of nine times. A tile row beyond the image is staged
// as zeros, which implements padding=1. A C_in that is a multiple of 8 but not of 16 has a fully
// padding second plane in its last group. That plane re-reads plane 0 and its zero weight rows make
// the contribution exactly nothing.
//
// Each block computes kBlockM output channels x an output row segment of kBlockN columns, so the
// C-shuffle epilogue writes straight into NCHW (row = channel, columns = contiguous pixels).
//
// kRows output rows per block (1 or 2) exist because a 3x3 window needs a 3-row halo for a single
// output row, so the input is read about three times per output row. Two rows per block stage four
// halo rows for two output rows and halve the per-block staging and epilogue work.
// The tap loop then reads halo row (output row + kh), so the B fragments of the second row are the
// same staged tile shifted by one row: two register fragments per tap instead of one, and the
// accumulator count doubles. That only fits where the single-row tile is narrow in M (a 128-wide
// image, or a C_out that cannot fill a 128-wide block), which is exactly where it is enabled.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef NO_PYTORCH
// Thrust emits a gfx1100/gfx1101 clock warning for gfx1151.
// This kernel does not use the affected timing APIs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#pragma clang diagnostic pop
#endif

#include <cstdint>
#include <optional>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWaveSize = 32;
constexpr int kTaps = 9;
constexpr int kChanPerGroup = 16;
constexpr int kChanPerPlane = 8;
// Largest LDS a workgroup may own on gfx11.
constexpr int kMaxLdsBytes = 65536;

// 16-row swizzle used by LDS physical mapping (see csrc/hip_kernel_fp16.cu).
__device__ __forceinline__ constexpr int c_row_logi_to_phys_16(const int x) {
    return ((x & 7) << 1) | ((x >> 3) & 1);
}

// Round-to-nearest-even float -> bf16 bits.
__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7fffu + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

template <bool kIsBf16>
__device__ __forceinline__ uint16_t float_to_bits(const float value) {
    if constexpr (kIsBf16) {
        return float_to_bf16_bits(value);
    } else {
        return __builtin_bit_cast(uint16_t, __float2half_rn(value));
    }
}

template <bool kIsBf16>
__device__ __forceinline__ float bits_to_float(const uint16_t bits) {
    if constexpr (kIsBf16) {
        return __builtin_bit_cast(float, static_cast<uint32_t>(bits) << 16);
    } else {
        return __half2float(__builtin_bit_cast(_Float16, bits));
    }
}

using fp16x16_t = uint16_t __attribute__((ext_vector_type(16)));
using fp32x8_t = float __attribute__((ext_vector_type(8)));

template <bool kIsBf16>
__device__ __forceinline__ fp32x8_t wmma(const fp16x16_t a, const fp16x16_t b, const fp32x8_t c) {
    if constexpr (kIsBf16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
}

template <bool kIsBf16, int kWarpsM, int kWarpsN, int kRepeatM, int kRepeatN, bool kPrefetchGroup = false, int kRows = 1,
          bool kPairLoads = false>
__global__ __launch_bounds__(kWaveSize * kWarpsM * kWarpsN) void conv3x3_kernel(
    const uint16_t* __restrict__ const x,      // [C_in, H, W] of one image
    const uint16_t* __restrict__ const w_pre,  // [(G*144)/8][C_out][8]
    const uint16_t* __restrict__ const bias,   // [C_out] or null
    uint16_t* __restrict__ const y,            // [C_out, H, W] of one image
    const int C_in,
    const int C_out,
    const int H,
    const int W,
    const int has_bias)
{
    constexpr int kBlockM = kWmmaM * kWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kWarpsN * kRepeatN;
    constexpr int kCols = kBlockN + 2;
    constexpr int kHaloRows = kRows + 2;
    constexpr int kPlane = kHaloRows * kCols * kChanPerPlane;
    constexpr int kASize = kTaps * 2 * kBlockM * kChanPerPlane;
    constexpr int kBSize = 2 * kPlane;

    union SharedStorage {
        struct {
            uint16_t a[kASize];
            uint16_t b[kBSize];
        } ab;
        uint16_t c[kWarpsM * kWarpsN][kWmmaM][kWmmaN];
    };
    __shared__ __align__(16) SharedStorage sh;

    const int tid = static_cast<int>(threadIdx.x);
    constexpr int kThreads = kWaveSize * kWarpsM * kWarpsN;
    const int wave_id = tid / kWaveSize;
    const int warp_m = wave_id % kWarpsM;
    const int warp_n = wave_id / kWarpsM;
    const int lane = tid % kWaveSize;
    const int lane_in_subgroup = lane % 16;

    // Partially covering the last rows/columns instead of masking: overlapping tiles recompute the
    // same rows with the same reduction order, so the duplicate stores are bit-identical.
    const int block_m = min(static_cast<int>(blockIdx.y) * kBlockM, C_out - kBlockM);
    const int w0 = min(static_cast<int>(blockIdx.x) * kBlockN, W - kBlockN);
    const int h0 = static_cast<int>(blockIdx.z) * kRows;

    const int64_t hw = static_cast<int64_t>(H) * static_cast<int64_t>(W);

    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    float acc[kRows][kRepeatTiles][8];
    #pragma unroll
    for (int oy = 0; oy < kRows; ++oy) {
    #pragma unroll
    for (int r = 0; r < kRepeatTiles; ++r) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[oy][r][i] = 0.0f;
        }
    }
    }

    // weight staging: one 16-channel group = 9 taps x 2 planes x kBlockM rows
    constexpr int kAVecs = kTaps * 2 * kBlockM;
    constexpr int kAVecsPerThread = (kAVecs + kThreads - 1) / kThreads;
    const auto fetch_a_group = [&](const int64_t w_row, uint4 (&dst)[kAVecsPerThread]) -> void {
        #pragma unroll
        for (int v = 0; v < kAVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kAVecs) continue;
            const int tap_k0 = vec / kBlockM;
            const int m = vec - tap_k0 * kBlockM;
            dst[v] = *reinterpret_cast<const uint4*>(w_pre + (w_row + tap_k0) * static_cast<int64_t>(C_out) * kChanPerPlane + static_cast<int64_t>(block_m + m) * kChanPerPlane);
        }
    };
    const auto store_a_group = [&](const uint4 (&src)[kAVecsPerThread]) -> void {
        #pragma unroll
        for (int v = 0; v < kAVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kAVecs) continue;
            const int tap_k0 = vec / kBlockM;
            const int m = vec - tap_k0 * kBlockM;
            *reinterpret_cast<uint4*>(&sh.ab.a[(tap_k0 * kBlockM + m) * kChanPerPlane]) = src[v];
        }
    };
    const auto load_a_group = [&](const int64_t w_row) -> void {
        uint4 staged[kAVecsPerThread];
        fetch_a_group(w_row, staged);
        store_a_group(staged);
    };

    // activation staging: 2 planes x 3 rows x (kBlockN + 2) columns of one 16-channel group
    constexpr int kBVecs = 2 * kHaloRows * kCols;
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;

    const auto fetch_b_group = [&](const int c_base, uint4 (&dst)[kBVecsPerThread]) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kBVecs) continue;
            const int k0 = vec / (kHaloRows * kCols);
            const int rem = vec - k0 * (kHaloRows * kCols);
            const int r = rem / kCols;
            const int c = rem - r * kCols;
            const int h = h0 + r - 1;
            const int w = w0 + c - 1;
            if (h < 0 || h >= H || w < 0 || w >= W) {
                dst[v] = make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int plane_lo = c_base + k0 * kChanPerPlane;
                const int c_plane = (plane_lo + kChanPerPlane <= C_in) ? plane_lo : 0;
                const uint16_t* const src = x + static_cast<int64_t>(c_plane) * hw + static_cast<int64_t>(h) * W + w;
                uint16_t tmp[kChanPerPlane] __attribute__((aligned(16)));
                #pragma unroll
                for (int j = 0; j < kChanPerPlane; ++j) {
                    tmp[j] = src[static_cast<int64_t>(j) * hw];
                }
                dst[v] = *reinterpret_cast<const uint4*>(tmp);
            }
        }
    };
    const auto store_b_group = [&](const uint4 (&src)[kBVecsPerThread]) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kBVecs) continue;
            const int k0 = vec / (kHaloRows * kCols);
            const int rem = vec - k0 * (kHaloRows * kCols);
            const int r = rem / kCols;
            const int c = rem - r * kCols;
            *reinterpret_cast<uint4*>(&sh.ab.b[(k0 * kHaloRows + r) * kCols * kChanPerPlane + c * kChanPerPlane]) = src[v];
        }
    };

    // Pair staging (kPairLoads): columns 1..kBlockN are staged as even-aligned pixel pairs, one
    // 32-bit load per (channel, pixel pair) instead of one 16-bit load per (channel, pixel), i.e.
    // half the VMEM instructions and ~10 fewer VGPRs. The two halo columns stay scalar, and an odd
    // W (only reachable through an explicit tile) falls back to the scalar path.
    constexpr int kPairsPerRow = kBlockN / 2;
    constexpr int kPairVecs = 2 * kHaloRows * kPairsPerRow;
    constexpr int kPairVecsPerThread = (kPairVecs + kThreads - 1) / kThreads;
    constexpr int kEdgeVecs = 2 * kHaloRows * 2;
    constexpr int kEdgeVecsPerThread = (kEdgeVecs + kThreads - 1) / kThreads;

    const auto load_b_pairs = [&](const int c_base) -> void {
        #pragma unroll
        for (int v = 0; v < kPairVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kPairVecs) break;
            const int k0 = vec / (kHaloRows * kPairsPerRow);
            const int rem = vec - k0 * (kHaloRows * kPairsPerRow);
            const int r = rem / kPairsPerRow;
            const int cp = rem - r * kPairsPerRow;
            const int h = h0 + r - 1;
            const int c = 1 + 2 * cp;  // staged column, global w = w0 + c - 1 = w0 + 2 * cp
            uint16_t* const dst = &sh.ab.b[(k0 * kHaloRows + r) * kCols * kChanPerPlane + c * kChanPerPlane];
            if (h < 0 || h >= H) {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(dst + kChanPerPlane) = make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int plane_lo = c_base + k0 * kChanPerPlane;
                const int c_plane = (plane_lo + kChanPerPlane <= C_in) ? plane_lo : 0;
                const uint16_t* const src = x + static_cast<int64_t>(c_plane) * hw + static_cast<int64_t>(h) * W + (w0 + 2 * cp);
                uint16_t lo[kChanPerPlane] __attribute__((aligned(16)));
                uint16_t hi[kChanPerPlane] __attribute__((aligned(16)));
                #pragma unroll
                for (int j = 0; j < kChanPerPlane; ++j) {
                    const uint32_t pair = *reinterpret_cast<const uint32_t*>(src + static_cast<int64_t>(j) * hw);
                    lo[j] = static_cast<uint16_t>(pair & 0xffffu);
                    hi[j] = static_cast<uint16_t>(pair >> 16);
                }
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(lo);
                *reinterpret_cast<uint4*>(dst + kChanPerPlane) = *reinterpret_cast<const uint4*>(hi);
            }
        }
        #pragma unroll
        for (int v = 0; v < kEdgeVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kEdgeVecs) break;
            const int k0 = vec / (kHaloRows * 2);
            const int rem = vec - k0 * (kHaloRows * 2);
            const int r = rem / 2;
            const int c = (rem - r * 2) == 0 ? 0 : (kBlockN + 1);
            const int h = h0 + r - 1;
            const int w = w0 + c - 1;
            uint16_t* const dst = &sh.ab.b[(k0 * kHaloRows + r) * kCols * kChanPerPlane + c * kChanPerPlane];
            if (h < 0 || h >= H || w < 0 || w >= W) {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int plane_lo = c_base + k0 * kChanPerPlane;
                const int c_plane = (plane_lo + kChanPerPlane <= C_in) ? plane_lo : 0;
                const uint16_t* const src = x + static_cast<int64_t>(c_plane) * hw + static_cast<int64_t>(h) * W + w;
                uint16_t tmp[kChanPerPlane] __attribute__((aligned(16)));
                #pragma unroll
                for (int j = 0; j < kChanPerPlane; ++j) {
                    tmp[j] = src[static_cast<int64_t>(j) * hw];
                }
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(tmp);
            }
        }
    };

    const auto load_b_group = [&](const int c_base) -> void {
        if constexpr (kPairLoads) {
            load_b_pairs(c_base);
            return;
        }
        // C_in is padded up to a multiple of 16 in the prepacked weight, and those weight rows are
        // exactly zero. When the last 8-channel plane is entirely padding it re-reads plane 0
        // instead of running off the tensor. 0 * (any finite activation) contributes exactly
        // nothing, so the hot path needs no per-channel masking. Requires C_in % 8 == 0.
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec = tid + v * kThreads;
            if (vec >= kBVecs) continue;
            const int k0 = vec / (kHaloRows * kCols);
            const int rem = vec - k0 * (kHaloRows * kCols);
            const int r = rem / kCols;
            const int c = rem - r * kCols;
            const int h = h0 + r - 1;
            const int w = w0 + c - 1;
            uint16_t* const dst = &sh.ab.b[(k0 * kHaloRows + r) * kCols * kChanPerPlane + c * kChanPerPlane];
            if (h < 0 || h >= H || w < 0 || w >= W) {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int plane_lo = c_base + k0 * kChanPerPlane;
                const int c_plane = (plane_lo + kChanPerPlane <= C_in) ? plane_lo : 0;
                const uint16_t* const src = x + static_cast<int64_t>(c_plane) * hw + static_cast<int64_t>(h) * W + w;
                uint16_t tmp[kChanPerPlane] __attribute__((aligned(16)));
                #pragma unroll
                for (int j = 0; j < kChanPerPlane; ++j) {
                    tmp[j] = src[static_cast<int64_t>(j) * hw];
                }
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(tmp);
            }
        }
    };

    // the nine taps of one 16-channel group
    // One WMMA k-step is 16 channels: the two 8-channel LDS planes of a (tap, n) pair.
    const auto compute_group = [&]() -> void {
        #pragma unroll
        for (int tap = 0; tap < kTaps; ++tap) {
            const int kh = tap / 3;
            const int kw = tap - kh * 3;

            // One B fragment per (output row, repeat_n tile), kept in registers across repeat_m.
            // Output row ``oy`` reads halo row ``oy + kh``, i.e. the same staged tile shifted.
            uint16_t reg_b[kRows][kRepeatN][16] __attribute__((aligned(16)));
            #pragma unroll
            for (int oy = 0; oy < kRows; ++oy) {
                const int plane_off = ((oy + kh) * kCols + kw) * kChanPerPlane;
                #pragma unroll
                for (int rn = 0; rn < kRepeatN; ++rn) {
                    const int tile_n = warp_n + rn * kWarpsN;
                    const int n_col = tile_n * kWmmaN + lane_in_subgroup;
                    const uint16_t* const src = &sh.ab.b[plane_off + n_col * kChanPerPlane];
                    *reinterpret_cast<uint4*>(&reg_b[oy][rn][0]) = *reinterpret_cast<const uint4*>(src);
                    *reinterpret_cast<uint4*>(&reg_b[oy][rn][8]) = *reinterpret_cast<const uint4*>(src + kPlane);
                }
            }

            #pragma unroll
            for (int rm = 0; rm < kRepeatM; ++rm) {
                const int tile_m = warp_m + rm * kWarpsM;
                const int m_row = tile_m * kWmmaM + lane_in_subgroup;
                uint16_t reg_a[16] __attribute__((aligned(16)));
                const uint16_t* const asrc = &sh.ab.a[((tap * 2) * kBlockM + m_row) * kChanPerPlane];
                *reinterpret_cast<uint4*>(&reg_a[0]) = *reinterpret_cast<const uint4*>(asrc);
                *reinterpret_cast<uint4*>(&reg_a[8]) = *reinterpret_cast<const uint4*>(asrc + kBlockM * kChanPerPlane);

                const fp16x16_t a_frag = *reinterpret_cast<const fp16x16_t*>(reg_a);
                #pragma unroll
                for (int oy = 0; oy < kRows; ++oy) {
                    #pragma unroll
                    for (int rn = 0; rn < kRepeatN; ++rn) {
                        const int idx = rm * kRepeatN + rn;
                        const fp16x16_t b_frag = *reinterpret_cast<const fp16x16_t*>(reg_b[oy][rn]);
                        fp32x8_t c_frag = *reinterpret_cast<fp32x8_t*>(&acc[oy][idx][0]);
                        c_frag = wmma<kIsBf16>(a_frag, b_frag, c_frag);
                        *reinterpret_cast<fp32x8_t*>(&acc[oy][idx][0]) = c_frag;
                    }
                }
            }
        }
    };

    const int groups = (C_in + kChanPerGroup - 1) / kChanPerGroup;

    if constexpr (kPrefetchGroup) {
        // Hold the next group in VGPRs across the current group's compute so the refill latency is
        // hidden. The commits still land after the barrier, so the two-barrier structure is kept.
        uint4 staged_a[kAVecsPerThread];
        uint4 staged_b[kBVecsPerThread];
        fetch_a_group(0, staged_a);
        fetch_b_group(0, staged_b);
        store_a_group(staged_a);
        store_b_group(staged_b);
        __syncthreads();
        for (int g = 0; g < groups; ++g) {
            const bool has_next = g + 1 < groups;
            if (has_next) {
                asm volatile("s_setprio 1" ::: "memory");
                fetch_a_group(static_cast<int64_t>(g + 1) * (kTaps * 2), staged_a);
                fetch_b_group((g + 1) * kChanPerGroup, staged_b);
                asm volatile("s_setprio 0" ::: "memory");
            }
            compute_group();
            if (has_next) {
                __syncthreads();
                store_a_group(staged_a);
                store_b_group(staged_b);
                __syncthreads();
            }
        }
    } else {
        load_a_group(0);
        load_b_group(0);
        __syncthreads();

        for (int g = 0; g < groups; ++g) {
            compute_group();
            if (g + 1 < groups) {
                __syncthreads();
                load_a_group(static_cast<int64_t>(g + 1) * (kTaps * 2));
                load_b_group((g + 1) * kChanPerGroup);
                __syncthreads();
            }
        }
    }

    // C-shuffle epilogue: column-major WMMA accumulator -> row-major NCHW stores
    __syncthreads();  // sh.c aliases sh.ab, which other waves may still be reading
    uint16_t* const sh_c = sh.c[wave_id][0];
    const int subgroup = lane / 16;
    const int read_row = lane / 2;
    const int read_row_phys = c_row_logi_to_phys_16(read_row);
    const int read_col_base = (lane % 2) * 8;

    #pragma unroll
    for (int oy = 0; oy < kRows; ++oy) {
        const bool row_live = h0 + oy < H;  // odd H: the last block's spare row is not in the output
        const int64_t row_base = static_cast<int64_t>(h0 + oy) * W;
        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int idx = rm * kRepeatN + rn;
                const int tile_m = warp_m + rm * kWarpsM;
                const int tile_n = warp_n + rn * kWarpsN;
                const int out_m_base = block_m + tile_m * kWmmaM;
                const int out_n_base = w0 + tile_n * kWmmaN;

                const int col = lane_in_subgroup;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row_logi = i * 2 + subgroup;
                    const int row_phys = c_row_logi_to_phys_16(row_logi);
                    const int out_row = out_m_base + row_logi;
                    float value = acc[oy][idx][i];
                    if (has_bias) {
                        value += bits_to_float<kIsBf16>(bias[out_row]);
                    }
                    sh_c[row_phys * kWmmaN + col] = float_to_bits<kIsBf16>(value);
                }

                const int out_row = out_m_base + read_row;
                const int out_col = out_n_base + read_col_base;
                if (row_live) {
                    *reinterpret_cast<uint4*>(y + static_cast<int64_t>(out_row) * hw + row_base + out_col) =
                        *reinterpret_cast<const uint4*>(sh_c + read_row_phys * kWmmaN + read_col_base);
                }
            }
        }
        // sh.c is shared by the whole block, so the next row's pass waits for this one.
        if (oy + 1 < kRows) {
            __syncthreads();
        }
    }
}

template <bool kIsBf16, int kWarpsM, int kWarpsN, int kRepeatM, int kRepeatN, bool kPrefetchGroup = false, int kRows = 1,
          bool kPairLoads = false>
bool launch_one(
    const void* x, const void* wp, const void* bias, void* y,
    int B, int C_in, int C_out, int H, int W, int has_bias,
    hipStream_t stream)
{
    constexpr int kBlockM = kWmmaM * kWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kWarpsN * kRepeatN;
    constexpr int kThreads = kWaveSize * kWarpsM * kWarpsN;
    constexpr int kHaloRows = kRows + 2;
    constexpr int kLdsBytes = (kTaps * 2 * kBlockM * kChanPerPlane + 2 * kHaloRows * (kBlockN + 2) * kChanPerPlane) * 2;
    static_assert(kLdsBytes <= kMaxLdsBytes, "workgroup LDS over budget");
    const dim3 block(kThreads, 1, 1);
    const dim3 grid(static_cast<uint32_t>((W + kBlockN - 1) / kBlockN),
                    static_cast<uint32_t>((C_out + kBlockM - 1) / kBlockM),
                    static_cast<uint32_t>((H + kRows - 1) / kRows));
    const int64_t in_hw = static_cast<int64_t>(H) * static_cast<int64_t>(W) * static_cast<int64_t>(C_in);
    const int64_t out_hw = static_cast<int64_t>(H) * static_cast<int64_t>(W) * static_cast<int64_t>(C_out);
    for (int b = 0; b < B; ++b) {
        hipLaunchKernelGGL(
            (conv3x3_kernel<kIsBf16, kWarpsM, kWarpsN, kRepeatM, kRepeatN, kPrefetchGroup, kRows, kPairLoads>),
            grid, block, 0, stream,
            reinterpret_cast<const uint16_t*>(x) + static_cast<int64_t>(b) * in_hw,
            reinterpret_cast<const uint16_t*>(wp),
            reinterpret_cast<const uint16_t*>(bias),
            reinterpret_cast<uint16_t*>(y) + static_cast<int64_t>(b) * out_hw,
            C_in, C_out, H, W, has_bias);
    }
    return true;
}

// Instantiation list: one entry per supported (kBlockM, kBlockN) tile, first match wins. Tiles are
// ordered widest-first per block_m so a tile lookup is one pass of constant comparisons.
#define FEATHER_CONV_CONFIGS(X, X_PF)                                               \
    X_PF(4, 4, 2, 4)  /* 128 x 256, next-group register prefetch */                 \
    X(4, 4, 2, 4)  /* 128 x 256 */                                                  \
    X(4, 4, 2, 3)  /* 128 x 192 */                                                  \
    X(4, 4, 2, 2)  /* 128 x 128 */                                                  \
    X(4, 2, 2, 3)  /* 128 x  96 */                                                  \
    X(8, 2, 1, 2)  /* 128 x  64 */                                                  \
    X(8, 1, 1, 2)  /* 128 x  32 */                                                  \
    X(3, 8, 2, 2)  /*  96 x 256 */                                                  \
    X(3, 4, 2, 3)  /*  96 x 192 */                                                  \
    X(3, 4, 2, 2)  /*  96 x 128 */                                                  \
    X(3, 3, 2, 2)  /*  96 x  96 */                                                  \
    X(3, 2, 2, 2)  /*  96 x  64 */                                                  \
    X(3, 1, 2, 2)  /*  96 x  32 */                                                  \
    X(2, 8, 2, 2)  /*  64 x 256 */                                                  \
    X(2, 4, 2, 3)  /*  64 x 192 */                                                  \
    X(2, 4, 2, 2)  /*  64 x 128 */                                                  \
    X(2, 2, 2, 3)  /*  64 x  96 */                                                  \
    X(2, 2, 2, 2)  /*  64 x  64 */                                                  \
    X(4, 1, 1, 2)  /*  64 x  32 */                                                  \
    X(1, 8, 3, 2)  /*  48 x 256 */                                                  \
    X(1, 4, 3, 3)  /*  48 x 192 */                                                  \
    X(1, 4, 3, 2)  /*  48 x 128 */                                                  \
    X(1, 2, 3, 3)  /*  48 x  96 */                                                  \
    X(1, 2, 3, 2)  /*  48 x  64 */                                                  \
    X(1, 1, 3, 2)  /*  48 x  32 */                                                  \
    X(2, 8, 1, 2)  /*  32 x 256 */                                                  \
    X(2, 4, 1, 3)  /*  32 x 192 */                                                  \
    X(2, 4, 1, 2)  /*  32 x 128 */                                                  \
    X(2, 3, 1, 2)  /*  32 x  96 */                                                  \
    X(2, 2, 1, 2)  /*  32 x  64 */                                                  \
    X(2, 1, 1, 2)  /*  32 x  32 */                                                  \
    X(1, 8, 1, 2)  /*  16 x 256 */                                                  \
    X(1, 4, 1, 3)  /*  16 x 192 */                                                  \
    X(1, 4, 1, 2)  /*  16 x 128 */                                                  \
    X(1, 3, 1, 2)  /*  16 x  96 */                                                  \
    X(1, 2, 1, 2)  /*  16 x  64 */                                                  \
    X(1, 1, 1, 2)  /*  16 x  32 */

// Two output rows per block: only the tiles where the single-row form is M-narrow (a 128-wide
// image, or a C_out that no 128-wide tile divides) and where the doubled accumulator set still fits
// in 128-192 VGPRs. Everything else keeps one row: at 128x256 the accumulators would not fit.
#define FEATHER_CONV_CONFIGS_2ROWS(X)                                               \
    X(4, 4, 2, 2)  /* 128 x 128 x 2 rows, 16 waves */                               \
    X(2, 8, 2, 2)  /*  64 x 256 x 2 rows, 16 waves */

} // namespace

// block_m / block_n / rows select the tile. The first instantiation with that shape wins. Returns
// 0 on success, 1 if no instantiation exists for the tile or the problem does not fit it.
extern "C" int feather_conv3x3_launch(
    const void* x, const void* wp, const void* bias, void* y,
    int B, int C_in, int C_out, int H, int W, int has_bias,
    int block_m, int block_n, int is_bf16, int rows, void* stream_ptr)
{
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
#define FEATHER_CONV_TRY_2ROWS(WM, WN, RM, RN)                                                                                       \
    if (rows == 2 && block_m == kWmmaM * WM * RM && block_n == kWmmaN * WN * RN) {                                                   \
        if (is_bf16) {                                                                                                               \
            if (launch_one<true, WM, WN, RM, RN, false, 2, true>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0;  \
        } else {                                                                                                                     \
            if (launch_one<false, WM, WN, RM, RN, false, 2, true>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0; \
        }                                                                                                                            \
        return 1;                                                                                                                    \
    }
    if (rows != 1) {
        // The two-row instantiations stage pixel pairs with 32-bit loads, which needs an even W
        // (guaranteed by the tile selector, since block_n >= 32 divides W). An odd W reached
        // through an explicit tile just falls through to the single-row form: same result, since
        // the per-pixel reduction order does not depend on the tiling.
        if ((W & 1) == 0) {
            FEATHER_CONV_CONFIGS_2ROWS(FEATHER_CONV_TRY_2ROWS)
            return 1;
        }
    }
#undef FEATHER_CONV_TRY_2ROWS
#define FEATHER_CONV_TRY(WM, WN, RM, RN)                                                                             \
    if (block_m == kWmmaM * WM * RM && block_n == kWmmaN * WN * RN) {                                                \
        if (is_bf16) {                                                                                               \
            if (launch_one<true, WM, WN, RM, RN>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0;  \
        } else {                                                                                                     \
            if (launch_one<false, WM, WN, RM, RN>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0; \
        }                                                                                                            \
        return 1;                                                                                                    \
    }
    // The next-group prefetch only pays for the 16-wave 128x256 tile with enough groups to hide
    // behind, so it is gated on both.
    const bool use_prefetch = block_n == 256 && block_m == 128 && C_in >= 128;
#define FEATHER_CONV_TRY_PF(WM, WN, RM, RN)                                                                                \
    if (use_prefetch && block_m == kWmmaM * WM * RM && block_n == kWmmaN * WN * RN) {                                      \
        if (is_bf16) {                                                                                                     \
            if (launch_one<true, WM, WN, RM, RN, true>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0;  \
        } else {                                                                                                           \
            if (launch_one<false, WM, WN, RM, RN, true>(x, wp, bias, y, B, C_in, C_out, H, W, has_bias, stream)) return 0; \
        }                                                                                                                  \
        return 1;                                                                                                          \
    }
#define FEATHER_CONV_SKIP(...)
    FEATHER_CONV_CONFIGS(FEATHER_CONV_SKIP, FEATHER_CONV_TRY_PF)
    FEATHER_CONV_CONFIGS(FEATHER_CONV_TRY, FEATHER_CONV_SKIP)
#undef FEATHER_CONV_SKIP
#undef FEATHER_CONV_TRY_PF
#undef FEATHER_CONV_TRY
    return 1;
}

#ifndef NO_PYTORCH
namespace {

void conv2d_3x3(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& w_prepacked,
    const std::optional<torch::stable::Tensor>& bias,
    torch::stable::Tensor& out,
    const int64_t block_m,
    const int64_t block_n,
    const int64_t rows)
{
    STD_TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    STD_TORCH_CHECK(w_prepacked.is_cuda(), "w_prepacked must be a CUDA tensor");
    STD_TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    const auto device_index = x.get_device_index();
    STD_TORCH_CHECK(w_prepacked.get_device_index() == device_index, "w_prepacked must be on the same device as x");
    STD_TORCH_CHECK(out.get_device_index() == device_index, "out must be on the same device as x");

    const auto dtype = x.scalar_type();
    STD_TORCH_CHECK(dtype == torch::stable::ScalarType::Half || dtype == torch::stable::ScalarType::BFloat16,
                    "x must be float16 or bfloat16");
    STD_TORCH_CHECK(w_prepacked.scalar_type() == dtype, "w_prepacked must have x's dtype");
    STD_TORCH_CHECK(out.scalar_type() == dtype, "out must have x's dtype");

    STD_TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    STD_TORCH_CHECK(w_prepacked.dim() == 3, "w_prepacked must be 3D (K/8, C_out, 8)");
    STD_TORCH_CHECK(out.dim() == 4, "out must be 4D NCHW");

    const int64_t B = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t C_out = out.size(1);
    STD_TORCH_CHECK(out.size(0) == B && out.size(2) == H && out.size(3) == W, "out must be [B, C_out, H, W]");
    STD_TORCH_CHECK(w_prepacked.size(1) == C_out, "w_prepacked.shape[1] must equal C_out");
    STD_TORCH_CHECK(w_prepacked.size(2) == 8, "w_prepacked.shape[2] must be 8");
    STD_TORCH_CHECK(w_prepacked.size(0) == ((C_in + 15) / 16) * 18, "w_prepacked.shape[0] must equal ceil(C_in/16)*18");
    STD_TORCH_CHECK(C_in % 8 == 0, "conv2d_3x3 needs a multiple of 8 input channels");

    STD_TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == W && x.stride(1) == H * W && x.stride(0) == C_in * H * W,
                    "x must be contiguous NCHW");
    STD_TORCH_CHECK(out.stride(3) == 1 && out.stride(2) == W && out.stride(1) == H * W && out.stride(0) == C_out * H * W,
                    "out must be contiguous NCHW");
    STD_TORCH_CHECK(w_prepacked.stride(2) == 1 && w_prepacked.stride(1) == 8 && w_prepacked.stride(0) == C_out * 8,
                    "w_prepacked must be contiguous");

    if (bias.has_value()) {
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as x");
        STD_TORCH_CHECK(bias_t.scalar_type() == dtype, "bias must have x's dtype");
        STD_TORCH_CHECK(bias_t.numel() == C_out, "bias must have C_out elements");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const void* const x_ptr = x.const_data_ptr();
    const void* const w_ptr = w_prepacked.const_data_ptr();
    const void* const bias_ptr = bias.has_value() ? bias->const_data_ptr() : nullptr;
    void* const out_ptr = out.mutable_data_ptr();

    const auto is_aligned_16 = [](const void* const p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
    };
    STD_TORCH_CHECK(is_aligned_16(x_ptr), "x data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(w_ptr), "w_prepacked data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(out_ptr), "out data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(bias_ptr == nullptr ? x_ptr : bias_ptr), "bias data pointer must be 16-byte aligned");

    const bool launched = feather_conv3x3_launch(
        x_ptr, w_ptr, bias_ptr, out_ptr,
        static_cast<int>(B), static_cast<int>(C_in), static_cast<int>(C_out),
        static_cast<int>(H), static_cast<int>(W),
        bias.has_value() ? 1 : 0,
        static_cast<int>(block_m), static_cast<int>(block_n),
        dtype == torch::stable::ScalarType::BFloat16 ? 1 : 0,
        static_cast<int>(rows), stream) == 0;
    STD_TORCH_CHECK(launched, "conv2d_3x3: unsupported tile ", block_m, "x", block_n, " rows ", rows);

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "conv2d_3x3 kernel launch failed: ", hipGetErrorString(launch_err));
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(feather_ops, m)
{
    m.def(
        "conv2d_3x3("
        "Tensor x, "
        "Tensor w_prepacked, "
        "Tensor? bias, "
        "Tensor(a!) out, "
        "int block_m, "
        "int block_n, "
        "int rows"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("conv2d_3x3", TORCH_BOX(&conv2d_3x3));
}
#endif
