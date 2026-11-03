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
#include <type_traits>

// Int8 GEMM over a prepacked weight, for the two matrix shapes of a training step:
//
//   forward   C[M, N] = A[M, K] @ W[N, K]^T     (contracts the weight's input axis)
//   backward  C[M, K] = A[M, N] @ W[N, K]       (contracts the weight's output axis, so no transposed copy)
//
// Both read the same single copy of the weight, prepacked as 4x4 byte micro-tiles
//
//   P[(k // 4), (n // 4), (n % 4) * 4 + (k % 4)]
//
// which is a pure permutation of the [N, K] int8 bytes. A gfx11 int8 WMMA fragment is the whole 16-byte
// contraction step of one lane's row, so the forward needs 16 contiguous k of one n (four dwords of four
// consecutive k4 tiles) and the backward needs 16 contiguous n of one k (four byte-i gathers of four
// consecutive n4 tiles). Either direction can therefore be served from one copy: the one whose fragment is
// not contiguous along a dword pays three v_perm_b32 per 16-byte chunk on the staging path, not per MMA.
//
// Everything else (LDS staging, register tilings, C-shuffle epilogue) follows csrc/hip_kernel.cu, the fp8
// kernel, which is the same dataflow with a fp8 -> bf16 widening that int8 does not need.

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
constexpr int kWaveSize = 32;

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// Round-to-nearest-even float -> bf16 bits. The fp8 kernel truncates because its values come from an
// fp8 widening. Here the scale multiply is a real rounding step, so match the reference kernels (RNE).
__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7fffu + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

template <bool kOutputFp16>
__device__ __forceinline__ uint16_t float_to_output_bits(const float value) {
    if constexpr (kOutputFp16) {
        return __builtin_bit_cast(uint16_t, __float2half_rn(value));
    } else {
        return float_to_bf16_bits(value);
    }
}

template <bool kOutputFp16>
__device__ __forceinline__ float output_bits_to_float(const uint16_t bits) {
    if constexpr (kOutputFp16) {
        return __half2float(__builtin_bit_cast(_Float16, bits));
    } else {
        const uint32_t expanded = static_cast<uint32_t>(bits) << 16;
        return __builtin_bit_cast(float, expanded);
    }
}

// Bytes {i, i+4, i+8, i+12} of a 16-byte micro-tile, i.e. the four n of one k. Three v_perm_b32.
// __byte_perm(x, y, s) selects destination byte i from {x bytes 0-3, y bytes 0-3} with nibble i of s.
__device__ __forceinline__ uint32_t gather_bytes(const uint32_t d0, const uint32_t d1, const uint32_t d2, const uint32_t d3, const int i) {
    const uint32_t sel = static_cast<uint32_t>(i) | (static_cast<uint32_t>(4 + i) << 4);
    const uint32_t t01 = __byte_perm(d0, d1, sel);
    const uint32_t t23 = __byte_perm(d2, d3, sel);
    return __byte_perm(t01, t23, 0x5410u);
}

template <int kWarpsM, int kWarpsN, int kUnrollK, int kRepeatM, int kRepeatN, bool kTransposedB, bool kOutputFp16>
__global__ __launch_bounds__(kWaveSize * kWarpsM * kWarpsN) void int8_scaled_mm_kernel(
    const int8_t* __restrict__ const a,       // [M, Kc], contraction axis contiguous
    const uint8_t* __restrict__ const b,      // prepacked, [weight_in / 4][weight_out / 4][4][4]
    const float* __restrict__ const a_scale,  // [M] per-row activation scale
    const float* __restrict__ const b_scale,  // [No] per-output-channel weight scale (forward only)
    const uint16_t* __restrict__ const bias,  // [No], output dtype (forward only)
    uint16_t* __restrict__ const c,           // [M, No]
    const int M,
    const int N_out,
    const int K_contract,
    const int b_stride_kc,  // micro-tile chunks per contraction step of 4
    const int b_stride_out, // micro-tile chunks per output step of 4
    const int m_base)       // first row of this launch (0, or the first row after the full tiles)
{
    constexpr int kBlockM = kWmmaM * kWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kWarpsN * kRepeatN;
    constexpr int kChunkK = kWmmaK * kUnrollK;
    constexpr int kThreads = kWaveSize * kWarpsM * kWarpsN;
    constexpr int kWarps = kWarpsM * kWarpsN;

    static_assert(kChunkK % 4 == 0, "chunk_k must be a multiple of the micro-tile size");
    static_assert(kBlockN % 4 == 0, "block_n must be a multiple of the micro-tile size");

    // LDS: A rows are one 16-byte contraction step per (m, stage). B rows are one 16-byte step of the
    // fragment's contiguous axis per output row. The epilogue reuses the same bytes for the C shuffle.
    // Two chunks are resident, so the store of the chunk being prefetched never touches the one under
    // computation and one barrier per chunk suffices.
    // Each stage gets 16 bytes of padding: a lane group that spans stages must not land on the same banks,
    // and the stage stride would otherwise be a multiple of the 128-byte bank period.
    constexpr int kAStageStride = kBlockM * kWmmaK + 16;
    constexpr int kBStageStride = kBlockN * kWmmaK + 16;
    union SharedStorage {
        struct {
            int8_t a[2][kUnrollK * kAStageStride];
            int8_t b[2][kUnrollK * kBStageStride];
        } ab;
        uint16_t c[kWarps][kWmmaM][kWmmaN];
    };
    __shared__ __align__(16) SharedStorage sh;

    // Traversal: output tiles fastest (blockIdx.x), so all output tiles of one M tile run together and its A
    // panel stays in L2. Grouping M tiles instead (to reuse each B panel) measured within noise on every cell.
    const int block_m = m_base + static_cast<int>(blockIdx.y) * kBlockM;
    const int block_n = static_cast<int>(blockIdx.x) * kBlockN;

    const int tid = static_cast<int>(threadIdx.x);
    const int wave_id = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_in_subgroup = lane % 16;
    const int warp_m = wave_id % kWarpsM;
    const int warp_n = wave_id / kWarpsM;
    const int half_wave = lane / 16;

    // Fragment ownership on gfx11: lane holds row/column (lane % 16) of its 16x16 tile and the two
    // half-waves are duplicated, so the accumulator's element e is row 2e + (lane / 16), column lane % 16.
    const auto acc_row = [&](const int e) -> int { return 2 * e + half_wave; };

    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    v8i acc[kRepeatTiles];
#pragma unroll
    for (int r = 0; r < kRepeatTiles; ++r) {
        acc[r] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    }

    // Staging plan, hoisted out of the chunk loop: every thread owns a fixed set of 128-bit rows whose
    // addresses advance by one constant per chunk. Both operand streams are assigned so that a warp's
    // lanes cover the contiguous axis of their layout first:
    // A: lanes walk the k-steps of a row (2 lanes per 32-byte row slice at chunk_k = 32), so a warp
    //    touches chunk_k/16 + 1 cache lines instead of one per lane.
    // B: lanes walk the axis that the prepacked layout stores contiguously (the weight's output axis in
    //    the forward, its input axis in the backward), i.e. one 512-byte span per warp instruction.
    constexpr int kAVecs = kUnrollK * kBlockM;
    constexpr int kAVecsPerThread = (kAVecs + kThreads - 1) / kThreads;
    constexpr int kBChunksPerRow = kBlockN / 4;  // micro-tiles per output row of the block
    constexpr int kCChunksPerRow = kChunkK / 4;  // micro-tiles per contraction step of the block
    constexpr int kBCount = kCChunksPerRow * kBChunksPerRow;
    constexpr int kBChunksPerThread = (kBCount + kThreads - 1) / kThreads;

    struct Stage {
        int4 a[kAVecsPerThread];
        uint4 b[kBChunksPerThread];
    };
    Stage stage;

    int a_valid[kAVecsPerThread];
    int a_stage_of[kAVecsPerThread];
    int8_t* a_lds[2][kAVecsPerThread];
    int64_t a_off[kAVecsPerThread];
    const int64_t a_row_stride = K_contract;
#pragma unroll
    for (int v = 0; v < kAVecsPerThread; ++v) {
        const int vec_idx = tid + v * kThreads;
        const int stage_idx = vec_idx % kUnrollK;
        const int m_row = min(vec_idx / kUnrollK, kBlockM - 1);  // idle threads keep an in-range row
        a_valid[v] = vec_idx < kAVecs ? 1 : 0;
        a_stage_of[v] = stage_idx;
        const int a_row = min(block_m + m_row, M - 1);  // rows past M only feed unstored outputs
        a_off[v] = static_cast<int64_t>(a_row) * a_row_stride + stage_idx * kWmmaK;
#pragma unroll
        for (int buf = 0; buf < 2; ++buf) {
            a_lds[buf][v] = &sh.ab.a[buf][stage_idx * kAStageStride + m_row * kWmmaK];
        }
    }

    int b_c_of[kBChunksPerThread];
    int b_valid_of[kBChunksPerThread];
    int8_t* b_lds[2][kBChunksPerThread];
    int64_t b_off[kBChunksPerThread];
    const int last_out_tile = max(N_out / 4 - 1, 0);
#pragma unroll
    for (int v = 0; v < kBChunksPerThread; ++v) {
        const int chunk_idx = tid + v * kThreads;
        int c_local;
        int o_local;
        if constexpr (kTransposedB) {
            c_local = chunk_idx % kCChunksPerRow;
            o_local = chunk_idx / kCChunksPerRow;
        } else {
            c_local = chunk_idx / kBChunksPerRow;
            o_local = chunk_idx % kBChunksPerRow;
        }
        c_local = min(c_local, kCChunksPerRow - 1);  // idle threads keep in-range indices
        o_local = min(o_local, kBChunksPerRow - 1);
        const int o4 = min(block_n / 4 + o_local, last_out_tile);  // outputs past N_out are not stored
        b_c_of[v] = c_local;
        b_valid_of[v] = chunk_idx < kBCount ? 1 : 0;
        b_off[v] = (static_cast<int64_t>(c_local) * b_stride_kc + static_cast<int64_t>(o4) * b_stride_out) * 16;
        int8_t* const dst = &sh.ab.b[0][(c_local / 4) * kBStageStride + o_local * 4 * kWmmaK + (c_local % 4) * 4];
#pragma unroll
        for (int buf = 0; buf < 2; ++buf) {
            b_lds[buf][v] = dst + (buf ? kUnrollK * kBStageStride : 0);
        }
    }

    const int64_t a_chunk_step = kChunkK;
    const int64_t b_chunk_step = static_cast<int64_t>(kChunkK / 4) * b_stride_kc * 16;

    const auto load_stage = [&](const int chunk_idx, const int n_steps) {  // chunk_idx counts chunks, not bytes
#pragma unroll
        for (int v = 0; v < kAVecsPerThread; ++v) {
            if (!a_valid[v] || a_stage_of[v] >= n_steps) continue;
            stage.a[v] = *reinterpret_cast<const int4*>(a + a_off[v] + static_cast<int64_t>(chunk_idx) * a_chunk_step);
        }
#pragma unroll
        for (int v = 0; v < kBChunksPerThread; ++v) {
            if (!b_valid_of[v] || b_c_of[v] >= n_steps * 4) continue;
            stage.b[v] = *reinterpret_cast<const uint4*>(b + b_off[v] + static_cast<int64_t>(chunk_idx) * b_chunk_step);
        }
    };

    const auto store_stage = [&](const int buf, const int n_steps) {
#pragma unroll
        for (int v = 0; v < kAVecsPerThread; ++v) {
            if (!a_valid[v] || a_stage_of[v] >= n_steps) continue;
            *reinterpret_cast<int4*>(a_lds[buf][v]) = stage.a[v];
        }
#pragma unroll
        for (int v = 0; v < kBChunksPerThread; ++v) {
            if (!b_valid_of[v] || b_c_of[v] >= n_steps * 4) continue;
            int8_t* const dst = b_lds[buf][v];
            const uint4 p = stage.b[v];
            if constexpr (kTransposedB) {
                // The contraction is the micro-tile's row axis: gather the four n of one k.
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *reinterpret_cast<uint32_t*>(dst + i * kWmmaK) = gather_bytes(p.x, p.y, p.z, p.w, i);
                }
            } else {
                // The output is the micro-tile's row axis: every dword is a 4-byte contraction step.
                *reinterpret_cast<uint32_t*>(dst) = p.x;
                *reinterpret_cast<uint32_t*>(dst + kWmmaK) = p.y;
                *reinterpret_cast<uint32_t*>(dst + 2 * kWmmaK) = p.z;
                *reinterpret_cast<uint32_t*>(dst + 3 * kWmmaK) = p.w;
            }
        }
    };

    // One LDS stage (16 contraction bytes): all B fragments once, then one A fragment per M repeat.
    const auto compute_stage = [&](const int buf, const int stage) {
        v4i reg_b[kRepeatN];
#pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kWarpsN;
            reg_b[rn] = *reinterpret_cast<const v4i*>(&sh.ab.b[buf][stage * kBStageStride + (tile_n * kWmmaN + lane_in_subgroup) * kWmmaK]);
        }
#pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            const int tile_m = warp_m + rm * kWarpsM;
            const v4i a_frag = *reinterpret_cast<const v4i*>(&sh.ab.a[buf][stage * kAStageStride + (tile_m * kWmmaM + lane_in_subgroup) * kWmmaK]);
#pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                v8i& tile = acc[rm * kRepeatN + rn];
                tile = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, reg_b[rn], tile, false);
            }
        }
    };

    const int n_full = K_contract / kChunkK;
    const int tail_steps = (K_contract - n_full * kChunkK) / kWmmaK;
    const int n_chunks = n_full + (tail_steps > 0 ? 1 : 0);
    const auto chunk_steps = [&](const int chunk) -> int { return chunk < n_full ? kUnrollK : tail_steps; };

    load_stage(0, chunk_steps(0));
    store_stage(0, chunk_steps(0));
    __syncthreads();

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int buf = chunk & 1;
        if (chunk + 1 < n_chunks) {
            // Global reads of the next chunk issue first. Their latency is covered by this chunk's WMMAs.
            asm volatile("s_setprio 1" ::: "memory");
            load_stage(chunk + 1, chunk_steps(chunk + 1));
            asm volatile("s_setprio 0" ::: "memory");
        }
#pragma unroll
        for (int stage_idx = 0; stage_idx < kUnrollK; ++stage_idx) {
            if (stage_idx < chunk_steps(chunk)) compute_stage(buf, stage_idx);
        }
        if (chunk + 1 < n_chunks) {
            // The other buffer was last read in the previous chunk, behind this loop's barrier.
            store_stage(buf ^ 1, chunk_steps(chunk + 1));
        }
        __syncthreads();
    }

    // Epilogue: dequantize, C-shuffle through LDS, coalesced 16-byte output stores.
    uint16_t* __restrict__ const sh_c = &sh.c[wave_id][0][0];
    constexpr int kColsPerLane = 8;

#pragma unroll
    for (int rm = 0; rm < kRepeatM; ++rm) {
        const int tile_m = warp_m + rm * kWarpsM;
        float row_scale[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            row_scale[e] = a_scale[min(block_m + tile_m * kWmmaM + acc_row(e), M - 1)];
        }
#pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kWarpsN;
            const int acc_col = tile_n * kWmmaN + lane_in_subgroup;
            float col_scale = 1.0f;
            float col_bias = 0.0f;
            if constexpr (!kTransposedB) {
                col_scale = b_scale[min(block_n + acc_col, N_out - 1)];
                if (bias != nullptr) col_bias = output_bits_to_float<kOutputFp16>(bias[min(block_n + acc_col, N_out - 1)]);
            }
            const int repeat_idx = rm * kRepeatN + rn;
            // Accumulator -> LDS in the gfx11 interleaved row order (column-major, 16-bit elements).
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float value = static_cast<float>(acc[repeat_idx][e]) * row_scale[e] * col_scale + col_bias;
                sh_c[acc_row(e) * kWmmaN + lane_in_subgroup] = float_to_output_bits<kOutputFp16>(value);
            }
            // 32 threads cover 16 rows x 16 columns: two threads per row, 8 columns each.
            const int read_row = lane / 2;
            const int read_col = (lane % 2) * kColsPerLane;
            const int64_t out_row = block_m + tile_m * kWmmaM + read_row;
            const int64_t out_col_base = block_n + tile_n * kWmmaN + read_col;
            const uint16_t* const src = sh_c + read_row * kWmmaN + read_col;
            if (out_row < M && out_col_base + kColsPerLane <= N_out) {
                uint16_t* const dst = c + out_row * N_out + out_col_base;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else if (out_row < M) {
#pragma unroll
                for (int i = 0; i < kColsPerLane; ++i) {
                    const int64_t col = out_col_base + i;
                    if (col < N_out) c[out_row * N_out + col] = src[i];
                }
            }
        }
    }
}

}  // namespace

template <int M, int N, int U, int RM, int RN, bool TB>
struct Int8ConfigTag {
    static constexpr int kBlockWarpsM = M;
    static constexpr int kBlockWarpsN = N;
    static constexpr int kUnrollK = U;
    static constexpr int kRepeatM = RM;
    static constexpr int kRepeatN = RN;
    static constexpr bool kTransposedB = TB;
};

template <typename Config>
void launch_int8_scaled_mm_configured(
    const int8_t* const a,
    const uint8_t* const b,
    const float* const a_scale,
    const float* const b_scale,
    const uint16_t* const bias,
    uint16_t* const c,
    const int64_t M,
    const int64_t m_base,
    const int64_t N_out,
    const int64_t K_contract,
    const int64_t b_stride_kc,
    const int64_t b_stride_out,
    const int output_dtype,
    hipStream_t stream)
{
    constexpr int kBlockM = kWmmaM * Config::kBlockWarpsM * Config::kRepeatM;
    constexpr int kBlockN = kWmmaN * Config::kBlockWarpsN * Config::kRepeatN;
    constexpr int kThreadsPerBlock = kWaveSize * Config::kBlockWarpsM * Config::kBlockWarpsN;
    const dim3 block(kThreadsPerBlock, 1, 1);
    const dim3 grid(static_cast<uint32_t>((N_out + kBlockN - 1) / kBlockN), static_cast<uint32_t>((M - m_base + kBlockM - 1) / kBlockM), 1);

    if (output_dtype == 0) {
        hipLaunchKernelGGL(
            (int8_scaled_mm_kernel<Config::kBlockWarpsM, Config::kBlockWarpsN, Config::kUnrollK, Config::kRepeatM, Config::kRepeatN, Config::kTransposedB, true>),
            grid, block, 0, stream,
            a, b, a_scale, b_scale, bias, c,
            static_cast<int>(M), static_cast<int>(N_out), static_cast<int>(K_contract), static_cast<int>(b_stride_kc), static_cast<int>(b_stride_out), static_cast<int>(m_base));
    } else {
        hipLaunchKernelGGL(
            (int8_scaled_mm_kernel<Config::kBlockWarpsM, Config::kBlockWarpsN, Config::kUnrollK, Config::kRepeatM, Config::kRepeatN, Config::kTransposedB, false>),
            grid, block, 0, stream,
            a, b, a_scale, b_scale, bias, c,
            static_cast<int>(M), static_cast<int>(N_out), static_cast<int>(K_contract), static_cast<int>(b_stride_kc), static_cast<int>(b_stride_out), static_cast<int>(m_base));
    }
}

extern "C" bool launch_int8_scaled_mm(
    const int8_t* const a,
    const uint8_t* const b,
    const float* const a_scale,
    const float* const b_scale,
    const uint16_t* const bias,
    uint16_t* const c,
    const int64_t M,
    const int64_t N_out,
    const int64_t K_contract,
    const int64_t b_stride_kc,
    const int64_t b_stride_out,
    const int transposed_b,
    const int block_warps_m,
    const int block_warps_n,
    const int unroll_k,
    const int repeat_m,
    const int repeat_n,
    const int output_dtype,
    hipStream_t stream)
{
    // A tile that does not divide M does ceil() worth of MMA work, so the rows that do not fill one get their
    // own launch with a short tile: 5.8% of a 4112-row cell is otherwise dead MMA work (4112 = 16x256 + 16).
    constexpr int kTailWarpsM = 1;
    constexpr int kTailRepeatM = 4;
    constexpr int kTailRepeatN = 2;  // 8 register tiles: the remainder is bandwidth bound, not MMA bound
    constexpr int kTailBlockM = kWmmaM * kTailWarpsM * kTailRepeatM;

    const auto launch_tag = [&](auto tag, const int64_t m_base, const int64_t rows) {
        launch_int8_scaled_mm_configured<decltype(tag)>(
            a, b, a_scale, b_scale, bias, c, m_base + rows, m_base, N_out, K_contract, b_stride_kc, b_stride_out, output_dtype, stream);
    };
    const auto try_launch_one = [&](auto tag, const int64_t m_base, const int64_t rows) -> bool {
        if (block_warps_m == decltype(tag)::kBlockWarpsM && block_warps_n == decltype(tag)::kBlockWarpsN && unroll_k == decltype(tag)::kUnrollK &&
            repeat_m == decltype(tag)::kRepeatM && repeat_n == decltype(tag)::kRepeatN && (transposed_b != 0) == decltype(tag)::kTransposedB) {
            launch_tag(tag, m_base, rows);
            return true;
        }
        return false;
    };
    const auto try_launch = [&](auto tag) -> bool {
        constexpr int kBlockM = kWmmaM * decltype(tag)::kBlockWarpsM * decltype(tag)::kRepeatM;
        const int64_t m_full = (M / kBlockM) * kBlockM;
        const bool split = kBlockM > kTailBlockM && m_full > 0 && m_full < M;
        if (!try_launch_one(tag, 0, split ? m_full : M)) return false;
        if (split) {
            // the remainder with a tile that leaves at most 3/4 of one tile row unused
            launch_tag(Int8ConfigTag<kTailWarpsM, decltype(tag)::kBlockWarpsN, decltype(tag)::kUnrollK, kTailRepeatM, kTailRepeatN, decltype(tag)::kTransposedB>{}, m_full, M - m_full);
        }
        return true;
    };

    // (warps_m, warps_n, unroll_k, repeat_m, repeat_n, transposed_b)
    return
        // forward: C = A @ W^T
        try_launch(Int8ConfigTag<1, 1, 2, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 2, 2, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 8, 2, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 1, 4, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 2, 4, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 4, 4, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 8, 4, 1, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 2, 2, 2, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 2, 2, false>{}) ||
        try_launch(Int8ConfigTag<2, 2, 2, 2, 2, false>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 2, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 4, 2, false>{}) ||
        try_launch(Int8ConfigTag<1, 8, 2, 4, 2, false>{}) ||
        try_launch(Int8ConfigTag<2, 2, 2, 4, 4, false>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 4, 4, false>{}) ||
        try_launch(Int8ConfigTag<4, 2, 2, 4, 4, false>{}) ||
        try_launch(Int8ConfigTag<2, 4, 4, 4, 4, false>{}) ||
        try_launch(Int8ConfigTag<2, 2, 4, 4, 2, false>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 4, 2, false>{}) ||
        try_launch(Int8ConfigTag<2, 8, 2, 4, 2, false>{}) ||
        try_launch(Int8ConfigTag<4, 4, 2, 4, 4, false>{}) ||
        // backward: C = A @ W (contraction along the weight's output axis)
        try_launch(Int8ConfigTag<1, 1, 2, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 2, 2, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 8, 2, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 1, 4, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 2, 4, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 4, 4, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 8, 4, 1, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 2, 2, 2, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 2, 2, true>{}) ||
        try_launch(Int8ConfigTag<2, 2, 2, 2, 2, true>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 2, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 4, 2, 4, 2, true>{}) ||
        try_launch(Int8ConfigTag<1, 8, 2, 4, 2, true>{}) ||
        try_launch(Int8ConfigTag<2, 2, 2, 4, 4, true>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 4, 4, true>{}) ||
        try_launch(Int8ConfigTag<4, 2, 2, 4, 4, true>{}) ||
        try_launch(Int8ConfigTag<2, 4, 4, 4, 4, true>{}) ||
        try_launch(Int8ConfigTag<2, 2, 4, 4, 2, true>{}) ||
        try_launch(Int8ConfigTag<2, 4, 2, 4, 2, true>{}) ||
        try_launch(Int8ConfigTag<2, 8, 2, 4, 2, true>{}) ||
        try_launch(Int8ConfigTag<4, 4, 2, 4, 4, true>{}) ||
        false;
}

#ifndef NO_PYTORCH
void int8_scaled_mm(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b_prepacked,
    const torch::stable::Tensor& a_scale,
    const std::optional<torch::stable::Tensor>& b_scale,
    const std::optional<torch::stable::Tensor>& bias,
    torch::stable::Tensor& c,
    const int64_t transposed_b,
    const int64_t block_warps_m,
    const int64_t block_warps_n,
    const int64_t unroll_k,
    const int64_t repeat_m,
    const int64_t repeat_n)
{
    STD_TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    STD_TORCH_CHECK(b_prepacked.is_cuda(), "b_prepacked must be a CUDA tensor");
    STD_TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    const auto device_index = a.get_device_index();
    STD_TORCH_CHECK(b_prepacked.get_device_index() == device_index, "b_prepacked must be on the same device as a");
    STD_TORCH_CHECK(c.get_device_index() == device_index, "c must be on the same device as a");

    STD_TORCH_CHECK(a.scalar_type() == torch::stable::ScalarType::Char, "a must be int8");
    STD_TORCH_CHECK(b_prepacked.scalar_type() == torch::stable::ScalarType::Char, "b_prepacked must be int8");
    STD_TORCH_CHECK(c.scalar_type() == torch::stable::ScalarType::Half || c.scalar_type() == torch::stable::ScalarType::BFloat16, "c must be float16 or bfloat16");
    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(c.dim() == 2, "c must be 2D");

    const int64_t M = a.size(0);
    const int64_t K_contract = a.size(1);
    // The prepacked weight holds the weight's (input, output) axes. Which of them is the contraction
    // depends on the direction.
    const int64_t weight_in = transposed_b ? c.size(1) : K_contract;
    const int64_t weight_out = transposed_b ? K_contract : c.size(1);
    STD_TORCH_CHECK(b_prepacked.numel() == weight_in * weight_out, "b_prepacked must hold weight_in * weight_out bytes");
    STD_TORCH_CHECK(weight_in % 4 == 0 && weight_out % 4 == 0, "the weight axes must be multiples of 4");
    STD_TORCH_CHECK(K_contract % kWmmaK == 0, "the contraction axis must be a multiple of 16");
    STD_TORCH_CHECK(c.size(0) == M, "c.shape[0] must equal M");

    STD_TORCH_CHECK(a.stride(0) == K_contract && a.stride(1) == 1, "a must be contiguous");
    STD_TORCH_CHECK(c.stride(0) == c.size(1) && c.stride(1) == 1, "c must be contiguous");
    STD_TORCH_CHECK(b_prepacked.is_contiguous(), "b_prepacked must be contiguous");
    STD_TORCH_CHECK(a_scale.is_cuda() && a_scale.scalar_type() == torch::stable::ScalarType::Float, "a_scale must be float32");
    STD_TORCH_CHECK(a_scale.numel() == M, "a_scale must have M elements");
    if (b_scale.has_value()) {
        STD_TORCH_CHECK(b_scale->scalar_type() == torch::stable::ScalarType::Float, "b_scale must be float32");
        STD_TORCH_CHECK(b_scale->numel() == c.size(1), "b_scale must have N elements");
    }
    if (bias.has_value()) {
        STD_TORCH_CHECK(bias->scalar_type() == c.scalar_type(), "bias must have the same dtype as c");
        STD_TORCH_CHECK(bias->numel() == c.size(1), "bias must have N elements");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    // Chunk index = (weight_in / 4) * (weight_out / 4) + (weight_out / 4), so the stride of whichever
    // weight axis is the contraction is the outer one for the forward and 1 for the backward.
    const int64_t inner = (transposed_b ? K_contract : c.size(1)) / 4;
    const int64_t b_stride_kc = transposed_b ? 1 : inner;
    const int64_t b_stride_out = transposed_b ? inner : 1;

    const bool launched = launch_int8_scaled_mm(
        reinterpret_cast<const int8_t*>(a.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(b_prepacked.const_data_ptr()),
        reinterpret_cast<const float*>(a_scale.const_data_ptr()),
        b_scale.has_value() ? reinterpret_cast<const float*>(b_scale->const_data_ptr()) : nullptr,
        bias.has_value() ? reinterpret_cast<const uint16_t*>(bias->const_data_ptr()) : nullptr,
        reinterpret_cast<uint16_t*>(c.mutable_data_ptr()),
        M,
        c.size(1),
        K_contract,
        b_stride_kc,
        b_stride_out,
        static_cast<int>(transposed_b),
        static_cast<int>(block_warps_m),
        static_cast<int>(block_warps_n),
        static_cast<int>(unroll_k),
        static_cast<int>(repeat_m),
        static_cast<int>(repeat_n),
        c.scalar_type() == torch::stable::ScalarType::Half ? 0 : 1,
        stream);
    STD_TORCH_CHECK(launched, "Unsupported config");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "int8_scaled_mm kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(feather_ops, m)
{
    m.def(
        "int8_scaled_mm("
        "Tensor a, "
        "Tensor b_prepacked, "
        "Tensor a_scale, "
        "Tensor? b_scale, "
        "Tensor? bias, "
        "Tensor(a!) c, "
        "int transposed_b, "
        "int block_warps_m, "
        "int block_warps_n, "
        "int unroll_k, "
        "int repeat_m, "
        "int repeat_n"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("int8_scaled_mm", TORCH_BOX(&int8_scaled_mm));
}
#endif

// Activation quantizer for the ConvRot format.
//
// The int8 GEMM above consumes per-row int8 activations in the rotated basis, so a training step needs
// `q = round((x . H) / scale)` with a per-row scale over the whole row. Done as two torch/Triton passes
// (a block-diagonal bf16 matmul, then a row-wise quantize) that is 7 bytes of traffic per element and
// 2*M*K*group_size flops. Done here it is one pass with the row resident in LDS, and the rotation is the
// radix-4 decomposition of H = h4^(x)k rather than a matmul:
//
//   y[d] = sum(a) - 2 * a[3 - d]        per 4-element block, one stage per index digit
//
// which is exactly `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` (each row has one -1) and needs no
// multiplies. Four stages of that equal the group-256 Hadamard that `_build_hadamard` builds.
//
// One wave owns one row and its own LDS slot, so there is no barrier anywhere in the kernel.

namespace {

constexpr int kQuantGroup = 256;      // ConvRot group size this kernel implements
constexpr int kQuantPerLane = kQuantGroup / kWaveSize;  // 8 values per lane per group

__device__ __forceinline__ float bits_to_float(const uint16_t bits) {
    const uint32_t expanded = static_cast<uint32_t>(bits) << 16;
    return __builtin_bit_cast(float, expanded);
}

__device__ __forceinline__ uint16_t float_to_bits_bf16(const float value) { return float_to_bf16_bits(value); }

template <bool kInputFp16>
__device__ __forceinline__ float input_bits_to_float(const uint16_t bits) {
    if constexpr (kInputFp16) {
        return __half2float(__builtin_bit_cast(_Float16, bits));
    } else {
        return bits_to_float(bits);
    }
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor(v, off));
    }
    return v;
}

// One radix-4 Walsh stage over index digit `shift` of the group's 256 values, in place in LDS.
// The staged values stay in fp32: rounding them to bf16 per stage would cost four mantissa roundings
// instead of the single one the reference recipe applies to the finished rotation.
__device__ __forceinline__ void radix4_stage(float* const row, const int g, const int shift, const int lane) {
    const int stride = 1 << shift;
    float out[kQuantPerLane];
#pragma unroll
    for (int i = 0; i < kQuantPerLane; ++i) {
        const int e = lane * kQuantPerLane + i;
        const int base = e & ~(3 << shift);
        const float a0 = row[g + base];
        const float a1 = row[g + base + stride];
        const float a2 = row[g + base + 2 * stride];
        const float a3 = row[g + base + 3 * stride];
        const int d = (e >> shift) & 3;
        const float complement = (d == 0) ? a3 : (d == 1) ? a2 : (d == 2) ? a1 : a0;
        out[i] = (a0 + a1 + a2 + a3) - 2.0f * complement;
    }
    // Read all partners before writing any of them back.
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#pragma unroll
    for (int i = 0; i < kQuantPerLane; ++i) {
        row[g + lane * kQuantPerLane + i] = out[i];
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// Four rounded int8 codes packed into one dword: two saturating i32->i16 packs and one byte permute, instead
// of four shifts, four masks and three ors. The scale keeps the reference recipe (`amax / 127`), so the codes
// are clamped - dropping the clamp by scaling to 1/127.5 like ~/sageattention-autotune was measured to be
// free either way (the pass is latency bound) and it perturbs the codes relative to that recipe.
constexpr float kInt8MaxCode = 127.0f;

__device__ __forceinline__ int round_clamped_int8(const float v, const float inv_divisor) {
    return static_cast<int>(fminf(fmaxf(rintf(v * inv_divisor), -kInt8MaxCode), kInt8MaxCode));
}

__device__ __forceinline__ uint32_t pack4_int8(const float v0, const float v1, const float v2, const float v3, const float inv_divisor) {
    const int32_t q0 = round_clamped_int8(v0, inv_divisor);
    const int32_t q1 = round_clamped_int8(v1, inv_divisor);
    const int32_t q2 = round_clamped_int8(v2, inv_divisor);
    const int32_t q3 = round_clamped_int8(v3, inv_divisor);
    const uint32_t p01 = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pk_i16(q0, q1));
    const uint32_t p23 = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pk_i16(q2, q3));
    return __byte_perm(p01, p23, 0x6420u);  // low bytes of the four i16 halves
}

template <bool kInputFp16>
__device__ __forceinline__ uint16_t float_to_bits(const float v) {
    if constexpr (kInputFp16) {
        return __builtin_bit_cast(uint16_t, __float2half_rn(v));
    } else {
        return float_to_bf16_bits(v);
    }
}

// Round trip through the input dtype: the reference does the rotation in that dtype, so this makes the kernel
// see the same values it would (and keeps the quantizer's own rounding out of the result).
template <bool kInputFp16>
__device__ __forceinline__ float as_input_rounded(const float v) {
    if constexpr (kInputFp16) {
        return __half2float(__float2half_rn(v));
    } else {
        return bits_to_float(float_to_bf16_bits(v));
    }
}

// Produce one 256-value group into a wave's fp32 LDS slot: load it (folding the per-column scale when the
// rotation is off), then, for the forward, run the four radix-4 Hadamard stages. The staged row and the
// wide-row streaming path both call this, so both apply exactly the same arithmetic to the same inputs.
template <bool kInputFp16>
__device__ __forceinline__ void produce_group(
    const uint16_t* __restrict__ x_row,
    const float* __restrict__ col_scale,
    float* const slot,
    const int g,
    const int lane,
    const int rotate)
{
    const int base = lane * kQuantPerLane;
#pragma unroll
    for (int i = 0; i < kQuantPerLane; ++i) {
        float value = input_bits_to_float<kInputFp16>(x_row[g + base + i]);
        if (!rotate && col_scale != nullptr) value = as_input_rounded<kInputFp16>(value * col_scale[g + base + i]);
        slot[base + i] = value;
    }
    if (rotate) {
        radix4_stage(slot, 0, 0, lane);
        radix4_stage(slot, 0, 2, lane);
        radix4_stage(slot, 0, 4, lane);
        radix4_stage(slot, 0, 6, lane);
    }
}

// One row per block. The row's 256-value groups are spread over the block's waves (one group per wave per
// round), so the block is as wide as the row and its LDS footprint is (K bf16 of finished values) plus one
// fp32 group slot per wave. The fp32 slot is what the radix-4 stages work in - rounding after each stage
// would cost four mantissa roundings instead of the single one the reference recipe applies - and the
// finished group is rounded into the bf16 row, which is where the max and the quantize pass read it from.
// A group's rotation stays inside its own wave (no barrier). The row absmax is the block's only reduction.
//
// kStageRow=false is the wide-row path (K > 16384, up to the 248320-wide Vocab projection of a 32B text
// encoder): the finished row would no longer leave room for a full block of waves in LDS, so the quantize
// pass recomputes each group from global memory (L2-hot) instead, which keeps 32 waves and 4 workgroups per
// WGP. The scale and the codes are identical to the staged path because both use the same producer.
template <bool kInputFp16, bool kStageRow>
__global__ void convrot_quantize_kernel(
    const uint16_t* __restrict__ x,       // [M, K] bf16 or fp16
    const float* __restrict__ col_scale,  // [K] or null: per-contraction-column multiplier (the weight scale fold)
    int8_t* __restrict__ q,               // [M, K] int8 output
    float* __restrict__ row_scale,        // [M] output scale
    const int K,
    const int groups_per_row,             // K / 256
    const int rotate)
{
    extern __shared__ __align__(16) float smem[];
    const int lane = static_cast<int>(threadIdx.x) % kWaveSize;
    const int wave = static_cast<int>(threadIdx.x) / kWaveSize;
    const int waves = static_cast<int>(blockDim.x) / kWaveSize;
    const int64_t row = blockIdx.x;
    float* const group_slot = smem;  // waves * 256 floats
    // The staged path keeps the finished (rotated, rounded) row for the quantize pass. The wide-row path
    // has no row buffer and recomputes each group from x instead.
    uint16_t* const row_vals = kStageRow ? reinterpret_cast<uint16_t*>(smem + waves * kQuantGroup) : nullptr;

    const uint16_t* const x_row = x + row * K;
    const int lane_base = lane * kQuantPerLane;
    float vmax = 0.0f;
    for (int gidx = wave; gidx < groups_per_row; gidx += waves) {
        const int g = gidx * kQuantGroup;
        float* const slot = group_slot + wave * kQuantGroup;
        produce_group<kInputFp16>(x_row, col_scale, slot, g, lane, rotate);
#pragma unroll
        for (int i = 0; i < kQuantPerLane; ++i) {
            const uint16_t bits = float_to_bits<kInputFp16>(slot[lane_base + i]);
            if constexpr (kStageRow) row_vals[g + lane_base + i] = bits;
            vmax = fmaxf(vmax, fabsf(input_bits_to_float<kInputFp16>(bits)));
        }
    }

    // Row absmax. The rotation is normalized by 1/sqrt(group). Folding that into the stored scale keeps the
    // quantized codes identical with and without the rotation.
    constexpr float kNormInv = 1.0f / 16.0f;  // 1 / sqrt(256)
    vmax = wave_max(vmax);  // across the wave's threads first, then across the block's waves
    __shared__ float wave_maxima[32];
    if (lane == 0) wave_maxima[wave] = vmax;
    __syncthreads();
    float row_max = 0.0f;
    for (int w = 0; w < waves; ++w) {
        row_max = fmaxf(row_max, wave_maxima[w]);
    }
    const float divisor = fmaxf(row_max / 127.0f, 1e-30f);  // scale of the values as stored
    if (threadIdx.x == 0) {
        row_scale[row] = rotate ? divisor * kNormInv : divisor;
    }
    const float inv_divisor = 1.0f / divisor;

    int8_t* const q_row = q + row * K;
    for (int gidx = wave; gidx < groups_per_row; gidx += waves) {
        const int g = gidx * kQuantGroup;
        const int base = g + lane_base;
        float* const slot = group_slot + wave * kQuantGroup;
        float v[kQuantPerLane];
        if constexpr (kStageRow) {
#pragma unroll
            for (int i = 0; i < kQuantPerLane; ++i) {
                v[i] = input_bits_to_float<kInputFp16>(row_vals[base + i]);
            }
        } else {
            // no row buffer to read back: recompute the group (x is L2-hot here)
            produce_group<kInputFp16>(x_row, col_scale, slot, g, lane, rotate);
#pragma unroll
            for (int i = 0; i < kQuantPerLane; ++i) {
                v[i] = input_bits_to_float<kInputFp16>(float_to_bits<kInputFp16>(slot[lane_base + i]));
            }
        }
        uint32_t packed[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            packed[i] = pack4_int8(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], inv_divisor);
        }
        *reinterpret_cast<uint2*>(q_row + base) = make_uint2(packed[0], packed[1]);
    }
}

}  // namespace

extern "C" bool launch_convrot_quantize(
    const void* const x,
    const float* const col_scale,
    int8_t* const q,
    float* const row_scale,
    const int64_t M,
    const int64_t K,
    const int input_fp16,
    const int rotate,
    hipStream_t stream)
{
    if (K % kQuantGroup != 0) return false;
    const int64_t groups_per_row = K / kQuantGroup;
    // LDS budget: gfx11 allows 64 KB per workgroup (a WGP holds two such workgroups), minus the kernel's
    // static shared memory (the per-wave maxima) and alignment slack. Staging the whole finished row costs
    // K * 2 bytes, which only leaves a full block of waves while K <= 16384. Above that the quantize pass
    // recomputes each group from global memory instead, so any K works without a per-workgroup limit.
    constexpr int64_t kWorkgroupLdsBytes = 65536;
    constexpr int64_t kStaticLdsBytes = 256;
    constexpr int64_t kMaxStagedRowBytes = 32768;
    const bool stage_row = K * static_cast<int64_t>(sizeof(uint16_t)) <= kMaxStagedRowBytes;
    int64_t waves = groups_per_row < 32 ? groups_per_row : 32;
    if (stage_row) {
        // the previous budget (32 KB of total footprint) collapsed K >= 16384 to a single wave, i.e. 32 threads
        // and 33 KB of LDS per block. The row now uses half the workgroup LDS, so a block always has >= 31 waves
        const int64_t budget_waves = (kWorkgroupLdsBytes - kStaticLdsBytes - K * static_cast<int64_t>(sizeof(uint16_t))) /
                                     (kQuantGroup * static_cast<int64_t>(sizeof(float)));
        if (waves > budget_waves) waves = budget_waves;
    }
    if (waves < 1) waves = 1;
    const size_t smem = static_cast<size_t>(waves) * kQuantGroup * sizeof(float) +
                        (stage_row ? static_cast<size_t>(K) * sizeof(uint16_t) : 0);
    if (smem > static_cast<size_t>(kWorkgroupLdsBytes - kStaticLdsBytes)) return false;
    const dim3 block(static_cast<uint32_t>(kWaveSize * waves), 1, 1);
    const dim3 grid(static_cast<uint32_t>(M), 1, 1);
    const uint16_t* const xp = static_cast<const uint16_t*>(x);
    const auto launch = [&](auto fp16_tag, auto stage_tag) {
        constexpr bool kFp16 = decltype(fp16_tag)::value;
        constexpr bool kStage = decltype(stage_tag)::value;
        const void* const fn = reinterpret_cast<const void*>(&convrot_quantize_kernel<kFp16, kStage>);
        if (smem > 48 * 1024) {
            (void)hipFuncSetAttribute(fn, hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        }
        hipLaunchKernelGGL(
            (convrot_quantize_kernel<kFp16, kStage>), grid, block, smem, stream,
            xp, col_scale, q, row_scale, static_cast<int>(K), static_cast<int>(groups_per_row), rotate);
    };
    if (input_fp16) {
        if (stage_row) {
            launch(std::integral_constant<bool, true>{}, std::integral_constant<bool, true>{});
        } else {
            launch(std::integral_constant<bool, true>{}, std::integral_constant<bool, false>{});
        }
    } else {
        if (stage_row) {
            launch(std::integral_constant<bool, false>{}, std::integral_constant<bool, true>{});
        } else {
            launch(std::integral_constant<bool, false>{}, std::integral_constant<bool, false>{});
        }
    }
    return hipGetLastError() == hipSuccess;
}

#ifndef NO_PYTORCH
void convrot_quantize(
    const torch::stable::Tensor& x,
    const std::optional<torch::stable::Tensor>& col_scale,
    torch::stable::Tensor& q,
    torch::stable::Tensor& row_scale,
    const int64_t rotate)
{
    STD_TORCH_CHECK(x.is_cuda() && q.is_cuda() && row_scale.is_cuda(), "tensors must be CUDA tensors");
    STD_TORCH_CHECK(x.dim() == 2, "x must be 2D");
    STD_TORCH_CHECK(x.scalar_type() == torch::stable::ScalarType::Half || x.scalar_type() == torch::stable::ScalarType::BFloat16,
                    "x must be float16 or bfloat16");
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Char, "q must be int8");
    STD_TORCH_CHECK(row_scale.scalar_type() == torch::stable::ScalarType::Float, "row_scale must be float32");
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    STD_TORCH_CHECK(q.sizes() == x.sizes(), "q must have the same shape as x");
    STD_TORCH_CHECK(row_scale.numel() == M, "row_scale must have M elements");
    STD_TORCH_CHECK(x.stride(0) == K && x.stride(1) == 1, "x must be contiguous");
    STD_TORCH_CHECK(q.stride(0) == K && q.stride(1) == 1, "q must be contiguous");
    STD_TORCH_CHECK(K % 256 == 0, "the contraction axis must be a multiple of the ConvRot group size (256)");
    if (col_scale.has_value()) {
        STD_TORCH_CHECK(col_scale->scalar_type() == torch::stable::ScalarType::Float, "col_scale must be float32");
        STD_TORCH_CHECK(col_scale->numel() == K, "col_scale must have K elements");
        STD_TORCH_CHECK(col_scale->is_cuda() && col_scale->get_device_index() == x.get_device_index(), "col_scale must be on x's device");
    }

    const auto device_index = x.get_device_index();
    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const bool ok = launch_convrot_quantize(
        x.const_data_ptr(),
        col_scale.has_value() ? reinterpret_cast<const float*>(col_scale->const_data_ptr()) : nullptr,
        reinterpret_cast<int8_t*>(q.mutable_data_ptr()),
        reinterpret_cast<float*>(row_scale.mutable_data_ptr()),
        M,
        K,
        x.scalar_type() == torch::stable::ScalarType::Half ? 1 : 0,
        static_cast<int>(rotate),
        stream);
    STD_TORCH_CHECK(ok, "convrot_quantize launch failed: ", hipGetErrorString(hipGetLastError()));
}

STABLE_TORCH_LIBRARY_FRAGMENT(feather_ops, m)
{
    m.def("convrot_quantize(Tensor x, Tensor? col_scale, Tensor(a!) q, Tensor(b!) row_scale, int rotate) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("convrot_quantize", TORCH_BOX(&convrot_quantize));
}
#endif
