#include <__clang_cuda_math_forward_declares.h>
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

struct bfloat16_t {
    uint16_t data;

    __device__ __forceinline__ bfloat16_t() = default;

    __device__ __forceinline__ bfloat16_t(const float f) {
        const uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
        data = static_cast<uint16_t>(u >> 16);
    }

    __device__ __forceinline__ operator float() const {
        const uint32_t u = static_cast<uint32_t>(data) << 16;
        return *reinterpret_cast<const float*>(&u);
    }
};

__device__ __forceinline__ bfloat16_t operator+(const bfloat16_t a, const bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) + static_cast<float>(b));
}

__device__ __forceinline__ uint16_t half_to_bits(const half h) {
    return *reinterpret_cast<const uint16_t*>(&h);
}

__device__ __forceinline__ half bits_to_half(const uint16_t bits) {
    return *reinterpret_cast<const half*>(&bits);
}

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
// rocwmma::Constants::AMDGCN_WAVE_SIZE returns 64 during host compilation
constexpr int kWaveSize = 32;

template <bool kOutputBf16>
__device__ __forceinline__ uint16_t float_to_output_bits(const float value) {
    if constexpr (kOutputBf16) {
        return bfloat16_t(value).data;
    }
    return half_to_bits(__float2half_rn(value));
}

template <bool kOutputBf16>
__device__ __forceinline__ uint16_t add_output_bits(const uint16_t a, const uint16_t b) {
    if constexpr (kOutputBf16) {
        bfloat16_t lhs;
        bfloat16_t rhs;
        lhs.data = a;
        rhs.data = b;
        return (lhs + rhs).data;
    }
    return half_to_bits(__hadd(bits_to_half(a), bits_to_half(b)));
}

// Two bfloat16 in one dword -> two half in one dword, with the rounding a tensor cast performs.
__device__ __forceinline__ uint32_t bf16x2_to_half2(const uint32_t v) {
    const float lo = __uint_as_float(v << 16);
    const float hi = __uint_as_float(v & 0xFFFF0000u);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Packed fp8 -> 4 values of `klsFp16 ? fp16 : bf16` bits. `lo` receives the first two, `hi` the last two.
//
// fp8e5m2 -> fp16 is free: both formats bias the exponent by 15 and fp8e5m2's 2 mantissa bits are the
// top of fp16's 10, so the byte is the high half of the fp16 bit pattern and the perm alone produces it.
// fp8e4m3 -> fp16 needs the +8 exponent rebias and a shift. The sign is added twice so it carries out
// of bit 14 into bit 15.
// Both -> bf16 need a different rebias (to 127) and a 7-bit mantissa, so those forms re-assemble the
// fields instead of shifting. Denormals are dropped in every form, which the fp16 path already did.
template <bool kE5M2, bool kFp16>
__device__ __forceinline__ void fp8x4_to_half2x2(const uint32_t p, uint32_t& out_lo, uint32_t& out_hi) {
    if constexpr (kE5M2) {
        const uint32_t lo = __builtin_amdgcn_perm(0u, p, 0x010c000cu);
        const uint32_t hi = __builtin_amdgcn_perm(0u, p, 0x030c020cu);
        if constexpr (kFp16) {
            out_lo = lo;
            out_hi = hi;
        } else {
            // lane = b << 8: sign stays at 15, (e, m) moves from 14:8 to 10:4, +112 rebias is an add
            // because it lands inside the exponent field. The per-lane sum cannot reach bit 15.
            out_lo = (lo & 0x80008000u) | (((lo >> 3) & 0x0fe00fe0u) + 0x38003800u);
            out_hi = (hi & 0x80008000u) | (((hi >> 3) & 0x0fe00fe0u) + 0x38003800u);
        }
    } else {
        const uint32_t lo_pair = __builtin_amdgcn_perm(0u, p, 0x0c010c00u);
        const uint32_t hi_pair = __builtin_amdgcn_perm(0u, p, 0x0c030c02u);
        if constexpr (kFp16) {
            out_lo = (lo_pair << 7) + ((lo_pair & 0x00800080u) << 7) + 0x20002000u;
            out_hi = (hi_pair << 7) + ((hi_pair & 0x00800080u) << 7) + 0x20002000u;
        } else {
            // lane = b: sign to 15, 4-bit exponent rebased by +120 into 14:7, mantissa into 6:4.
            out_lo = ((lo_pair & 0x00800080u) << 8) | (((lo_pair & 0x007f007fu) << 4) + 0x3c003c00u);
            out_hi = ((hi_pair & 0x00800080u) << 8) | (((hi_pair & 0x007f007fu) << 4) + 0x3c003c00u);
        }
    }
}

// 16-row swizzle used by LDS physical mapping.
__device__ __forceinline__ constexpr int c_row_logi_to_phys_16(const int x) {
    return ((x & 7) << 1) | ((x >> 3) & 1);
}

__device__ __forceinline__ constexpr int a_row_phys_to_logi_16(const int x) {
    return ((x & 1) << 3) | ((x >> 1) & 7);
}

// Shape contract, all of it enforced here rather than in the caller:
// - M may be any value >= 1. The grid rounds up, the A staging clamps out-of-range rows to M - 1
//   (in-bounds memory, garbage products), and the epilogue drops the rows it is not asked for.
// - N must be a multiple of 16, which is the store and tile granularity, so the epilogue only tests
//   tile_n_base < N. B staging clamps the column the same way A clamps the row.
// - K must be a multiple of 16, which is the contraction granularity, and the prepacked weight is
//   zero-padded to a multiple of 128 so the last chunk always reads zeros instead of running off the
//   end. A is clamped into range for those steps, and a clamped A times a zero B row is still zero,
//   so no staging path needs a branch.
template <int kBlockWarpsM,
          int kBlockWarpsN,
          int kUnrollK,
          int kRepeatM,
          int kRepeatN,
          bool kEnableAWsgr,
          bool kUseFp8E5M2,
          bool kABf16In,
          bool kOutputBf16,
          bool kTailFree>
__global__ void scaled_mm_kernel(
    const uint16_t* __restrict__ const a,
    const uint8_t* __restrict__ const b_prepacked,
    const float* __restrict__ const scale,
    const uint16_t* __restrict__ const bias,
    uint16_t* __restrict__ const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int has_scale,
    const int has_bias)
{

    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

    // K0xMxK1 layout for A matrix (no extra LDS padding).
    // Apply row permutation on A store to improve LDS local-read banking while
    // keeping compact LDS footprint and 128-bit accesses.
    // K1 = 8 for fp16: enables vec8 LDS reads (like CK)
    constexpr int kK1 = 8;
    // K0 = 16 / 8 = 2
    constexpr int kK0 = kWmmaK / kK1;
    constexpr int kAStrideK1 = kK1;
    constexpr int kShASize = kUnrollK * kK0 * kBlockM * kAStrideK1;

    // B uses NxK prepacked layout for efficient vec16 stores during loading
    // C-shuffle epilogue reuses sh_a and sh_b memory. Each warp needs 16*16 output elements.
    constexpr int kCStride = kWmmaN;  // 16 elements per row

    union SharedStorage {
        struct {
            uint16_t a[kShASize];
            uint8_t b[kUnrollK][kBlockN][kWmmaK];
        } ab;
        uint16_t c[kBlockWarpsM * kBlockWarpsN][kWmmaM][kCStride];
    };
    __shared__ __align__(16) SharedStorage sh;

    const int block_m = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_n = static_cast<int>(blockIdx.x) * kBlockN;
    // Rows of this block that exist. The rest still run through the tile and are dropped by the epilogue.
    // A tail-free launch is the shipped tile exactly: every tile is full, so none of the clamps or masks
    // below exist in its code. Everything else takes the masked path.
    const int m_rows = kTailFree ? kBlockM : static_cast<int>(min(static_cast<int64_t>(kBlockM), M - block_m));
    const int64_t m_clamp_row = M - 1;

    const int tid = static_cast<int>(threadIdx.x);
    constexpr int kThreads = kWaveSize * kBlockWarpsM * kBlockWarpsN;

    // Flattened wave mapping with 1D thread blocks.
    const int wave_id = tid / kWaveSize;
    const int warp_m = wave_id % kBlockWarpsM;
    const int warp_n = wave_id / kBlockWarpsM;
    const int lane = tid % kWaveSize;

    // Accumulator registers: 8 floats per WMMA tile in wave32 mode
    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    float acc[kRepeatTiles][8];
    #pragma unroll
    for (int r = 0; r < kRepeatTiles; ++r) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    // Loading A: K0xMxK1 layout with physical/inverse row mapping and
    // wave-separated global-read ownership.
    constexpr int kAVecs = kK0 * kBlockM;
    constexpr int kAOwnerWaves = kEnableAWsgr ? kBlockWarpsM : (kBlockWarpsM * kBlockWarpsN);
    constexpr int kAOwnerThreads = kAOwnerWaves * kWaveSize;
    constexpr int kAVecsPerOwnerThread = (kAVecs + kAOwnerThreads - 1) / kAOwnerThreads;
    const auto a_row_phys_to_logi = [&](const int row_phys) -> int {
        const int tile_base = row_phys & ~15;
        const int local = row_phys & 15;
        return tile_base + a_row_phys_to_logi_16(local);
    };
    const auto sh_a_row_ptr = [&](const int stage, const int k0, const int m) -> uint16_t* {
        const int idx = ((stage * kK0 + k0) * kBlockM + m) * kAStrideK1;
        return &sh.ab.a[idx];
    };

    // Staging is branch-free: an out-of-range k is clamped to the last valid 8-element slot instead of
    // skipping the load, and the tail stages are zeroed through B afterwards (a zero B row makes the
    // product zero regardless of the A that was clamped into place).
    const auto load_a_lds_k0mk1 = [&](const int stage, const int64_t kk) -> void {
        // WSGR ownership: only A-owner waves issue A global->LDS stores.
        if constexpr (kEnableAWsgr) {
            if (wave_id >= kAOwnerWaves) return;
        }

        const int a_owner_tid = kEnableAWsgr ? (wave_id * kWaveSize + lane) : tid;

        // Physical LDS space is traversed directly. Global logical row is obtained by inverse map.
        #pragma unroll
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            // Decode vec_idx to [k0][m_phys].
            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;
            const int m_logi = a_row_phys_to_logi(m_phys);

            const int64_t a_row = kTailFree ? (block_m + m_logi) : ((m_logi < m_rows) ? (block_m + m_logi) : m_clamp_row);
            const int64_t a_k_raw = kk + k0 * kK1; // Start K position for this K0 slice
            const int64_t a_k = kTailFree ? a_k_raw : (a_k_raw < K ? a_k_raw : (K - kK1));
            const uint16_t* __restrict__ const a_src = a + a_row * K + a_k;

            uint16_t* __restrict__ const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);
            if constexpr (kABf16In) {
                const uint4 raw = *reinterpret_cast<const uint4*>(a_src);
                uint4 conv;
                conv.x = bf16x2_to_half2(raw.x);
                conv.y = bf16x2_to_half2(raw.y);
                conv.z = bf16x2_to_half2(raw.z);
                conv.w = bf16x2_to_half2(raw.w);
                *reinterpret_cast<uint4*>(sh_a_dst) = conv;
            } else {
                *reinterpret_cast<uint4*>(sh_a_dst) = *reinterpret_cast<const uint4*>(a_src);
            }
        }
    };

    // Loading B: NxK prepacked layout with vec16 loads
    constexpr int kBVecs = kBlockN;
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;

    const auto load_b_lds_prepacked = [&](const int stage, const int64_t kk) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            if (vec_idx >= kBVecs) continue;

            const int n_phys = vec_idx;
            const int64_t b_col = kTailFree ? static_cast<int64_t>(block_n + n_phys)
                                            : min(static_cast<int64_t>(block_n + n_phys), N - 1);
            const uint8_t* __restrict__ const b_src = b_prepacked + kk * N + b_col * kWmmaK;

            uint8_t* __restrict__ const sh_b_dst = &sh.ab.b[stage][n_phys][0];
            *reinterpret_cast<uint4*>(sh_b_dst) = *reinterpret_cast<const uint4*>(b_src);
        }
    };

    // Pipeline setup
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>((K + kChunkK - 1) / kChunkK);

    // WMMA compute lambda for one sub-iteration (one stage).
    // Register-tiling: load all B fragments once, then iterate rm with A loads.
    // Reduces LDS reads from (kRepeatM*kRepeatN)*(16+16) to kRepeatN*16 + kRepeatM*16.
    const auto wmma_compute_stage = [&](const int stage) -> void {
        if (wave_id >= kBlockWarpsM * kBlockWarpsN) return;

        using frag16_t = uint16_t __attribute__((ext_vector_type(16)));
        using fp32x8_t = float __attribute__((ext_vector_type(8)));

        const int lane_in_subgroup = lane % 16;

        // Pre-load all B fragments (one per rn tile)
        uint16_t all_reg_b[kRepeatN][16];
        #pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kBlockWarpsN;
            const int n_col = tile_n * kWmmaN + lane_in_subgroup;

            const uint4 p = *reinterpret_cast<const uint4*>(&sh.ab.b[stage][n_col][0]);
            const uint32_t p32[4] = {p.x, p.y, p.z, p.w};
            uint32_t h32[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp8x4_to_half2x2<kUseFp8E5M2, true>(p32[j], h32[2 * j], h32[2 * j + 1]);
            }
            *reinterpret_cast<uint4*>(&all_reg_b[rn][0]) = *reinterpret_cast<const uint4*>(&h32[0]);
            *reinterpret_cast<uint4*>(&all_reg_b[rn][8]) = *reinterpret_cast<const uint4*>(&h32[4]);
        }

        // For each rm: load A once, compute all rn tiles with cached B
        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            const int tile_m = warp_m + rm * kBlockWarpsM;
            const int m_row = tile_m * kWmmaM + lane_in_subgroup;

            uint16_t reg_a[16];
            #pragma unroll
            for (int k0 = 0; k0 < kK0; ++k0) {
                const uint16_t* __restrict__ const sh_a_src = sh_a_row_ptr(stage, k0, m_row);
                #pragma unroll
                for (int k1 = 0; k1 < kK1; ++k1) {
                    reg_a[k0 * kK1 + k1] = sh_a_src[k1];
                }
            }

            const frag16_t a_frag = *reinterpret_cast<const frag16_t*>(reg_a);

            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const frag16_t b_frag = *reinterpret_cast<const frag16_t*>(all_reg_b[rn]);
                fp32x8_t c_frag = *reinterpret_cast<fp32x8_t*>(&acc[repeat_idx][0]);

                c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

                *reinterpret_cast<fp32x8_t*>(&acc[repeat_idx][0]) = c_frag;
            }
        }
    };

    const auto load_chunk = [&](const int chunk) {
        const int64_t k_base = static_cast<int64_t>(chunk) * kChunkK;
        #pragma unroll
        for (int u = 0; u < kUnrollK; ++u) {
            load_a_lds_k0mk1(u, k_base + static_cast<int64_t>(u) * kWmmaK);
        }
        #pragma unroll
        for (int u = 0; u < kUnrollK; ++u) {
            load_b_lds_prepacked(u, k_base + static_cast<int64_t>(u) * kWmmaK);
        }
    };

    // Prologue: load first chunk into LDS
    load_chunk(0);
    __syncthreads();

    // Main loop
    for (int iter_idx = 0; iter_idx < total_chunks; ++iter_idx) {
        #pragma unroll
        for (int u = 0; u < kUnrollK; ++u) {
            wmma_compute_stage(u);
        }
        __syncthreads();

        if (iter_idx + 1 < total_chunks) {
            asm volatile("s_setprio 1" ::: "memory");
            load_chunk(iter_idx + 1);
            asm volatile("s_setprio 0" ::: "memory");
            __syncthreads();
        }
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        uint16_t* __restrict__ const sh_c = sh.c[wave_id][0];

        const float scale_f = has_scale ? scale[0] : 1.0f;
        const int subgroup = lane / 16;
        const int lane_in_subgroup = lane % 16;

        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const int tile_m = warp_m + rm * kBlockWarpsM;
                const int tile_n = warp_n + rn * kBlockWarpsN;
                const int64_t tile_m_base = block_m + tile_m * kWmmaM;
                const int64_t tile_n_base = block_n + tile_n * kWmmaN;
                // Whole-tile test first: a warp's tile is either fully live, partially live (M tail) or dead.
                if constexpr (!kTailFree) {
                    if (tile_m_base >= M || tile_n_base >= N) continue;
                }

                // Step 1: Write acc to LDS in column-major order (WMMA layout)
                // Each thread writes 8 values to one column. Rows past M are written too (garbage that
                // the store below drops), which keeps this loop free of divergence.
                const int col = lane_in_subgroup;
                #pragma unroll
                for (int acc_idx = 0; acc_idx < 8; ++acc_idx) {
                    const int row_logi = subgroup * 8 + acc_idx;
                    const int row_phys = c_row_logi_to_phys_16(row_logi);
                    sh_c[row_phys * kCStride + col] = float_to_output_bits<kOutputBf16>(acc[repeat_idx][acc_idx] * scale_f);
                }

                // Wave executes in lockstep (SIMT), so all writes complete before reads
                // No explicit barrier needed within a wave

                // Step 2: Read from LDS in row-major order for coalesced global write
                // 32 threads -> 16 rows, 2 threads per row, each handles 8 columns
                const int read_row = lane / 2;
                const int read_row_phys = c_row_logi_to_phys_16(read_row);
                const int col_half = lane % 2;  // 0 = cols 0-7, 1 = cols 8-15
                const int read_col_base = col_half * 8;

                const int64_t out_row = tile_m_base + read_row;
                const int64_t out_col = tile_n_base + read_col_base;
                if constexpr (!kTailFree) {
                    if (out_row >= M || out_col >= N) continue;
                }

                uint16_t* __restrict__ const out_ptr = c + out_row * N + out_col;
                uint16_t* __restrict__ const h = sh_c + read_row_phys * kCStride + read_col_base;

                if (has_bias) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        h[i] = add_output_bits<kOutputBf16>(h[i], bias[out_col + i]);
                    }
                }

                *reinterpret_cast<uint4*>(out_ptr) = *reinterpret_cast<uint4*>(h);
            }
        }
    }
}

} // namespace

template <int M, int N, int U, int RM, int RN>
struct ConfigTag {
    static constexpr int kBlockWarpsM = M;
    static constexpr int kBlockWarpsN = N;
    static constexpr int kUnrollK = U;
    static constexpr int kRepeatM = RM;
    static constexpr int kRepeatN = RN;
};

// Mixed-precision A WSGR is only robust in a few regions on the current ROCm nightly: small-K u=4
// repeat_m=1 kernels, tall u=2 repeat_m=2 kernels, and square u=2 repeat_m=4 kernels.
template <typename Config>
constexpr bool should_enable_a_wsgr(const int64_t K)
{
    if constexpr (Config::kBlockWarpsN == 1) {
        return false;
    }

    if constexpr (Config::kRepeatM == 1) {
        return Config::kUnrollK == 4 && K <= 1024;
    }

    if constexpr (Config::kRepeatM >= 8) {
        return false;
    }

    if constexpr (Config::kRepeatM == 2) {
        return Config::kBlockWarpsM > Config::kBlockWarpsN && Config::kUnrollK == 2;
    }

    if constexpr (Config::kRepeatM == 4) {
        return Config::kBlockWarpsM == Config::kBlockWarpsN && Config::kUnrollK == 2;
    }

    return false;
}

// Whether any K can enable WSGR for this config, i.e. whether the WSGR body is worth instantiating.
template <typename Config>
constexpr bool a_wsgr_instance_needed()
{
    if constexpr (Config::kBlockWarpsN == 1) {
        return false;
    }

    if constexpr (Config::kRepeatM == 1) {
        return Config::kUnrollK == 4;
    }

    if constexpr (Config::kRepeatM >= 8) {
        return false;
    }

    if constexpr (Config::kRepeatM == 2) {
        return Config::kBlockWarpsM > Config::kBlockWarpsN && Config::kUnrollK == 2;
    }

    if constexpr (Config::kRepeatM == 4) {
        return Config::kBlockWarpsM == Config::kBlockWarpsN && Config::kUnrollK == 2;
    }

    return false;
}

struct LaunchArgs {
    const uint16_t* a;
    const uint8_t* b_prepacked;
    const float* scale;
    const uint16_t* bias;
    uint16_t* c;
    int64_t M;
    int64_t N;
    int64_t K;
    int has_scale;
    int has_bias;
    bool use_fp8_e5m2;
    bool a_bf16;
    bool output_bf16;
    hipStream_t stream;
};

template <typename Config, bool kEnableAWsgr, bool kUseFp8E5M2, bool kABf16In, bool kOutputBf16, bool kTailFree>
void launch_kernel(const LaunchArgs& args) {
    constexpr int kBlockM = kWmmaM * Config::kBlockWarpsM * Config::kRepeatM;
    constexpr int kBlockN = kWmmaN * Config::kBlockWarpsN * Config::kRepeatN;
    constexpr int kThreadsPerBlock = kWaveSize * Config::kBlockWarpsM * Config::kBlockWarpsN;
    const dim3 block(kThreadsPerBlock, 1, 1);
    const dim3 grid(static_cast<uint32_t>((args.N + kBlockN - 1) / kBlockN),
                    static_cast<uint32_t>((args.M + kBlockM - 1) / kBlockM), 1);
    hipLaunchKernelGGL(
        (scaled_mm_kernel<Config::kBlockWarpsM, Config::kBlockWarpsN, Config::kUnrollK, Config::kRepeatM, Config::kRepeatN,
                          kEnableAWsgr, kUseFp8E5M2, kABf16In, kOutputBf16, kTailFree>),
        grid, block, 0, args.stream,
        args.a, args.b_prepacked, args.scale, args.bias, args.c,
        args.M, args.N, args.K, args.has_scale, args.has_bias);
}

// bf16 activations are widened to fp16 while staging, so one body serves both: the fp16 decode is the
// cheaper one and the widening is the conversion a tensor cast would do anyway. The native bf16 MMA
// was measured against this and lost on the wide cells.
template <typename Config, bool kEnableAWsgr, bool kUseFp8E5M2, bool kABf16In, bool kOutputBf16>
void launch_dtype(const LaunchArgs& args, const bool tail_free) {
    if (tail_free) {
        launch_kernel<Config, kEnableAWsgr, kUseFp8E5M2, kABf16In, kOutputBf16, true>(args);
    } else {
        launch_kernel<Config, kEnableAWsgr, kUseFp8E5M2, kABf16In, kOutputBf16, false>(args);
    }
}

template <typename Config, bool kEnableAWsgr, bool kUseFp8E5M2>
void launch_for_a_dtype(const LaunchArgs& args) {
    constexpr int kBlockM = kWmmaM * Config::kBlockWarpsM * Config::kRepeatM;
    constexpr int kBlockN = kWmmaN * Config::kBlockWarpsN * Config::kRepeatN;
    constexpr int kChunkK = kWmmaK * Config::kUnrollK;
    const bool tail_free = args.M % kBlockM == 0 && args.N % kBlockN == 0 && args.K % kChunkK == 0;
    if (args.a_bf16) {
        if (args.output_bf16) {
            launch_dtype<Config, kEnableAWsgr, kUseFp8E5M2, true, true>(args, tail_free);
        } else {
            launch_dtype<Config, kEnableAWsgr, kUseFp8E5M2, true, false>(args, tail_free);
        }
    } else if (args.output_bf16) {
        launch_dtype<Config, kEnableAWsgr, kUseFp8E5M2, false, true>(args, tail_free);
    } else {
        launch_dtype<Config, kEnableAWsgr, kUseFp8E5M2, false, false>(args, tail_free);
    }
}

template <typename Config>
void launch_scaled_mm_configured(const LaunchArgs& args) {
    if (args.use_fp8_e5m2) {
        if constexpr (a_wsgr_instance_needed<Config>()) {
            if (should_enable_a_wsgr<Config>(args.K)) {
                launch_for_a_dtype<Config, true, true>(args);
                return;
            }
        }
        launch_for_a_dtype<Config, false, true>(args);
    } else {
        if constexpr (a_wsgr_instance_needed<Config>()) {
            if (should_enable_a_wsgr<Config>(args.K)) {
                launch_for_a_dtype<Config, true, false>(args);
                return;
            }
        }
        launch_for_a_dtype<Config, false, false>(args);
    }
}

// One row per instantiated tile: (warps_m, warps_n, unroll_k, repeat_m, repeat_n). This is the search
// space the heuristic in feather_ops/scaled_mm.py selects from.
#define FEATHER_SCALED_MM_TILES(X) \
    X(1, 1, 2, 1, 2)               \
    X(1, 1, 4, 1, 2)               \
    X(1, 2, 2, 1, 2)               \
    X(1, 2, 4, 1, 2)               \
    X(1, 4, 2, 1, 2)               \
    X(1, 4, 4, 1, 2)               \
    X(1, 8, 2, 1, 2)               \
    X(1, 8, 4, 1, 2)               \
    X(1, 1, 2, 2, 2)               \
    X(1, 1, 4, 2, 2)               \
    X(1, 1, 2, 4, 4)               \
    X(1, 1, 4, 4, 4)               \
    X(1, 2, 2, 2, 2)               \
    X(1, 2, 4, 2, 2)               \
    X(2, 1, 2, 2, 2)               \
    X(2, 2, 2, 2, 2)               \
    X(2, 1, 4, 2, 2)               \
    X(1, 4, 2, 4, 2)               \
    X(1, 4, 4, 4, 2)               \
    X(1, 8, 2, 8, 2)               \
    X(1, 8, 4, 8, 2)               \
    X(2, 2, 2, 4, 4)               \
    X(2, 2, 4, 4, 4)               \
    X(2, 4, 2, 4, 2)               \
    X(2, 4, 4, 4, 2)               \
    X(2, 4, 2, 4, 4)               \
    X(2, 4, 4, 4, 4)               \
    X(4, 2, 2, 2, 4)               \
    X(4, 2, 4, 2, 4)               \
    X(4, 4, 2, 4, 2)               \
    X(4, 4, 2, 4, 4)

struct TileQuery {
    int block_warps_m;
    int block_warps_n;
    int unroll_k;
    int repeat_m;
    int repeat_n;
};

bool dispatch_scaled_mm(const LaunchArgs& args, const TileQuery& tile)
{
    const auto try_launch = [&](auto tag) -> bool {
        using Config = decltype(tag);
        if (tile.block_warps_m != Config::kBlockWarpsM || tile.block_warps_n != Config::kBlockWarpsN ||
            tile.unroll_k != Config::kUnrollK || tile.repeat_m != Config::kRepeatM ||
            tile.repeat_n != Config::kRepeatN) {
            return false;
        }
        launch_scaled_mm_configured<Config>(args);
        return true;
    };

#define FEATHER_TRY_TILE(m, n, u, rm, rn) || try_launch(ConfigTag<m, n, u, rm, rn>{})
    return false FEATHER_SCALED_MM_TILES(FEATHER_TRY_TILE);
#undef FEATHER_TRY_TILE
}

// Standalone C ABI for the C++ benchmarks in cpp_benchmarks/, which do not link torch.
extern "C" bool launch_scaled_mm(
    const uint16_t* const a,
    const uint8_t* const b_prepacked,
    const float* const scale,
    const uint16_t* const bias,
    uint16_t* const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int has_scale,
    const int has_bias,
    const bool use_fp8_e5m2,
    const bool a_bf16,
    const bool output_bf16,
    const int block_warps_m,
    const int block_warps_n,
    const int unroll_k,
    const int repeat_m,
    const int repeat_n,
    hipStream_t stream)
{
    const LaunchArgs args{a, b_prepacked, scale, bias, c, M, N, K, has_scale, has_bias,
                          use_fp8_e5m2, a_bf16, output_bf16, stream};
    const TileQuery tile{block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n};
    return dispatch_scaled_mm(args, tile);
}

#ifndef NO_PYTORCH
void scaled_mm(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b_prepacked,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& bias,
    torch::stable::Tensor& c,
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

    const auto a_scalar_type = a.scalar_type();
    STD_TORCH_CHECK(a_scalar_type == torch::stable::ScalarType::Half || a_scalar_type == torch::stable::ScalarType::BFloat16,
                    "a must be float16 or bfloat16");
    STD_TORCH_CHECK(c.scalar_type() == torch::stable::ScalarType::Half || c.scalar_type() == torch::stable::ScalarType::BFloat16,
                    "c must be float16 or bfloat16");
    const auto b_scalar_type = b_prepacked.scalar_type();
    STD_TORCH_CHECK(b_scalar_type == torch::stable::ScalarType::Float8_e4m3fn || b_scalar_type == torch::stable::ScalarType::Float8_e5m2,
                    "b_prepacked must be float8_e4m3fn or float8_e5m2");
    const bool use_fp8_e5m2 = b_scalar_type == torch::stable::ScalarType::Float8_e5m2;
    const bool a_bf16 = a_scalar_type == torch::stable::ScalarType::BFloat16;
    const bool output_bf16 = c.scalar_type() == torch::stable::ScalarType::BFloat16;

    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(b_prepacked.dim() == 3, "b_prepacked must be 3D (K/16, N, 16)");
    STD_TORCH_CHECK(c.dim() == 2, "c must be 2D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b_prepacked.size(1);
    STD_TORCH_CHECK(K % kWmmaK == 0, "K must be divisible by 16");
    STD_TORCH_CHECK(N % kWmmaN == 0, "N must be divisible by 16");
    STD_TORCH_CHECK(b_prepacked.size(0) >= K / 16, "b_prepacked.shape[0] must be at least K/16 (", K / 16, ")");
    STD_TORCH_CHECK(b_prepacked.size(0) % (128 / 16) == 0, "b_prepacked.shape[0] must be a multiple of 8 (K padded to 128)");
    STD_TORCH_CHECK(b_prepacked.size(2) == 16, "b_prepacked.shape[2] must be 16");
    STD_TORCH_CHECK(c.size(0) == M, "c.shape[0] must equal M");
    STD_TORCH_CHECK(c.size(1) == N, "c.shape[1] must equal N");

    // Contiguous fast path requirements
    STD_TORCH_CHECK(a.stride(0) == K, "a.stride(0) must equal K (", K, ")");
    STD_TORCH_CHECK(a.stride(1) == 1, "a.stride(1) must be 1");
    STD_TORCH_CHECK(b_prepacked.stride(0) == N * 16, "b_prepacked.stride(0) must equal N*16 (", N * 16, ")");
    STD_TORCH_CHECK(b_prepacked.stride(1) == 16, "b_prepacked.stride(1) must be 16");
    STD_TORCH_CHECK(b_prepacked.stride(2) == 1, "b_prepacked.stride(2) must be 1");
    STD_TORCH_CHECK(c.stride(0) == N, "c.stride(0) must equal N (", N, ")");
    STD_TORCH_CHECK(c.stride(1) == 1, "c.stride(1) must be 1");

    if (scale.has_value()) {
        const auto& scale_t = *scale;
        STD_TORCH_CHECK(scale_t.is_cuda(), "scale must be a CUDA tensor");
        STD_TORCH_CHECK(scale_t.get_device_index() == device_index, "scale must be on the same device as a");
        STD_TORCH_CHECK(scale_t.scalar_type() == torch::stable::ScalarType::Float, "scale must be float32");
        STD_TORCH_CHECK(scale_t.numel() == 1, "scale must have one element");
    }
    if (bias.has_value()) {
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as a");
        STD_TORCH_CHECK(bias_t.scalar_type() == c.scalar_type(), "bias must have the same dtype as c");
        STD_TORCH_CHECK(bias_t.numel() == N, "bias must have N elements");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const auto is_aligned_16 = [](const void* const p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
    };
    STD_TORCH_CHECK(is_aligned_16(a.const_data_ptr()), "a data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(b_prepacked.const_data_ptr()), "b_prepacked data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(c.mutable_data_ptr()), "c data pointer must be 16-byte aligned");

    const int64_t threads_per_block = kWaveSize * block_warps_m * block_warps_n;
    STD_TORCH_CHECK(threads_per_block <= 1024, "Block size exceeds HIP thread-per-block limit");

    const TileQuery tile{static_cast<int>(block_warps_m), static_cast<int>(block_warps_n), static_cast<int>(unroll_k),
                         static_cast<int>(repeat_m), static_cast<int>(repeat_n)};
    LaunchArgs args{
        reinterpret_cast<const uint16_t*>(a.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(b_prepacked.const_data_ptr()),
        scale.has_value() ? reinterpret_cast<const float*>(scale->const_data_ptr()) : nullptr,
        bias.has_value() ? reinterpret_cast<const uint16_t*>(bias->const_data_ptr()) : nullptr,
        reinterpret_cast<uint16_t*>(c.mutable_data_ptr()),
        M, N, K,
        scale.has_value() ? 1 : 0,
        bias.has_value() ? 1 : 0,
        use_fp8_e5m2,
        a_bf16,
        output_bf16,
        stream};
    const bool launched = dispatch_scaled_mm(args, tile);
    STD_TORCH_CHECK(launched, "Unsupported config");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "scaled_mm kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(feather_ops, m)
{
    m.def(
        "scaled_mm("
        "Tensor a, "
        "Tensor b_prepacked, "
        "Tensor? scale, "
        "Tensor? bias, "
        "Tensor(a!) c, "
        "int block_warps_m, "
        "int block_warps_n, "
        "int unroll_k, "
        "int repeat_m, "
        "int repeat_n"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("scaled_mm", TORCH_BOX(&scaled_mm));
}
#endif
