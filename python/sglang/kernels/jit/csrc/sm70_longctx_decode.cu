// SPDX-License-Identifier: Apache-2.0
// Long-context grouped decode for Volta (SM70), D256 G6 (H6/Hkv1), E5M2 KV.
//
// Read-once split-KV design: grid = (kv_heads, splits, batch). Each CTA owns
// one kv head and one context partition, streaming that partition's K/V once
// with 16-byte vectorized loads from the dense token-major layout. This keeps
// DRAM traffic at the theoretical minimum (unlike per-query-head partition
// schemes that re-read the same KV once per GQA head). A split-merge kernel
// (the existing TileLang combine) rescales the FP32 softmax states, with the
// same partial-output ABI. Numerics mirror _kernels_paged_decode.py: scores
// are scaled by softmax_scale*k_scale, probabilities are exp2((s-m)*scale*log2e),
// and the partial output is normalized by the partition sum and v_scale.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>

namespace sm70_longctx {

constexpr int kThreads = 512;
constexpr int kBlockN = 32;          // tokens per tile (smem-bounded)
constexpr int kDim = 256;
constexpr int kPairsPerLane = (kDim / 2) / 32;  // half2 pairs per lane (4)
constexpr int kAccPerLane = 2 * kPairsPerLane;  // fp32 accumulators per lane
constexpr int kGroup = 6;            // heads / kv_heads (TP4 GQA layout)
constexpr int kPageSize = 16;        // KV cache tokens per page
constexpr int kKVStride = 264;       // padded smem row stride (half units)
constexpr int kSmemKV = kBlockN * kKVStride;  // halves per K or V tile
constexpr float kLog2E = 1.4426950408889634f;

constexpr int kIndexerThreads = 256;
constexpr int kIndexerWarps = kIndexerThreads / 32;
constexpr int kIndexerHeads = 4;
constexpr int kIndexerDim = 128;
constexpr int kIndexerPageSize = 4;
constexpr int kIndexerKeysPerWarp = 8;
constexpr int kIndexerKeysPerBlock = kIndexerWarps * kIndexerKeysPerWarp;

// smem (bytes): ks + vs (fp16 tiles) + qs + scores (fp32) + probs (fp16)
constexpr int kSmemBytes = 2 * kSmemKV * sizeof(__half) +
                           kGroup * kDim * sizeof(__half) +
                           kGroup * kBlockN * sizeof(float) +
                           kGroup * kBlockN * sizeof(__half);

__device__ __forceinline__ void e5m2_to_fp16_16(const uint4 raw, half* dst) {
  // E5M2 shares sign/exponent bit positions with FP16: each byte shifted left
  // by 8 is an exact value conversion, and the 16 output halves must keep the
  // same dim order as the 16 input bytes. Each 32-bit input word b = [b0..b3]
  // produces one uint32 pair [fp16(b0), fp16(b1)] and one [fp16(b2), fp16(b3)];
  // the pairs go to consecutive uint32 slots so dim order is preserved.
  const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
  uint32_t out[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t b = r[w];
    out[2 * w] = (b & 0x000000ffu) << 8 | ((b >> 8) & 0x000000ffu) << 24;
    out[2 * w + 1] = ((b >> 16) & 0x000000ffu) << 8 |
                     ((b >> 24) & 0x000000ffu) << 24;
  }
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(out);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(out + 4);
}

// E4M3 shares no exponent layout with FP16 (4-bit exponent, bias 7, three
// mantissa bits), so the byte-shift trick above is invalid for it. Widen each
// byte through a 256-entry FP16 lookup that the host precomputes once; the
// table is 512 bytes and stays in cache. Same 16-values-per-thread layout.
__device__ __forceinline__ void e4m3_lut_to_fp16_16(
    const uint4 raw, half* dst, const __half* __restrict__ lut) {
  const uint8_t* r = reinterpret_cast<const uint8_t*>(&raw);
  __half out[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = lut[r[i]];
  }
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(out);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(out + 8);
}

// Byte-cache conversion selected by the LUT pointer: E4M3 when provided,
// E5M2 (byte shift) otherwise. FP16 caches never reach this helper.
__device__ __forceinline__ void kv8_to_fp16_16(
    const uint4 raw, half* dst, const __half* __restrict__ lut) {
  if (lut != nullptr) {
    e4m3_lut_to_fp16_16(raw, dst, lut);
  } else {
    e5m2_to_fp16_16(raw, dst);
  }
}

// Load one K/V token tile into shared memory. The cache element type selects
// the read path: the E5M2 byte cache is read as one 16-byte vector per thread
// and dequantized to FP16; an FP16 cache is read as two 16-byte vectors per
// thread and copied directly. Both write 16 FP16 values per thread to the same
// smem row layout, so the QK/softmax/PV math that reads ks/vs is dtype-agnostic.
template <typename KV>
__device__ __forceinline__ void load_kv_tile(const KV* __restrict__ k_cache,
                                             const KV* __restrict__ v_cache,
                                             const int* __restrict__ slots,
                                             const int tile_tokens, const int tid,
                                             const __half* __restrict__ lut,
                                             __half* __restrict__ ks,
                                             __half* __restrict__ vs) {
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
#pragma unroll
  for (int it = 0; it < kLoadIters; ++it) {
    const int vector = tid + it * kThreads;
    const int off = vector << 4;       // 16 elements per thread
    const int tok_local = off >> 8;    // 256 elements per token row
    const int d_off = off & 255;       // dim offset, 16-aligned
    const int dst = tok_local * kKVStride + d_off;
    const int slot = slots[tok_local];
    uint4* dk = reinterpret_cast<uint4*>(ks + dst);
    uint4* dv = reinterpret_cast<uint4*>(vs + dst);
    if (tok_local < tile_tokens && slot >= 0) {
      const int64_t base = (int64_t)slot * kDim + d_off;
      if constexpr (sizeof(KV) == 1) {
        // Byte cache (E5M2 or E4M3): base is in bytes, one 16-byte vector
        // per thread; the LUT pointer selects the conversion.
        kv8_to_fp16_16(*reinterpret_cast<const uint4*>(k_cache + base),
                       ks + dst, lut);
        kv8_to_fp16_16(*reinterpret_cast<const uint4*>(v_cache + base),
                       vs + dst, lut);
      } else {
        // FP16 cache: base is in halves, two 16-byte vectors per thread.
        const uint4* sk = reinterpret_cast<const uint4*>(k_cache + base);
        const uint4* sv = reinterpret_cast<const uint4*>(v_cache + base);
        dk[0] = sk[0];
        dk[1] = sk[1];
        dv[0] = sv[0];
        dv[1] = sv[1];
      }
    } else {
      // Out-of-range token or invalid slot: zero the 16 FP16 smem slots,
      // matching the E5M2 path's zero dequant.
      const uint4 zero = make_uint4(0, 0, 0, 0);
      dk[0] = zero;
      dk[1] = zero;
      dv[0] = zero;
      dv[1] = zero;
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
decode_partial_kernel(const __half* __restrict__ q,
                      const uint8_t* __restrict__ k_cache,
                      const uint8_t* __restrict__ v_cache,
                      const int* __restrict__ page_table,
                      const int* __restrict__ seq_lens, const int max_splits,
                      const int min_tokens_per_split, const int max_blocks,
                      const float score_scale, const float kv_scale,
                      __half* __restrict__ partial_o,
                      float* __restrict__ partial_lse) {
  const int kv_head = blockIdx.x;
  const int kv_heads = gridDim.x;
  const int kv_stride = kDim * kv_heads;   // per-token bytes across KV heads
  const int heads_total = kGroup * kv_heads;
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;

  const int context = seq_lens[seq_id];
  const int active_splits =
      min(max_splits, max(1, (context + min_tokens_per_split - 1) /
                                 min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  const int split_len = (context + active_splits - 1) / active_splits;
  const int split_begin = split_id * split_len;
  const int split_end = min(context, split_begin + split_len);
  if (split_begin >= split_end) {
    return;
  }
  const float scale_log2 = score_scale * kLog2E;

  // Load the six query rows for this kv head into shared memory.
  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    const int r = i / kDim;
    const int d = i % kDim;
    qs[r * kDim + d] = q[(kv_head * kGroup + r) * kDim + d];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];  // fp32 accumulators per lane (dim pair)
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);

    // ---- Cooperative vectorized load + E5M2 -> FP16 convert (K and V) ----
    // kBlockN tokens * 256 bytes = 8KB/tensor. Each thread: 16B (one uint4)
    // per tensor. slot = tid + it*512; byte_off = slot*16; token = byte>>8;
    // d_off = byte & 255. Consecutive threads read consecutive 16B chunks of
    // a 256B token row -> fully coalesced 128B transactions.
#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int slot = tid + it * kThreads;   // 16-byte units [0, kBlockN*16)
      const int byte_off = slot << 4;
      const int tok_local = byte_off >> 8;
      const int d_off = byte_off & 255;
      const int token = tile_begin + tok_local;
      const bool valid = (token < split_end) && (tok_local < kBlockN);
      uint4 raw_k = make_uint4(0, 0, 0, 0);
      uint4 raw_v = make_uint4(0, 0, 0, 0);
      if (valid) {
        const int64_t base =
            page_table != nullptr
                ? (int64_t)page_table[seq_id * max_blocks + (token >> 4)] *
                          (kPageSize * kv_stride) +
                      (int64_t)(token & (kPageSize - 1)) * kv_stride +
                      kv_head * kDim + d_off
                : (int64_t)token * kv_stride + kv_head * kDim + d_off;
        raw_k = *reinterpret_cast<const uint4*>(k_cache + base);
        raw_v = *reinterpret_cast<const uint4*>(v_cache + base);
      }
      const int dst = tok_local * kKVStride + d_off;
      e5m2_to_fp16_16(raw_k, ks + dst);
      e5m2_to_fp16_16(raw_v, vs + dst);
    }
    __syncthreads();

    // ---- QK: each compute warp owns one query row ----
    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      {
        const int n = lane;  // kBlockN == 32 == warp width
        const __half2* krow =
            reinterpret_cast<const __half2*>(ks + n * kKVStride);
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
        for (int h = 0; h < kDim / 8; ++h) {
          const __half2 qh0 = qr[4 * h];
          const __half2 kh0 = krow[4 * h];
          const __half2 qh1 = qr[4 * h + 1];
          const __half2 kh1 = krow[4 * h + 1];
          const __half2 qh2 = qr[4 * h + 2];
          const __half2 kh2 = krow[4 * h + 2];
          const __half2 qh3 = qr[4 * h + 3];
          const __half2 kh3 = krow[4 * h + 3];
          acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
          acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
          acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
          acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
          acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
          acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
          acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
          acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
        }
        const float acc = acc0 + acc1 + acc2 + acc3;
        if (n < tile_tokens) {
          scores[r * kBlockN + n] = acc;
        }
      }
      __syncwarp();
      float loc_max = -1.0e30f;
      if (lane < tile_tokens) {
        loc_max = scores[r * kBlockN + lane];
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float loc_sum = 0.f;
      {
        const int n = lane;
        float p = 0.f;
        if (n < tile_tokens) {
          p = exp2f((scores[r * kBlockN + n] - m_new) * scale_log2);
          probs[r * kBlockN + n] = __float2half(p);
        }
        loc_sum = p;
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    // ---- PV: o_acc += p * V ----
    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 v = vrow[lane + j * 32];
          o_acc[r][2 * j] = fmaf(p, __half2float(v.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(v.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  // ---- Finalize partial output for this partition ----
  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    const int oh = kv_head * kGroup + r;
    __half* o_row = partial_o +
                    (((int64_t)seq_id * max_splits + split_id) * heads_total +
                     oh) *
                        kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      o_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[r][2 * j] * inv_l * kv_scale);
      o_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l * kv_scale);
    }
  }
  if (is_compute_warp && lane == 0) {
    const int r = warp;
    const float lse =
        l_row[r] > 0.f ? __log2f(l_row[r]) + m_row[r] * scale_log2 : -1.0e30f;
    partial_lse[((int64_t)seq_id * max_splits + split_id) * heads_total +
                kv_head * kGroup + r] = lse;
  }
}

// Split-KV QSA decode for the TP4 H6/Hkv1/D256 layout.  Unlike the generic
// fallback, this resolves the selected logical positions while loading the
// cache and keeps K/V shared by all six query heads.  No FP16 compact-KV
// scratch is written or read.
template <typename KV>
__global__ void __launch_bounds__(kThreads, 1)
qsa_decode_partial_kernel(
    const __half* __restrict__ q, const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    const int* __restrict__ req_to_token,
    const int* __restrict__ req_indices, const int* __restrict__ indices,
    const int* __restrict__ seq_lens, const int req_stride, const int topk,
    const int max_splits, const int min_tokens_per_split,
    const float score_scale, const __half* __restrict__ lut,
    __half* __restrict__ partial_o,
    float* __restrict__ partial_lse) {
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;
  const int seq_len = seq_lens[seq_id];
  const int context = min(seq_len, topk);
  const int active_splits =
      min(max_splits, max(1, (context + min_tokens_per_split - 1) /
                                 min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  const int split_len = (context + active_splits - 1) / active_splits;
  const int split_begin = split_id * split_len;
  const int split_end = min(context, split_begin + split_len);
  if (split_begin >= split_end) {
    return;
  }

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const int req_index = req_indices[seq_id];
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)seq_id * kGroup * kDim + i];
  }

  // Each compute warp owns one query head. Thread-local arrays indexed by
  // the runtime warp ID force these accumulators into local memory on SM70.
  // Keep only this thread's head state so ptxas can retain it in registers.
  float m_row = -1.0e30f;
  float l_row = 0.f;
  float o_acc[kAccPerLane];
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < split_end) {
        const int logical = indices[(int64_t)seq_id * topk + selected];
        if (logical >= 0 && logical < seq_len) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

    load_kv_tile<KV>(k_cache, v_cache, slots, tile_tokens, tid, lut, ks, vs);
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      const int n = lane;
      const __half2* krow =
          reinterpret_cast<const __half2*>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] =
          (n < tile_tokens && slots[n] >= 0) ? acc0 + acc1 + acc2 + acc3
                                             : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row, loc_max);
      const float alpha = exp2f((m_row - m_new) * scale_log2);
      m_row = m_new;
      l_row *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[2 * j] =
              fmaf(p, __half2float(value.x), o_acc[2 * j]);
          o_acc[2 * j + 1] =
              fmaf(p, __half2float(value.y), o_acc[2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row > 0.f ? 1.f / l_row : 0.f;
    __half* out_row =
        partial_o +
        (((int64_t)seq_id * max_splits + split_id) * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[2 * j + 1] * inv_l);
    }
    if (lane == 0) {
      partial_lse[((int64_t)seq_id * max_splits + split_id) * kGroup + r] =
          l_row > 0.f
              ? __log2f(l_row) + m_row * scale_log2
              : -1.0e30f;
    }
  }
}

// QSA indexer decode scoring for Qwen3.8.  A warp owns one compressed key,
// reads its 128 FP16 values once, and accumulates the four query-head scores
// in registers.  This avoids padding four real heads to Volta MMA's 16-head
// tile and emits the exact FP32 logits consumed by fast_topk.
__global__ void __launch_bounds__(kIndexerThreads, 2)
qsa_indexer_decode_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ context_lens, const int max_pages,
    const int max_model_len, const float score_scale,
    float* __restrict__ logits) {
  const int seq_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int context = context_lens[seq_id];
  const int block_begin = blockIdx.x * kIndexerKeysPerBlock;
  if (block_begin >= context) {
    return;
  }

  __shared__ __half qs[kIndexerHeads * kIndexerDim];
  for (int i = tid; i < kIndexerHeads * kIndexerDim; i += kIndexerThreads) {
    qs[i] = q[(int64_t)seq_id * kIndexerHeads * kIndexerDim + i];
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < kIndexerKeysPerWarp; ++it) {
    const int position = block_begin + it * kIndexerWarps + warp;
    if (position >= context || position >= max_model_len) {
      continue;
    }
    const int logical_page = position / kIndexerPageSize;
    const int page_offset = position % kIndexerPageSize;
    const int physical_page =
        page_table[(int64_t)seq_id * max_pages + logical_page];
    const __half2* key = reinterpret_cast<const __half2*>(
        k_cache +
        ((int64_t)physical_page * kIndexerPageSize + page_offset) *
            kIndexerDim);
    const __half2 k0 = key[lane * 2];
    const __half2 k1 = key[lane * 2 + 1];
    float scores[kIndexerHeads];
#pragma unroll
    for (int h = 0; h < kIndexerHeads; ++h) {
      const __half2* qh =
          reinterpret_cast<const __half2*>(qs + h * kIndexerDim);
      const __half2 q0 = qh[lane * 2];
      const __half2 q1 = qh[lane * 2 + 1];
      float acc = 0.f;
      acc = fmaf(__half2float(q0.x), __half2float(k0.x), acc);
      acc = fmaf(__half2float(q0.y), __half2float(k0.y), acc);
      acc = fmaf(__half2float(q1.x), __half2float(k1.x), acc);
      acc = fmaf(__half2float(q1.y), __half2float(k1.y), acc);
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
      }
      scores[h] = acc;
    }
    if (lane == 0) {
      float score = 0.f;
#pragma unroll
      for (int h = 0; h < kIndexerHeads; ++h) {
        score += fmaxf(scores[h], 0.f);
      }
      logits[(int64_t)seq_id * max_model_len + position] = score / score_scale;
    }
  }
}

// QSA chunk-prefill for the exact TP4 full-attention shape.  One CTA owns one
// query token and evaluates all six GQA heads together, so every selected FP8
// K/V row is fetched and decoded once instead of once per query head.  The
// logical QSA indices are resolved through req_to_token in the load phase;
// this avoids materializing and converting the entire accumulated context on
// every 8K serving chunk.
__global__ void __launch_bounds__(kThreads, 1)
qsa_prefill_kernel(const __half* __restrict__ q,
                   const uint8_t* __restrict__ k_cache,
                   const uint8_t* __restrict__ v_cache,
                   const int* __restrict__ req_to_token,
                   const int* __restrict__ req_indices,
                   const int* __restrict__ indices,
                   const int* __restrict__ seq_lens, const int req_stride,
                   const int topk,
                   const float score_scale, const __half* __restrict__ lut,
                   __half* __restrict__ output) {
  const int query = blockIdx.x;
  const int req_index = req_indices[0];
  const int seq_len = seq_lens[0];
  // seq_len includes the complete serving chunk.  QSA stores each row's
  // causal candidates first, so mirror _sparse_gqa_chunk_prefill and consume
  // only the prefix visible to this query row.
  const int visible = min(seq_len, query + seq_len - gridDim.x + 1);
  const int row_topk = min(topk, max(0, visible));

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)query * kGroup * kDim + i];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (row_topk + kBlockN - 1) / kBlockN;
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = tile * kBlockN;
    const int tile_tokens = min(kBlockN, row_topk - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < row_topk) {
        const int logical = indices[(int64_t)query * topk + selected];
        if (logical >= 0 && logical < visible) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int vector = tid + it * kThreads;
      const int byte_off = vector << 4;
      const int tok_local = byte_off >> 8;
      const int d_off = byte_off & 255;
      const int slot = slots[tok_local];
      uint4 raw_k = make_uint4(0, 0, 0, 0);
      uint4 raw_v = make_uint4(0, 0, 0, 0);
      if (tok_local < tile_tokens && slot >= 0) {
        const int64_t base = (int64_t)slot * kDim + d_off;
        raw_k = *reinterpret_cast<const uint4*>(k_cache + base);
        raw_v = *reinterpret_cast<const uint4*>(v_cache + base);
      }
      const int dst = tok_local * kKVStride + d_off;
      kv8_to_fp16_16(raw_k, ks + dst, lut);
      kv8_to_fp16_16(raw_v, vs + dst, lut);
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      const int n = lane;
      const __half2* krow =
          reinterpret_cast<const __half2*>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] =
          (n < tile_tokens && slots[n] >= 0) ? acc0 + acc1 + acc2 + acc3
                                             : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[r][2 * j] =
              fmaf(p, __half2float(value.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(value.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    __half* out_row =
        output + ((int64_t)query * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[r][2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l);
    }
  }
}

// ---- Tensor-core variant of the chunk-prefill (SGLANG_SM70_QSA_MMA=1) ----
// Same load phase and online softmax as qsa_prefill_kernel; the score and
// p·V products run on mma.sync.m8n8k4 (Volta: four independent 8x8x4 products
// per warp, one per quad-pair). Fragment layout (matches turbomind's
// SM70_MMA_884): thread jj = (lane&3) + 4*(lane>>4) of quad-pair
// qp = (lane>>2)&3 holds A row jj k0..3 / B col jj k0..3; C register i is
// row (lane&1)+4*(lane>>4)+2*((i>>1)&1), column (lane&2)+4*(i>>2)+(i&1).
// QK: warps 0..3 each own 64 dims (16 k-steps of 4); QP q covers cells
// 8q..8q+7; two 8-row blocks cover the 16 padded head rows (6 real). The
// query is already FP16 so no hi/lo split is needed. PV: warp w owns dims
// [64w, 64w+64); QP q covers 16 dims as two n-tiles of 8; k walks the 32
// cells in steps of 4. Padded rows load zero probabilities, so their
// accumulators stay zero and are skipped at the write-out.
__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1,
                                       uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
        "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
  __trap();  // m8n8k4 is a Volta instruction; host only launches on sm_70
#endif
}

constexpr int kQRows = 16;                      // 6 heads padded to 16
constexpr int kPartStride = kBlockN + 2;        // float partials (34)
constexpr int kProbStride = kBlockN + 4;        // halves, 8B-aligned rows (72B)

__global__ void __launch_bounds__(kThreads, 1)
qsa_prefill_kernel_mma(const __half* __restrict__ q,
                       const uint8_t* __restrict__ k_cache,
                       const uint8_t* __restrict__ v_cache,
                       const int* __restrict__ req_to_token,
                       const int* __restrict__ req_indices,
                       const int* __restrict__ indices,
                       const int* __restrict__ seq_lens, const int req_stride,
                       const int topk, const float score_scale,
                       const __half* __restrict__ lut,
                       __half* __restrict__ output) {
  const int query = blockIdx.x;
  const int req_index = req_indices[0];
  const int seq_len = seq_lens[0];
  const int visible = min(seq_len, query + seq_len - gridDim.x + 1);
  const int row_topk = min(topk, max(0, visible));

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qh[kGroup * kDim];
  __shared__ float part[4][kQRows][kPartStride];
  __shared__ __half probs16[kQRows * kProbStride];  // rows 6..15 stay zero
  __shared__ int slots[kBlockN];
  __shared__ float alpha16[kQRows];
  __shared__ float lsum16[kQRows];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const float scale_log2 = score_scale * kLog2E;
  const int qp = (lane >> 2) & 3;
  const int jj = (lane & 3) + 4 * (lane >> 4);
  const int crow = (lane & 1) + 4 * (lane >> 4);
  const int ccol = lane & 2;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qh[i] = q[(int64_t)query * kGroup * kDim + i];
  }
  // Padded head rows (6..15) keep zero probabilities for the whole kernel:
  // the PV A-fragments of those rows read zeros, so their accumulators stay
  // zero and the write-out skips them. Rows 0..5 are rewritten per tile.
  for (int i = tid; i < kQRows * kProbStride; i += kThreads) {
    probs16[i] = __float2half(0.f);
  }
  if (tid < kQRows) {
    alpha16[tid] = 0.f;
    lsum16[tid] = 0.f;
  }

  float m_row[kGroup];
  float l_row[kGroup];
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  float acc[2][2][8];  // [row block][n tile][C registers]
#pragma unroll
  for (int rb = 0; rb < 2; ++rb)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[rb][nt][i] = 0.f;
      }
  __syncthreads();

  const int num_tiles = (row_topk + kBlockN - 1) / kBlockN;
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = tile * kBlockN;
    const int tile_tokens = min(kBlockN, row_topk - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < row_topk) {
        const int logical = indices[(int64_t)query * topk + selected];
        if (logical >= 0 && logical < visible) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int vector = tid + it * kThreads;
      const int byte_off = vector << 4;
      const int tok_local = byte_off >> 8;
      const int d_off = byte_off & 255;
      const int slot = slots[tok_local];
      uint4 raw_k = make_uint4(0, 0, 0, 0);
      uint4 raw_v = make_uint4(0, 0, 0, 0);
      if (tok_local < tile_tokens && slot >= 0) {
        const int64_t base = (int64_t)slot * kDim + d_off;
        raw_k = *reinterpret_cast<const uint4*>(k_cache + base);
        raw_v = *reinterpret_cast<const uint4*>(v_cache + base);
      }
      const int dst = tok_local * kKVStride + d_off;
      kv8_to_fp16_16(raw_k, ks + dst, lut);
      kv8_to_fp16_16(raw_v, vs + dst, lut);
    }
    __syncthreads();

    // ---- QK on tensor cores: warp w owns dims [64w, 64w+64) ----
    if (warp < 4) {
      float tg[2][8];
#pragma unroll
      for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          tg[rb][i] = 0.f;
        }
      const int cell_b = qp * 8 + jj;
#pragma unroll
      for (int kstep = 0; kstep < kDim / 4 / 4; ++kstep) {  // 16 k-steps
        const int d0 = warp * 64 + kstep * 4;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
            ks + cell_b * kKVStride + d0);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
            ks + cell_b * kKVStride + d0 + 2);
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          // Rows 6..15 are zero padding: the query smem only holds the six
          // real heads, so out-of-range rows contribute exact zeros.
          uint2 ah = make_uint2(0u, 0u);
          if (rb * 8 + jj < kGroup) {
            ah = *reinterpret_cast<const uint2*>(
                &qh[(rb * 8 + jj) * kDim + d0]);
          }
          mma884(tg[rb], ah.x, ah.y, b0, b1);
        }
      }
#pragma unroll
      for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
          const int cell = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
          part[warp][row][cell] = tg[rb][i];
        }
    }
    __syncthreads();

    // ---- Online softmax: unchanged, but the score sums the four 64-dim
    // partials from the tensor-core warps. ----
    if (is_compute_warp) {
      const int r = warp;
      const int n = lane;
      // The four 64-dim tensor-core partials for this (row, cell) sum in a
      // fixed order; invalid cells go to -inf so softmax masks them.
      const float sc =
          (n < tile_tokens && slots[n] >= 0)
              ? ((part[0][r][n] + part[1][r][n]) +
                 (part[2][r][n] + part[3][r][n]))
              : -1.0e30f;
      __syncwarp();

      float loc_max = sc;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (lane == 0) {
        alpha16[r] = alpha;
        lsum16[r] = l_row[r];
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((sc - m_new) * scale_log2);
      }
      probs16[r * kProbStride + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
      if (lane == 0) {
        lsum16[r] = l_row[r];
      }
    }
    __syncthreads();

    // ---- p·V on tensor cores: warp w owns dims [64w, 64w+64) ----
    if (warp < 4) {
      float tmp[2][2][8];
#pragma unroll
      for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            tmp[rb][nt][i] = 0.f;
          }
#pragma unroll
      for (int kstep = 0; kstep < kBlockN / 4; ++kstep) {  // 8 k-steps
        const int c0 = kstep * 4;
        uint2 ah[2];
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          // A row jj of this 8-row block: P[row][c0..c0+3]; rows 6..15 are
          // the zero-initialized padding of probs16.
          ah[rb] = *reinterpret_cast<const uint2*>(
              &probs16[(rb * 8 + jj) * kProbStride + c0]);
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int d = warp * 64 + qp * 16 + nt * 8 + jj;
          const __half2 h0 = __halves2half2(
              *reinterpret_cast<const __half*>(&vs[c0 * kKVStride + d]),
              *reinterpret_cast<const __half*>(&vs[(c0 + 1) * kKVStride + d]));
          const __half2 h1 = __halves2half2(
              *reinterpret_cast<const __half*>(&vs[(c0 + 2) * kKVStride + d]),
              *reinterpret_cast<const __half*>(&vs[(c0 + 3) * kKVStride + d]));
          const uint32_t bb0 = *reinterpret_cast<const uint32_t*>(&h0);
          const uint32_t bb1 = *reinterpret_cast<const uint32_t*>(&h1);
#pragma unroll
          for (int rb = 0; rb < 2; ++rb) {
            mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, bb0, bb1);
          }
        }
      }
      // Rescale by this tile's alpha per output row, then accumulate.
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        const float a0 = alpha16[rb * 8 + crow];
        const float a2 = alpha16[rb * 8 + crow + 2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            acc[rb][nt][i] =
                fmaf(acc[rb][nt][i], ((i >> 1) & 1) ? a2 : a0, tmp[rb][nt][i]);
          }
      }
    }
    __syncthreads();
  }

  // ---- Write-out: warp w owns dims [64w, 64w+64) of the six real rows ----
  if (warp < 4) {
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = rb * 8 + crow + 2 * h;
        if (row >= kGroup) {
          continue;
        }
        const float inv_l = lsum16[row] > 0.f ? 1.f / lsum16[row] : 0.f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int i = hh * 4 + h * 2;
            const int d =
                warp * 64 + qp * 16 + nt * 8 + ccol + 4 * hh;
            output[(int64_t)query * kGroup * kDim + row * kDim + d] =
                __float2half_rn(acc[rb][nt][i] * inv_l);
            output[(int64_t)query * kGroup * kDim + row * kDim + d + 1] =
                __float2half_rn(acc[rb][nt][i + 1] * inv_l);
          }
      }
  }
}

}  // namespace sm70_longctx

void sm70_longctx_decode(torch::Tensor q, torch::Tensor k_cache,
                         torch::Tensor v_cache, torch::Tensor page_table,
                         torch::Tensor seq_lens, int64_t max_splits,
                         int64_t min_tokens_per_split, double softmax_scale,
                         double k_scale, double v_scale,
                         torch::Tensor partial_o,
                         torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  const int heads = q.size(1);
  const int heads_kv = k_cache.size(2);
  const int dim = q.size(2);
  const int batch = q.size(0);
  TORCH_CHECK(heads_kv >= 1 && heads == heads_kv * kGroup && dim == kDim,
              "sm70_longctx expects H6/D256 GQA with a full KV-head split "
              "(TP1 Hkv4, TP2 Hkv2, TP4 Hkv1)");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8 &&
                  v_cache.scalar_type() == torch::kUInt8,
              "sm70_longctx expects E5M2 byte KV cache");
  TORCH_CHECK(q.dtype() == torch::kHalf, "sm70_longctx expects fp16 queries");
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)heads_kv, (unsigned)max_splits, (unsigned)batch);
  const int* page_ptr = page_table.numel() > 0 ? page_table.data_ptr<int>()
                                               : nullptr;
  const int max_blocks = page_table.numel() > 0 ? (int)page_table.size(1) : 0;
  decode_partial_kernel<<<grid, sm70_longctx::kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_ptr,
      seq_lens.data_ptr<int>(), (int)max_splits, (int)min_tokens_per_split,
      max_blocks, (float)(softmax_scale * k_scale), (float)v_scale,
      reinterpret_cast<__half*>(partial_o.data_ptr()),
      partial_lse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_decode(torch::Tensor q, torch::Tensor k_cache,
                     torch::Tensor v_cache, torch::Tensor req_to_token,
                     torch::Tensor req_indices, torch::Tensor indices,
                     torch::Tensor seq_lens, int64_t max_splits,
                     int64_t min_tokens_per_split, double softmax_scale,
                     c10::optional<torch::Tensor> lut,
                     torch::Tensor partial_o, torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kGroup && q.size(2) == kDim,
              "sm70_qsa_decode expects FP16 Q [batch,6,256]");
  const bool is_e5m2 = k_cache.scalar_type() == torch::kUInt8;
  const bool is_fp16 = k_cache.scalar_type() == torch::kHalf;
  TORCH_CHECK((is_e5m2 || is_fp16) &&
                  v_cache.scalar_type() == k_cache.scalar_type() &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_decode expects E5M2 byte or FP16 KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_decode expects int32 indices [batch,topk]");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt &&
                  req_to_token.dim() == 2 &&
                  req_indices.scalar_type() == torch::kInt &&
                  req_indices.numel() == q.size(0) &&
                  seq_lens.scalar_type() == torch::kInt &&
                  seq_lens.numel() == q.size(0),
              "sm70_qsa_decode expects one int32 request and length per row");
  TORCH_CHECK(max_splits > 0 && min_tokens_per_split > 0,
              "sm70_qsa_decode split parameters must be positive");
  TORCH_CHECK(partial_o.scalar_type() == torch::kHalf &&
                  partial_o.dim() == 4 && partial_o.size(0) == q.size(0) &&
                  partial_o.size(1) == max_splits &&
                  partial_o.size(2) == kGroup && partial_o.size(3) == kDim,
              "sm70_qsa_decode partial output has the wrong shape");
  TORCH_CHECK(partial_lse.scalar_type() == torch::kFloat &&
                  partial_lse.dim() == 3 &&
                  partial_lse.size(0) == q.size(0) &&
                  partial_lse.size(1) == max_splits &&
                  partial_lse.size(2) == kGroup,
              "sm70_qsa_decode partial LSE has the wrong shape");
  const __half* lut_ptr = nullptr;
  if (lut.has_value() && lut->defined()) {
    TORCH_CHECK(lut->scalar_type() == torch::kHalf && lut->numel() == 256,
                "sm70_qsa_decode lut must be 256 fp16 entries");
    lut_ptr = reinterpret_cast<const __half*>(lut->data_ptr());
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(1, (unsigned int)max_splits, (unsigned int)q.size(0));
  if (is_e5m2) {
    qsa_decode_partial_kernel<uint8_t><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
        (int)min_tokens_per_split, (float)softmax_scale, lut_ptr,
        reinterpret_cast<__half*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>());
  } else {
    qsa_decode_partial_kernel<__half><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k_cache.data_ptr()),
        reinterpret_cast<const __half*>(v_cache.data_ptr()),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
        (int)min_tokens_per_split, (float)softmax_scale, lut_ptr,
        reinterpret_cast<__half*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_indexer_decode(torch::Tensor q, torch::Tensor k_cache,
                             torch::Tensor page_table,
                             torch::Tensor context_lens,
                             int64_t max_model_len, double score_scale,
                             torch::Tensor logits) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kIndexerHeads && q.size(2) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 Q [batch,4,128]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kHalf && k_cache.dim() == 4 &&
                  k_cache.size(1) == kIndexerPageSize &&
                  k_cache.size(2) == 1 && k_cache.size(3) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 K [pages,4,1,128]");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt &&
                  page_table.dim() == 2 && page_table.size(0) == q.size(0) &&
                  context_lens.scalar_type() == torch::kInt &&
                  context_lens.numel() == q.size(0),
              "sm70_qsa_indexer_decode expects int32 page metadata");
  TORCH_CHECK(max_model_len > 0 && score_scale > 0,
              "sm70_qsa_indexer_decode dimensions and scale must be positive");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.dim() == 2 &&
                  logits.size(0) == q.size(0) &&
                  logits.size(1) == max_model_len,
              "sm70_qsa_indexer_decode logits have the wrong shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      (unsigned int)((max_model_len + kIndexerKeysPerBlock - 1) /
                     kIndexerKeysPerBlock),
      (unsigned int)q.size(0));
  qsa_indexer_decode_kernel<<<grid, kIndexerThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const __half*>(k_cache.data_ptr()),
      page_table.data_ptr<int>(), context_lens.data_ptr<int>(),
      (int)page_table.size(1), (int)max_model_len, (float)score_scale,
      logits.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_prefill(torch::Tensor q, torch::Tensor k_cache,
                      torch::Tensor v_cache, torch::Tensor req_to_token,
                      torch::Tensor req_indices, torch::Tensor indices,
                      torch::Tensor seq_lens, double softmax_scale,
                      c10::optional<torch::Tensor> lut,
                      torch::Tensor output) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kGroup && q.size(2) == kDim,
              "sm70_qsa_prefill expects FP16 Q [tokens,6,256]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8 &&
                  v_cache.scalar_type() == torch::kUInt8 &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_prefill expects E5M2 byte KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_prefill expects int32 indices [tokens,topk]");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt &&
                  req_to_token.dim() == 2 &&
                  req_indices.scalar_type() == torch::kInt &&
                  req_indices.numel() == 1 && seq_lens.scalar_type() == torch::kInt &&
                  seq_lens.numel() == 1,
              "sm70_qsa_prefill currently supports one int32 request row");
  TORCH_CHECK(output.scalar_type() == torch::kHalf &&
                  output.sizes() == q.sizes(),
              "sm70_qsa_prefill output must match Q");
  const __half* lut_ptr = nullptr;
  if (lut.has_value() && lut->defined()) {
    TORCH_CHECK(lut->scalar_type() == torch::kHalf && lut->numel() == 256,
                "sm70_qsa_prefill lut must be 256 fp16 entries");
    lut_ptr = reinterpret_cast<const __half*>(lut->data_ptr());
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  static const bool use_mma = [] {
    const char* env = std::getenv("SGLANG_SM70_QSA_MMA");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  if (use_mma) {
    qsa_prefill_kernel_mma<<<q.size(0), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1),
        (float)softmax_scale, lut_ptr,
        reinterpret_cast<__half*>(output.data_ptr()));
  } else {
    qsa_prefill_kernel<<<q.size(0), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1),
        (float)softmax_scale, lut_ptr,
        reinterpret_cast<__half*>(output.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm70_longctx_decode", &sm70_longctx_decode,
        "Long-context grouped decode partial (D256 G6 E5M2)");
  m.def("sm70_qsa_decode", &sm70_qsa_decode,
        "QSA grouped split-KV decode (D256 G6 E5M2)");
  m.def("sm70_qsa_indexer_decode", &sm70_qsa_indexer_decode,
        "QSA indexer decode scoring (H4 D128 FP16)");
  m.def("sm70_qsa_prefill", &sm70_qsa_prefill,
        "QSA chunk-prefill (D256 G6 E5M2)");
}
