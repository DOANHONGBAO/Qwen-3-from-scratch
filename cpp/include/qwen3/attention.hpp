#pragma once

#include "qwen3/kv_cache.hpp"
#include "qwen3/rmsnorm.hpp"
#include "qwen3/rope.hpp"
#include "qwen3/tensor.hpp"

// Self-attention cua Qwen3: GQA + QK-Norm + RoPE + causal mask. Khong co bias.
//   x [B, T, H]
//   q = q_proj(x) -> [B, T, Nq, D] -> q_norm -> [B, Nq, T, D] -> RoPE
//   k = k_proj(x) -> [B, T, Nkv, D] -> k_norm -> [B, Nkv, T, D] -> RoPE
//   v = v_proj(x) -> [B, Nkv, T, D]                 (khong norm, khong RoPE)
//   query head h dung kv head  h / (Nq / Nkv)       (GQA, nhom lien tiep)
//   score(i, j) = q_i . k_j / sqrt(D), chi voi j <= i (causal)
//   out = o_proj( softmax(score) @ v )             [B, T, H]
// Co cache: k, v moi duoc noi vao cache, query nhin toan bo token da luu.
class Attention {
public:
    Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim,
              float rope_theta, float rms_norm_eps);

    // cache = nullptr: xu ly ca chuoi tu vi tri 0, khong luu gi.
    // cache != nullptr: token moi bat dau o vi tri cache->seq_len, k/v duoc luu vao layer.
    Tensor forward(Tensor& x, KVCache* cache, int layer);

    int hidden_size;
    int num_heads;        // Nq
    int num_kv_heads;     // Nkv
    int head_dim;         // D (khong nhat thiet = H / Nq)
    Tensor q_proj;        // [Nq*D, H]
    Tensor k_proj;        // [Nkv*D, H]
    Tensor v_proj;        // [Nkv*D, H]
    Tensor o_proj;        // [H, Nq*D]
    RMSNorm q_norm;       // weight [D]
    RMSNorm k_norm;       // weight [D]
    RoPE rope;
};
