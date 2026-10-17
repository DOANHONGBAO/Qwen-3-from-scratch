#pragma once

#include "qwen3/config.hpp"
#include "qwen3/decoder.hpp"
#include "qwen3/kv_cache.hpp"
#include "qwen3/rmsnorm.hpp"

// Qwen3ForCausalLM: embedding -> N decoder layer -> RMSNorm cuoi -> LM head.
// Khi tie_word_embeddings = true (Qwen3-0.6B), LM head CHINH LA embed_tokens
// (khong co tensor lm_head rieng, tiet kiem V*H so float).
class Qwen3Model {
public:
    Qwen3Model(qwen3::Config config);

    // input_ids: B hang, moi hang T token (cung do dai).
    // Tra ve logits [B, T, V]; last_only = true thi chi tinh vi tri cuoi: [B, 1, V].
    // cache != nullptr: input_ids chi chua token MOI, vi tri tiep noi cache->seq_len.
    Tensor forward(vector<vector<int>>& input_ids, KVCache* cache, bool last_only);

    // Tham chieu toi weight theo TEN trong checkpoint (vd "model.layers.3.mlp.up_proj.weight")
    Tensor& weight(string name);
    vector<string> weight_names();          // dung thu tu cua checkpoint Hugging Face
    KVCache new_cache();

    qwen3::Config config;
    Tensor embed_tokens;                    // [V, H]
    vector<DecoderLayer> layers;
    RMSNorm norm;                           // [H]
    Tensor lm_head;                         // [V, H], chi dung khi KHONG tie

private:
    vector<pair<string, Tensor*>> weight_table();
};

// Danh sach ten + shape cua moi tensor trong checkpoint, tinh tu config
// (khong cap phat bo nho, dung duoc cho ca Qwen3-0.6B).
vector<pair<string, vector<int>>> checkpoint_tensors(qwen3::Config config);

// Doc config.json cua Hugging Face.
qwen3::Config config_from_json(string path);

// Nap model tu thu muc checkpoint (config.json + model.safetensors hoac
// model.safetensors.index.json + cac shard). Moi weight phai co, dung shape, nap dung 1 lan.
Qwen3Model load_pretrained(string model_dir);
