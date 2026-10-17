#pragma once

#include "qwen3/tensor.hpp"

// KV cache cho sinh token tu hoi quy (autoregressive decoding).
// Moi layer luu key (DA qua QK-Norm + RoPE) va value cua moi token da xu ly,
// de buoc sau khong phai tinh lai ca prompt.
//
// Layout luu tru: "time-major"  [S, B, Nkv, D]  (S = so token da luu).
// Token moi chi can noi vao CUOI vector (push_back, O(1) trung binh), khong phai
// chep lai ca cache nhu khi luu [B, Nkv, S, D].
//
// Quy tac: moi lan model forward, MOI layer append dung 1 lan, sau do model goi
// commit(T). seq_len = so token da commit = vi tri cua token tiep theo.
class KVCache {
public:
    KVCache(int num_layers, int max_length);

    // k, v: [B, Nkv, T, D] cua T token moi
    void append(int layer, Tensor& k, Tensor& v);
    int layer_length(int layer);           // so token dang luu o layer nay
    void commit(int num_new);              // tat ca layer da append num_new token
    void rollback();                       // bo cac append chua commit (forward bi loi)
    void reset();                          // xoa het, dung cho phien moi
    long long nbytes();

    // Con tro toi key / value cua token j, batch b, kv head h (D so float lien tiep)
    float* key(int layer, int j, int b, int h);
    float* value(int layer, int j, int b, int h);

    int num_layers;
    int max_length;
    int seq_len = 0;
    int batch = 0;            // co dinh o lan append dau tien
    int num_kv_heads = 0;
    int head_dim = 0;
    vector<vector<float>> keys;    // keys[layer]: [S, B, Nkv, D]
    vector<vector<float>> values;
};
