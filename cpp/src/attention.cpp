#include "qwen3/attention.hpp"

#include "qwen3/ops.hpp"
#include "qwen3/profiler.hpp"

// RMSNorm va RoPE khong co constructor mac dinh, nen phai khoi tao chung
// trong "member initializer list" (phan sau dau ':') truoc khi vao than ham.
Attention::Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim,
                     float rope_theta, float rms_norm_eps)
    : q_norm(head_dim, rms_norm_eps), k_norm(head_dim, rms_norm_eps), rope(head_dim, rope_theta) {
    if (hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
        throw invalid_argument("Attention: kich thuoc phai > 0");
    }
    if (num_heads % num_kv_heads != 0) {
        throw invalid_argument("Attention: num_heads phai chia het cho num_kv_heads");
    }
    this->hidden_size = hidden_size;
    this->num_heads = num_heads;
    this->num_kv_heads = num_kv_heads;
    this->head_dim = head_dim;
    q_proj = Tensor({num_heads * head_dim, hidden_size});
    k_proj = Tensor({num_kv_heads * head_dim, hidden_size});
    v_proj = Tensor({num_kv_heads * head_dim, hidden_size});
    o_proj = Tensor({hidden_size, num_heads * head_dim});
}

Tensor Attention::forward(Tensor& x, KVCache* cache, int layer) {
    if (x.dim() != 3 || x.size(2) != hidden_size) {
        throw invalid_argument("Attention: x phai co shape [B, T, hidden_size]");
    }
    vector<int> q_shape = {num_heads * head_dim, hidden_size};
    vector<int> kv_shape = {num_kv_heads * head_dim, hidden_size};
    if (q_proj.get_shape() != q_shape || k_proj.get_shape() != kv_shape ||
        v_proj.get_shape() != kv_shape || o_proj.get_shape() != vector<int>{hidden_size, num_heads * head_dim}) {
        throw invalid_argument("Attention: weight sai shape");
    }
    int B = x.size(0), T = x.size(1);
    if (B == 0 || T == 0) throw invalid_argument("Attention: can it nhat 1 batch va 1 token");
    int Nq = num_heads, Nkv = num_kv_heads, D = head_dim;
    int start_pos = cache ? cache->seq_len : 0;

    // 1. Chieu q, k, v va tach head: [B, T, N*D] -> [B, T, N, D]
    Tensor q = linear(x, q_proj);
    Tensor k = linear(x, k_proj);
    Tensor v = linear(x, v_proj);
    Tensor q4 = q.reshape({B, T, Nq, D});
    Tensor k4 = k.reshape({B, T, Nkv, D});
    Tensor v4 = v.reshape({B, T, Nkv, D});

    // 2. QK-Norm theo tung head (RMSNorm tren chieu cuoi D), roi doi thanh [B, N, T, D]
    Tensor qn = q_norm.forward(q4);
    Tensor kn = k_norm.forward(k4);
    Tensor qt = qn.transpose(1, 2);
    Tensor kt = kn.transpose(1, 2);
    Tensor vt = v4.transpose(1, 2);

    // 3. RoPE theo vi tri tuyet doi start_pos .. start_pos + T - 1
    Tensor qr = rope.forward(qt, start_pos);
    Tensor kr = rope.forward(kt, start_pos);

    // 4. Luu k, v: vao cache that, hoac vao 1 cache tam neu khong dung cache
    KVCache local(1, T);
    KVCache* store = cache ? cache : &local;
    int store_layer = cache ? layer : 0;
    store->append(store_layer, kr, vt);

    // 5. Attention tung (batch, query head, token)
    Tensor ctx({B, T, Nq * D});                // ghi thang layout [B, T, Nq*D] cho o_proj
    {
        ScopedTimer core_timer("attention.core");   // chi score/softmax/tong, khong tinh proj
        int group = Nq / Nkv;
        float scaling = static_cast<float>(1.0 / sqrt(static_cast<double>(D)));
        float* pq = qr.data();
        float* pc = ctx.data();
        vector<float> probs(static_cast<size_t>(start_pos + T));

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < Nq; h++) {
                int kvh = h / group;                       // GQA: [0,0,1,1,...] khi group = 2
                for (int t = 0; t < T; t++) {
                    int qpos = start_pos + t;              // vi tri tuyet doi cua query
                    float* qv = pq + ((b * Nq + h) * T + t) * D;

                    // score voi key j = 0..qpos (j > qpos bi mask = xac suat 0, bo qua luon)
                    float mx = -INFINITY;
                    for (int j = 0; j <= qpos; j++) {
                        float* kv = store->key(store_layer, j, b, kvh);
                        probs[static_cast<size_t>(j)] = dot_product(qv, kv, D) * scaling;
                        mx = max(mx, probs[static_cast<size_t>(j)]);
                    }
                    // softmax on dinh: tru max truoc khi exp
                    float sum = 0.0f;
                    for (int j = 0; j <= qpos; j++) {
                        float e = exp(probs[static_cast<size_t>(j)] - mx);
                        probs[static_cast<size_t>(j)] = e;
                        sum += e;
                    }
                    // ctx = sum_j p_j * v_j
                    float* out = pc + (b * T + t) * Nq * D + h * D;
                    for (int j = 0; j <= qpos; j++) {
                        float p = probs[static_cast<size_t>(j)] / sum;
                        float* vv = store->value(store_layer, j, b, kvh);
                        for (int d = 0; d < D; d++) out[d] += p * vv[d];
                    }
                }
            }
        }
    }

    // 6. Gop head va chieu ra: [B, T, Nq*D] -> [B, T, H]
    return linear(ctx, o_proj);
}
