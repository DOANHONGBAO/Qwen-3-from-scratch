#include "qwen3/kv_cache.hpp"

KVCache::KVCache(int num_layers, int max_length) {
    if (num_layers <= 0 || max_length <= 0) {
        throw invalid_argument("KVCache: num_layers va max_length phai > 0");
    }
    this->num_layers = num_layers;
    this->max_length = max_length;
    reset();
}

void KVCache::reset() {
    keys.assign(num_layers, vector<float>());
    values.assign(num_layers, vector<float>());
    seq_len = 0;
    batch = 0;
    num_kv_heads = 0;
    head_dim = 0;
}

static void check_layer(int layer, int num_layers) {
    if (layer < 0 || layer >= num_layers) {
        throw out_of_range("KVCache: layer ngoai pham vi");
    }
}

int KVCache::layer_length(int layer) {
    check_layer(layer, num_layers);
    int per_token = batch * num_kv_heads * head_dim;
    if (per_token == 0) return 0;
    return static_cast<int>(keys[layer].size()) / per_token;
}

void KVCache::append(int layer, Tensor& k, Tensor& v) {
    check_layer(layer, num_layers);
    if (k.dim() != 4 || k.get_shape() != v.get_shape()) {
        throw invalid_argument("KVCache: k va v phai cung shape [B, Nkv, T, D]");
    }
    int B = k.size(0), N = k.size(1), T = k.size(2), D = k.size(3);
    if (T == 0) throw invalid_argument("KVCache: khong append 0 token");
    if (batch == 0) {   // lan dau: co dinh layout cho ca phien
        batch = B;
        num_kv_heads = N;
        head_dim = D;
    } else if (B != batch || N != num_kv_heads || D != head_dim) {
        throw invalid_argument("KVCache: B / Nkv / D khac voi du lieu da luu (phien moi thi goi reset())");
    }
    if (layer_length(layer) != seq_len) {
        throw runtime_error("KVCache: layer nay da append trong forward hien tai (moi layer chi 1 lan)");
    }
    if (seq_len + T > max_length) {
        throw invalid_argument("KVCache: vuot qua max_length");
    }

    // [B, N, T, D]  ->  noi vao cuoi theo thu tu [t][b][n][d]
    float* pk = k.data();
    float* pv = v.data();
    vector<float>& K = keys[layer];
    vector<float>& V = values[layer];
    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                int base = ((b * N + n) * T + t) * D;
                K.insert(K.end(), pk + base, pk + base + D);
                V.insert(V.end(), pv + base, pv + base + D);
            }
        }
    }
}

void KVCache::commit(int num_new) {
    int target = seq_len + num_new;
    for (int l = 0; l < num_layers; l++) {
        if (layer_length(l) != target) {
            throw runtime_error("KVCache: commit khi chua du moi layer append");
        }
    }
    seq_len = target;
}

void KVCache::rollback() {
    size_t keep = static_cast<size_t>(seq_len) * static_cast<size_t>(batch * num_kv_heads * head_dim);
    for (int l = 0; l < num_layers; l++) {
        if (keys[l].size() > keep) {
            keys[l].resize(keep);
            values[l].resize(keep);
        }
    }
}

long long KVCache::nbytes() {
    long long total = 0;
    for (int l = 0; l < num_layers; l++) {
        total += static_cast<long long>(keys[l].size() + values[l].size()) * 4;
    }
    return total;
}

float* KVCache::key(int layer, int j, int b, int h) {
    return keys[layer].data() + ((j * batch + b) * num_kv_heads + h) * head_dim;
}

float* KVCache::value(int layer, int j, int b, int h) {
    return values[layer].data() + ((j * batch + b) * num_kv_heads + h) * head_dim;
}
