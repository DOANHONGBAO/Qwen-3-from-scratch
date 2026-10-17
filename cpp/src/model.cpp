#include "qwen3/model.hpp"

#include "qwen3/json.hpp"
#include "qwen3/ops.hpp"
#include "qwen3/safetensors.hpp"

static int as_int(int64_t v) { return static_cast<int>(v); }

Qwen3Model::Qwen3Model(qwen3::Config c)
    : config(c), norm(as_int(c.hidden_size), static_cast<float>(c.rms_norm_eps)) {
    config.validate();
    if (c.vocab_size * c.hidden_size > 2147483647LL) {
        throw invalid_argument("Qwen3Model: vocab_size * hidden_size vuot qua gioi han int");
    }
    embed_tokens = Tensor({as_int(c.vocab_size), as_int(c.hidden_size)});
    for (int i = 0; i < as_int(c.num_hidden_layers); i++) {
        layers.push_back(DecoderLayer(c));
    }
    if (!c.tie_word_embeddings) {
        lm_head = Tensor({as_int(c.vocab_size), as_int(c.hidden_size)});
    }
}

vector<pair<string, vector<int>>> checkpoint_tensors(qwen3::Config c) {
    int V = as_int(c.vocab_size), H = as_int(c.hidden_size), I = as_int(c.intermediate_size);
    int qd = as_int(c.num_attention_heads * c.head_dim);
    int kvd = as_int(c.num_key_value_heads * c.head_dim);
    int D = as_int(c.head_dim);
    vector<pair<string, vector<int>>> out;
    out.push_back({"model.embed_tokens.weight", {V, H}});
    for (int i = 0; i < as_int(c.num_hidden_layers); i++) {
        string p = "model.layers." + to_string(i) + ".";
        out.push_back({p + "input_layernorm.weight", {H}});
        out.push_back({p + "self_attn.q_proj.weight", {qd, H}});
        out.push_back({p + "self_attn.k_proj.weight", {kvd, H}});
        out.push_back({p + "self_attn.v_proj.weight", {kvd, H}});
        out.push_back({p + "self_attn.o_proj.weight", {H, qd}});
        out.push_back({p + "self_attn.q_norm.weight", {D}});
        out.push_back({p + "self_attn.k_norm.weight", {D}});
        out.push_back({p + "post_attention_layernorm.weight", {H}});
        out.push_back({p + "mlp.gate_proj.weight", {I, H}});
        out.push_back({p + "mlp.up_proj.weight", {I, H}});
        out.push_back({p + "mlp.down_proj.weight", {H, I}});
    }
    out.push_back({"model.norm.weight", {H}});
    if (!c.tie_word_embeddings) out.push_back({"lm_head.weight", {V, H}});
    return out;
}

vector<pair<string, Tensor*>> Qwen3Model::weight_table() {
    vector<pair<string, Tensor*>> t;
    t.push_back({"model.embed_tokens.weight", &embed_tokens});
    for (size_t i = 0; i < layers.size(); i++) {
        string p = "model.layers." + to_string(i) + ".";
        DecoderLayer& L = layers[i];
        t.push_back({p + "input_layernorm.weight", &L.input_layernorm.weight});
        t.push_back({p + "self_attn.q_proj.weight", &L.self_attn.q_proj});
        t.push_back({p + "self_attn.k_proj.weight", &L.self_attn.k_proj});
        t.push_back({p + "self_attn.v_proj.weight", &L.self_attn.v_proj});
        t.push_back({p + "self_attn.o_proj.weight", &L.self_attn.o_proj});
        t.push_back({p + "self_attn.q_norm.weight", &L.self_attn.q_norm.weight});
        t.push_back({p + "self_attn.k_norm.weight", &L.self_attn.k_norm.weight});
        t.push_back({p + "post_attention_layernorm.weight", &L.post_attention_layernorm.weight});
        t.push_back({p + "mlp.gate_proj.weight", &L.mlp.gate_proj});
        t.push_back({p + "mlp.up_proj.weight", &L.mlp.up_proj});
        t.push_back({p + "mlp.down_proj.weight", &L.mlp.down_proj});
    }
    t.push_back({"model.norm.weight", &norm.weight});
    if (!config.tie_word_embeddings) t.push_back({"lm_head.weight", &lm_head});
    return t;
}

Tensor& Qwen3Model::weight(string name) {
    for (auto& entry : weight_table()) {
        if (entry.first == name) return *entry.second;
    }
    throw invalid_argument("Qwen3Model: khong co weight ten '" + name + "'");
}

vector<string> Qwen3Model::weight_names() {
    vector<string> names;
    for (auto& entry : weight_table()) names.push_back(entry.first);
    return names;
}

KVCache Qwen3Model::new_cache() {
    return KVCache(as_int(config.num_hidden_layers), as_int(config.max_position_embeddings));
}

Tensor Qwen3Model::forward(vector<vector<int>>& input_ids, KVCache* cache, bool last_only) {
    int B = static_cast<int>(input_ids.size());
    if (B == 0 || input_ids[0].empty()) throw invalid_argument("Qwen3Model: input_ids rong");
    int T = static_cast<int>(input_ids[0].size());
    int V = as_int(config.vocab_size), H = as_int(config.hidden_size);
    for (vector<int>& row : input_ids) {
        if (static_cast<int>(row.size()) != T) {
            throw invalid_argument("Qwen3Model: moi hang input_ids phai cung do dai");
        }
        for (int id : row) {
            if (id < 0 || id >= V) throw invalid_argument("Qwen3Model: token id ngoai [0, vocab_size)");
        }
    }
    int start_pos = cache ? cache->seq_len : 0;
    if (start_pos + T > as_int(config.max_position_embeddings)) {
        throw invalid_argument("Qwen3Model: vuot qua max_position_embeddings");
    }

    // 1. Embedding: chep hang `id` cua embed_tokens -> h [B, T, H]
    Tensor h({B, T, H});
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* src = embed_tokens.data() + input_ids[b][t] * H;
            copy(src, src + H, h.data() + (b * T + t) * H);
        }
    }

    // 2. Cac decoder layer. Neu loi giua chung, bo cac append chua commit.
    try {
        for (size_t i = 0; i < layers.size(); i++) {
            h = layers[i].forward(h, cache, static_cast<int>(i));
        }
        if (cache) cache->commit(T);
    } catch (...) {
        if (cache) cache->rollback();
        throw;
    }

    // 3. Norm cuoi (+ chi giu vi tri cuoi neu last_only), roi LM head
    Tensor hn = norm.forward(h);
    if (last_only && T > 1) {
        Tensor last({B, 1, H});
        for (int b = 0; b < B; b++) {
            float* src = hn.data() + (b * T + (T - 1)) * H;
            copy(src, src + H, last.data() + b * H);
        }
        hn = last;
    }
    Tensor& head = config.tie_word_embeddings ? embed_tokens : lm_head;
    return linear(hn, head);
}

// ------------------------------------------------------------ nap checkpoint

qwen3::Config config_from_json(string path) {
    Json j = Json::parse(read_text_file(path));
    qwen3::Config c{};
    c.vocab_size = j.get("vocab_size").as_int();
    c.hidden_size = j.get("hidden_size").as_int();
    c.intermediate_size = j.get("intermediate_size").as_int();
    c.num_hidden_layers = j.get("num_hidden_layers").as_int();
    c.num_attention_heads = j.get("num_attention_heads").as_int();
    c.num_key_value_heads = j.get("num_key_value_heads").as_int();
    c.head_dim = j.has("head_dim") ? j.get("head_dim").as_int()
                                   : c.hidden_size / c.num_attention_heads;
    c.max_position_embeddings = j.get("max_position_embeddings").as_int();
    // transformers 5 co the ghi rope_theta trong "rope_parameters"
    if (j.has("rope_theta")) {
        c.rope_theta = j.get("rope_theta").as_number();
    } else {
        c.rope_theta = j.get("rope_parameters").get("rope_theta").as_number();
    }
    c.rms_norm_eps = j.get("rms_norm_eps").as_number();
    c.tie_word_embeddings = j.has("tie_word_embeddings") ? j.get("tie_word_embeddings").as_bool() : false;
    c.attention_bias = j.has("attention_bias") ? j.get("attention_bias").as_bool() : false;
    c.qk_norm = true;
    if (c.attention_bias) throw invalid_argument("config: attention_bias = true chua duoc ho tro");
    if (j.has("rope_scaling") && j.get("rope_scaling").type != "null") {
        throw invalid_argument("config: rope_scaling chua duoc ho tro");
    }
    c.validate();
    return c;
}

static bool file_exists(string path) {
    ifstream f(path, ios::binary);
    return static_cast<bool>(f);
}

Qwen3Model load_pretrained(string model_dir) {
    qwen3::Config c = config_from_json(model_dir + "/config.json");

    // Danh sach file safetensors: 1 file, hoac nhieu shard liet ke trong index
    vector<string> files;
    if (file_exists(model_dir + "/model.safetensors")) {
        files.push_back(model_dir + "/model.safetensors");
    } else if (file_exists(model_dir + "/model.safetensors.index.json")) {
        Json index = Json::parse(read_text_file(model_dir + "/model.safetensors.index.json"));
        Json& map = index.get("weight_map");
        for (Json& f : map.values) {
            string p = model_dir + "/" + f.as_string();
            if (find(files.begin(), files.end(), p) == files.end()) files.push_back(p);
        }
    } else {
        throw runtime_error("load_pretrained: khong tim thay model.safetensors trong " + model_dir);
    }

    Qwen3Model model(c);
    vector<string> expected = model.weight_names();
    vector<int> loaded(expected.size(), 0);
    vector<string> unexpected;

    for (string& path : files) {
        SafetensorsFile st(path);
        for (SafetensorsEntry& e : st.entries) {
            auto it = find(expected.begin(), expected.end(), e.name);
            if (it == expected.end()) {
                // lm_head.weight du thua khi da tie: bo qua (LM head = embed_tokens)
                if (!(e.name == "lm_head.weight" && c.tie_word_embeddings)) unexpected.push_back(e.name);
                continue;
            }
            size_t idx = static_cast<size_t>(it - expected.begin());
            Tensor& dst = model.weight(e.name);
            if (dst.get_shape() != e.shape) {
                throw runtime_error("load_pretrained: sai shape o " + e.name);
            }
            if (loaded[idx] > 0) throw runtime_error("load_pretrained: tensor bi lap lai: " + e.name);
            st.read_into(e.name, dst.data(), dst.numel());
            loaded[idx]++;
        }
    }

    string missing;
    for (size_t i = 0; i < expected.size(); i++) {
        if (loaded[i] == 0) missing += " " + expected[i];
    }
    if (!missing.empty()) throw runtime_error("load_pretrained: thieu tensor:" + missing);
    if (!unexpected.empty()) {
        string list;
        for (string& n : unexpected) list += " " + n;
        throw runtime_error("load_pretrained: tensor la khong thuoc model:" + list);
    }
    return model;
}
