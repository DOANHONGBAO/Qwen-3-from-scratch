// Unit tests cho ca model Qwen3 (cpp/include/qwen3/model.hpp), KV cache trong model,
// greedy generation va nap checkpoint safetensors.
// Gia tri chuan sinh boi scripts/export_cpp_model_reference.py: model Qwen3 TI HON,
// weight ngau nhien (khong phai checkpoint that), tinh bang Qwen3ForCausalLM PyTorch.
#include "qwen3/generate.hpp"
#include "qwen3/json.hpp"
#include "qwen3/model.hpp"
#include "qwen3/ops.hpp"
#include "qwen3/safetensors.hpp"
#include "model_reference.hpp"
#include "test_utils.hpp"

string temp_dir() {
    string dir = filesystem::temp_directory_path().string() + "/qwen3_cpp_test_model";
    filesystem::create_directories(dir);
    return dir;
}

void write_text(string path, string text) {
    ofstream f(path);
    f << text;
}

qwen3::Config tiny_config() {
    string path = temp_dir() + "/config.json";
    write_text(path, CONFIG_JSON);
    return config_from_json(path);
}

Qwen3Model make_model() {
    Qwen3Model model(tiny_config());
    for (size_t i = 0; i < WEIGHT_NAMES.size(); i++) {
        model.weight(WEIGHT_NAMES[i]) = Tensor(*WEIGHT_SHAPES[i], *WEIGHT_VALUES[i]);
    }
    return model;
}

vector<vector<int>> reference_ids() {
    int B = IDS_SHAPE[0], T = IDS_SHAPE[1];
    vector<vector<int>> ids(static_cast<size_t>(B));
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) ids[static_cast<size_t>(b)].push_back(static_cast<int>(IDS[static_cast<size_t>(b * T + t)]));
    return ids;
}

// Lay token [start, end) cua moi hang
vector<vector<int>> slice_ids(vector<vector<int>>& ids, int start, int end) {
    vector<vector<int>> out;
    for (vector<int>& row : ids) out.push_back(vector<int>(row.begin() + start, row.begin() + end));
    return out;
}

bool same_values(Tensor& a, Tensor& b) {
    if (a.get_shape() != b.get_shape()) return false;
    for (int i = 0; i < a.numel(); i++)
        if (a.data()[i] != b.data()[i]) return false;
    return true;
}

// ------------------------------------------------------------ cau truc

void test_config_and_names() {
    qwen3::Config c = tiny_config();
    check(c.vocab_size == 48 && c.hidden_size == 16 && c.head_dim == 8 && c.tie_word_embeddings,
          "config_from_json doc dung cac truong");
    Qwen3Model model(c);
    check(model.weight_names() == WEIGHT_NAMES, "ten weight + thu tu giong state_dict PyTorch");
    check_throws<invalid_argument>([&] { model.weight("lm_head.weight"); }, "tie: khong co lm_head rieng");
    check_throws<invalid_argument>([&] { model.weight("model.layers.9.mlp.up_proj.weight"); }, "layer khong ton tai");
}

void test_checkpoint_map_qwen3_0_6b() {
    // So voi danh sach tensor THAT cua Qwen3-0.6B-Base (fixture tu M2), khong can cap phat 2.4 GB
    Json fixture = Json::parse(read_text_file(string(QWEN3_FIXTURES_DIR) + "/checkpoint_name_map.json"));
    Json& tensors = fixture.get("inputs").get("tensors");
    vector<pair<string, vector<int>>> ours = checkpoint_tensors(qwen3::qwen3_0_6b_base());
    bool ok = tensors.items.size() == ours.size();
    for (Json& item : tensors.items) {
        string name = item.items[0].as_string();
        vector<int> shape;
        for (Json& d : item.items[1].items) shape.push_back(d.as_int());
        bool found = false;
        for (auto& entry : ours)
            if (entry.first == name && entry.second == shape) found = true;
        if (!found) ok = false;
    }
    check(ok, "checkpoint_tensors(Qwen3-0.6B) == danh sach tensor cua model.safetensors that");
    check(ours.size() == 310, "Qwen3-0.6B co 310 tensor = 1 + 28 * 11 + 1 (khong co lm_head vi tie)");
}

// ------------------------------------------------------------ so voi PyTorch

void test_hidden_states_per_layer() {
    // Tinh tay tung buoc de biet layer nao sai (neu co)
    Qwen3Model model = make_model();
    vector<vector<int>> ids = reference_ids();
    int B = 2, T = 7, H = 16;
    Tensor h({B, T, H});
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < H; i++) h.at(b, t, i) = model.embed_tokens.at(ids[static_cast<size_t>(b)][static_cast<size_t>(t)], i);
    check_close(h, HIDDEN_0_SHAPE, HIDDEN_0, "embedding [2,7,16]");
    Tensor h1 = model.layers[0].forward(h, nullptr, 0);
    // Residual stream lon (|h| ~ 250 voi weight ngau nhien std 1): float32 o 250 co buoc
    // ~3e-5, nen dung atol 1e-3 (~4e-6 tuong doi so voi max |h|).
    check_close(h1, HIDDEN_1_SHAPE, HIDDEN_1, "sau decoder layer 0", 1e-3f, 1e-5f);
    Tensor h2 = model.layers[1].forward(h1, nullptr, 1);
    check_close(h2, HIDDEN_2_SHAPE, HIDDEN_2, "sau decoder layer 1", 1e-3f, 1e-5f);
}

void test_logits_against_pytorch() {
    Qwen3Model model = make_model();
    vector<vector<int>> ids = reference_ids();
    Tensor logits = model.forward(ids, nullptr, false);
    check_close(logits, LOGITS_SHAPE, LOGITS, "logits [2,7,48]", 1e-4f, 1e-5f);

    Tensor last = model.forward(ids, nullptr, true);
    bool ok = last.get_shape() == vector<int>{2, 1, 48};
    for (int b = 0; b < 2 && ok; b++)
        for (int v = 0; v < 48; v++)
            if (last.at(b, 0, v) != logits.at(b, 6, v)) ok = false;
    check(ok, "last_only = logits cua vi tri cuoi");
}

void test_cached_logits() {
    // prefill 4 token, roi 1 + 1 + 1: logits phai bang forward ca chuoi
    Qwen3Model model = make_model();
    vector<vector<int>> ids = reference_ids();
    KVCache cache = model.new_cache();
    Tensor all({2, 7, 48});
    int pos = 0;
    for (int len : {4, 1, 1, 1}) {
        vector<vector<int>> chunk = slice_ids(ids, pos, pos + len);
        Tensor l = model.forward(chunk, &cache, false);
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < len; t++)
                for (int v = 0; v < 48; v++) all.at(b, pos + t, v) = l.at(b, t, v);
        pos += len;
    }
    check(cache.seq_len == 7, "cache.seq_len = 7 sau 4 lan forward");
    check_close(all, LOGITS_SHAPE, LOGITS, "logits co cache (4+1+1+1)", 1e-4f, 1e-5f);
}

void test_greedy_generation() {
    Qwen3Model model = make_model();
    vector<vector<int>> ids = reference_ids();
    Tensor prompt_logits;
    GenerationResult r = greedy_generate(model, ids[0], 10, {}, &prompt_logits);
    vector<int> expected;
    for (float g : GREEDY) expected.push_back(static_cast<int>(g));
    check(r.tokens == expected, "greedy 10 token giong het PyTorch generate()");
    check(r.decode_steps == 9, "1 token tu prefill + 9 buoc decode");
    bool ok = prompt_logits.get_shape() == vector<int>{1, 7, 48};
    for (int i = 0; i < 7 * 48 && ok; i++)
        if (fabs(prompt_logits.data()[i] - LOGITS[static_cast<size_t>(i)]) > 1e-4f) ok = false;
    check(ok, "prompt_logits = logits hang 0");

    // EOS: dung ngay sau token EOS dau tien (token EOS van duoc giu)
    GenerationResult stop = greedy_generate(model, ids[0], 10, {expected[2]}, nullptr);
    int cut = static_cast<int>(find(expected.begin(), expected.end(), expected[2]) - expected.begin()) + 1;
    check(stop.tokens == vector<int>(expected.begin(), expected.begin() + cut), "dung khi gap EOS");
    check(argmax(vector<float>{1, 3, 3, 2}.data(), 4) == 1, "argmax bang nhau -> id nho nhat");
    check_throws<invalid_argument>([&] { greedy_generate(model, ids[0], 60, {}, nullptr); },
                                   "prompt + max_new_tokens > max_position_embeddings");
}

void test_untied_head() {
    // tie_word_embeddings = false voi lm_head = ban sao embed_tokens => logits khong doi
    Qwen3Model tied = make_model();
    qwen3::Config c = tiny_config();
    c.tie_word_embeddings = false;
    Qwen3Model untied(c);
    for (size_t i = 0; i < WEIGHT_NAMES.size(); i++) {
        untied.weight(WEIGHT_NAMES[i]) = Tensor(*WEIGHT_SHAPES[i], *WEIGHT_VALUES[i]);
    }
    untied.weight("lm_head.weight") = tied.embed_tokens;
    vector<vector<int>> ids = reference_ids();
    Tensor a = tied.forward(ids, nullptr, false);
    Tensor b = untied.forward(ids, nullptr, false);
    check(same_values(a, b), "model khong tie dung lm_head rieng");
}

// ------------------------------------------------------------ nap checkpoint

void save_checkpoint(string dir, Qwen3Model& model, vector<string> names) {
    filesystem::create_directories(dir);
    write_text(dir + "/config.json", CONFIG_JSON);
    vector<Tensor*> tensors;
    for (string& n : names) tensors.push_back(&model.weight(n));
    save_safetensors(dir + "/model.safetensors", names, tensors);
}

void test_load_pretrained() {
    Qwen3Model model = make_model();
    string dir = temp_dir() + "/ckpt_ok";
    save_checkpoint(dir, model, model.weight_names());
    Qwen3Model loaded = load_pretrained(dir);
    vector<vector<int>> ids = reference_ids();
    Tensor a = model.forward(ids, nullptr, false);
    Tensor b = loaded.forward(ids, nullptr, false);
    check(same_values(a, b), "load_pretrained (safetensors F32) cho logits giong het");

    vector<string> missing = model.weight_names();
    missing.erase(missing.begin() + 3);
    string dir2 = temp_dir() + "/ckpt_missing";
    save_checkpoint(dir2, model, missing);
    check_throws<runtime_error>([&] { load_pretrained(dir2); }, "thieu 1 tensor -> loi");

    string dir3 = temp_dir() + "/ckpt_wrong_shape";
    filesystem::create_directories(dir3);
    write_text(dir3 + "/config.json", CONFIG_JSON);
    vector<string> names = model.weight_names();
    vector<Tensor*> tensors;
    for (string& n : names) tensors.push_back(&model.weight(n));
    Tensor wrong({16, 16});
    tensors[2] = &wrong;                                   // q_proj phai la [32, 16]
    save_safetensors(dir3 + "/model.safetensors", names, tensors);
    check_throws<runtime_error>([&] { load_pretrained(dir3); }, "sai shape -> loi");

    string dir4 = temp_dir() + "/ckpt_extra";
    filesystem::create_directories(dir4);
    write_text(dir4 + "/config.json", CONFIG_JSON);
    vector<string> extra = model.weight_names();
    vector<Tensor*> extra_t;
    for (string& n : extra) extra_t.push_back(&model.weight(n));
    Tensor junk({3});
    extra.push_back("model.layers.0.self_attn.q_proj.bias");
    extra_t.push_back(&junk);
    save_safetensors(dir4 + "/model.safetensors", extra, extra_t);
    check_throws<runtime_error>([&] { load_pretrained(dir4); }, "tensor la (bias) -> loi");

    check_throws<runtime_error>([&] { load_pretrained(temp_dir() + "/khong_ton_tai"); }, "thu muc khong co checkpoint");
}

int main() {
    test_config_and_names();
    test_checkpoint_map_qwen3_0_6b();
    test_hidden_states_per_layer();
    test_logits_against_pytorch();
    test_cached_logits();
    test_greedy_generation();
    test_untied_head();
    test_load_pretrained();
    return report("test_model");
}
