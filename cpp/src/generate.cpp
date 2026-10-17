#include "qwen3/generate.hpp"

int argmax(float* values, int n) {
    if (n <= 0) throw invalid_argument("argmax: mang rong");
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (values[i] > values[best]) best = i;   // dau '>' : giu id nho hon khi bang nhau
    }
    return best;
}

static double seconds_since(chrono::steady_clock::time_point start) {
    return chrono::duration<double>(chrono::steady_clock::now() - start).count();
}

GenerationResult greedy_generate(Qwen3Model& model, vector<int> prompt, int max_new_tokens,
                                 vector<int> eos_ids, Tensor* prompt_logits) {
    if (prompt.empty()) throw invalid_argument("greedy_generate: prompt rong");
    if (max_new_tokens < 0) throw invalid_argument("greedy_generate: max_new_tokens phai >= 0");
    int limit = static_cast<int>(model.config.max_position_embeddings);
    if (static_cast<int>(prompt.size()) + max_new_tokens > limit) {
        throw invalid_argument("greedy_generate: prompt + max_new_tokens vuot max_position_embeddings");
    }
    GenerationResult result;
    if (max_new_tokens == 0) return result;
    KVCache cache = model.new_cache();
    int V = static_cast<int>(model.config.vocab_size);

    // Prefill
    auto t0 = chrono::steady_clock::now();
    vector<vector<int>> ids = {prompt};
    bool keep_all = prompt_logits != nullptr;
    Tensor logits = model.forward(ids, &cache, !keep_all);
    int rows = logits.size(1);
    int next = argmax(logits.data() + (rows - 1) * V, V);
    if (keep_all) *prompt_logits = logits;
    result.prefill_seconds = seconds_since(t0);
    result.tokens.push_back(next);

    // Decode tung token
    auto t1 = chrono::steady_clock::now();
    while (static_cast<int>(result.tokens.size()) < max_new_tokens) {
        if (find(eos_ids.begin(), eos_ids.end(), next) != eos_ids.end()) break;
        vector<vector<int>> step = {{next}};
        Tensor step_logits = model.forward(step, &cache, true);
        next = argmax(step_logits.data(), V);
        result.tokens.push_back(next);
        result.decode_steps++;
    }
    result.decode_seconds = seconds_since(t1);
    return result;
}
