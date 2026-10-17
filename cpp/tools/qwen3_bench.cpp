// qwen3_bench: do toc do prefill / decode cua model C++ tren checkpoint that.
//
//   qwen3_bench --model-dir DIR [--prompt-len 16] [--new-tokens 16] [--threads 4]
//               [--repeats 2]
//
// Prompt la token id tong hop (khong can tokenizer), greedy, KHONG dung o EOS.
// Moi lan lap: 1 prefill (P token) + (N - 1) buoc decode. Bao cao trung vi (median)
// cua cac lan lap, sau 1 lan khoi dong (warm-up). In 1 dong JSON tren stdout.
#include "qwen3/generate.hpp"
#include "qwen3/model.hpp"
#include "qwen3/ops.hpp"
#include "qwen3/profiler.hpp"

double median(vector<double> v) {
    sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double peak_rss_gib() {
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return stod(line.substr(6)) / (1024.0 * 1024.0);
    }
    return -1.0;
}

int main(int argc, char** argv) {
    string model_dir;
    int prompt_len = 16, new_tokens = 16, threads = 0, repeats = 2;
    for (int i = 1; i + 1 < argc; i += 2) {
        string a = argv[i], v = argv[i + 1];
        if (a == "--model-dir") model_dir = v;
        else if (a == "--prompt-len") prompt_len = stoi(v);
        else if (a == "--new-tokens") new_tokens = stoi(v);
        else if (a == "--threads") threads = stoi(v);
        else if (a == "--repeats") repeats = stoi(v);
        else {
            cerr << "unknown argument " << a << "\n";
            return 2;
        }
    }
    if (model_dir.empty() || prompt_len < 1 || new_tokens < 2 || repeats < 1) {
        cerr << "usage: qwen3_bench --model-dir DIR [--prompt-len P] [--new-tokens N>=2]"
                " [--threads T] [--repeats R]\n";
        return 2;
    }
    try {
        if (threads > 0) set_num_threads(threads);
        auto t0 = chrono::steady_clock::now();
        Qwen3Model model = load_pretrained(model_dir);
        double load_s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

        int V = static_cast<int>(model.config.vocab_size);
        vector<int> prompt;
        for (int i = 0; i < prompt_len; i++) prompt.push_back((i * 7919 + 13) % V);

        greedy_generate(model, prompt, 2, {}, nullptr);   // warm-up
        vector<double> prefill, decode;
        Profiler::enabled = true;
        Profiler::reset();
        for (int r = 0; r < repeats; r++) {
            GenerationResult g = greedy_generate(model, prompt, new_tokens, {}, nullptr);
            prefill.push_back(g.prefill_seconds);
            decode.push_back(g.decode_steps / g.decode_seconds);
        }
        Profiler::enabled = false;

        double total = 0;
        for (auto& e : Profiler::seconds) total += e.second;
        cout << "{\"threads\": " << get_num_threads() << ", \"prompt_len\": " << prompt_len
             << ", \"new_tokens\": " << new_tokens << ", \"repeats\": " << repeats
             << ", \"load_seconds\": " << load_s
             << ", \"prefill_seconds\": " << median(prefill)
             << ", \"prefill_tokens_per_second\": " << prompt_len / median(prefill)
             << ", \"decode_tokens_per_second\": " << median(decode)
             << ", \"peak_rss_gib\": " << peak_rss_gib() << ", \"profile_seconds\": {";
        bool first = true;
        for (auto& e : Profiler::seconds) {
            cout << (first ? "" : ", ") << "\"" << e.first << "\": " << e.second;
            first = false;
        }
        cout << "}, \"profile_total_seconds\": " << total << "}" << endl;
    } catch (const exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
