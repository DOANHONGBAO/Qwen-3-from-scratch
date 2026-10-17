// qwen3_run: nap checkpoint Hugging Face (config.json + safetensors) va sinh token greedy.
//
//   qwen3_run --model-dir DIR --ids 9707,11,1879 [--ids ...] [--max-new-tokens 16]
//             [--eos 151643] [--logits-out PREFIX]
//
// Tokenizer khong nam trong C++: token id do Python tao ra
// (scripts/verify_cpp_pretrained.py). Moi prompt in ra 1 dong JSON tren stdout.
// --logits-out PREFIX: ghi logits cua prompt, float32 tho [T, V], vao PREFIX_p<i>.bin
#include "qwen3/generate.hpp"
#include "qwen3/model.hpp"

vector<int> parse_ids(string text) {
    vector<int> ids;
    stringstream ss(text);
    string item;
    while (getline(ss, item, ',')) {
        if (!item.empty()) ids.push_back(stoi(item));
    }
    return ids;
}

string to_json_list(vector<int>& v) {
    string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) s += ", ";
        s += to_string(v[i]);
    }
    return s + "]";
}

// Bo nho dinh (peak RSS) cua tien trinh, doc tu /proc (Linux)
double peak_rss_gib() {
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return stod(line.substr(6)) / (1024.0 * 1024.0);   // kB -> GiB
        }
    }
    return -1.0;
}

void usage() {
    cerr << "usage: qwen3_run --model-dir DIR --ids 1,2,3 [--ids ...] [--max-new-tokens N]"
            " [--eos ID] [--logits-out PREFIX]\n";
}

int main(int argc, char** argv) {
    string model_dir, logits_out;
    vector<vector<int>> prompts;
    vector<int> eos_ids;
    int max_new_tokens = 16;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (i + 1 >= argc) {
            usage();
            return 2;
        }
        string value = argv[++i];
        if (a == "--model-dir") model_dir = value;
        else if (a == "--ids") prompts.push_back(parse_ids(value));
        else if (a == "--max-new-tokens") max_new_tokens = stoi(value);
        else if (a == "--eos") eos_ids.push_back(stoi(value));
        else if (a == "--logits-out") logits_out = value;
        else {
            usage();
            return 2;
        }
    }
    if (model_dir.empty() || prompts.empty()) {
        usage();
        return 2;
    }

    try {
        auto t0 = chrono::steady_clock::now();
        Qwen3Model model = load_pretrained(model_dir);
        double load_s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        cerr << "loaded " << model_dir << " in " << load_s << " s, peak RSS " << peak_rss_gib() << " GiB\n";

        for (size_t p = 0; p < prompts.size(); p++) {
            Tensor prompt_logits;
            Tensor* keep = logits_out.empty() ? nullptr : &prompt_logits;
            GenerationResult r = greedy_generate(model, prompts[p], max_new_tokens, eos_ids, keep);
            if (keep) {
                string path = logits_out + "_p" + to_string(p) + ".bin";
                ofstream f(path, ios::binary);
                f.write(reinterpret_cast<char*>(prompt_logits.data()),
                        static_cast<streamsize>(prompt_logits.numel()) * 4);
            }
            double tok_s = r.decode_seconds > 0 ? r.decode_steps / r.decode_seconds : 0.0;
            cout << "{\"prompt\": " << p << ", \"ids\": " << to_json_list(prompts[p])
                 << ", \"generated\": " << to_json_list(r.tokens)
                 << ", \"prefill_seconds\": " << r.prefill_seconds
                 << ", \"decode_steps\": " << r.decode_steps
                 << ", \"decode_tokens_per_second\": " << tok_s
                 << ", \"load_seconds\": " << load_s
                 << ", \"peak_rss_gib\": " << peak_rss_gib() << "}" << endl;
        }
    } catch (const exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
