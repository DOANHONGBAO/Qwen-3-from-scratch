#pragma once

#include "qwen3/model.hpp"

class GenerationResult {
public:
    vector<int> tokens;          // chi cac token MOI sinh ra
    double prefill_seconds = 0;  // thoi gian xu ly prompt
    double decode_seconds = 0;   // tong thoi gian cac buoc decode
    int decode_steps = 0;        // so buoc decode (moi buoc 1 token, dung cache)
};

// Vi tri cua gia tri lon nhat (bang nhau thi lay id nho nhat, giong torch.argmax).
int argmax(float* values, int n);

// Greedy decoding batch 1 voi KV cache:
//   prefill: ca prompt 1 lan -> token dau tien
//   decode : moi buoc dua 1 token moi, cache dai them 1
// Dung khi sinh ra 1 token trong eos_ids (token EOS van duoc giu, giong HF).
// prompt_logits != nullptr: luu logits [1, T, V] cua prompt (de so sanh voi Python).
GenerationResult greedy_generate(Qwen3Model& model, vector<int> prompt, int max_new_tokens,
                                 vector<int> eos_ids, Tensor* prompt_logits);
