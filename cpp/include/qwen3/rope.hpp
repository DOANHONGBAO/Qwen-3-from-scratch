#pragma once

#include "qwen3/tensor.hpp"

class RoPE {
public:
    RoPE(int head_dim, float rope_theta);

    Tensor forward(Tensor& x, int start_pos);

    int head_dim;
    float rope_theta;
    vector<float> inv_freq;   
};