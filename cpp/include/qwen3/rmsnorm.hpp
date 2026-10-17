#pragma once

#include "qwen3/tensor.hpp"


class RMSNorm {
public:
    RMSNorm(int hidden_size, float eps);

    Tensor forward(Tensor& x);

    int hidden_size;
    float eps;
    Tensor weight;   
};