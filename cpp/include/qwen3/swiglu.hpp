#pragma once

#include "qwen3/tensor.hpp"


class SwiGLU {
public:
    SwiGLU(int hidden_size, int intermediate_size);

    Tensor forward(Tensor& x);

    int hidden_size;
    int intermediate_size;
    Tensor gate_proj;   
    Tensor up_proj;     
    Tensor down_proj;   
};