#pragma once

#include "qwen3/tensor.hpp"

Tensor add(Tensor& a, Tensor& b);     
Tensor mul(Tensor& a, Tensor& b);     
Tensor scale(Tensor& a, float s);    

Tensor silu(Tensor& x);               

Tensor softmax_lastdim(Tensor& x);

Tensor matmul(Tensor& a, Tensor& b);

Tensor linear(Tensor& x, Tensor& w);