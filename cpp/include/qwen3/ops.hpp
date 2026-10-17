#pragma once

#include "qwen3/tensor.hpp"

Tensor add(Tensor& a, Tensor& b);     
Tensor mul(Tensor& a, Tensor& b);     
Tensor scale(Tensor& a, float s);    

Tensor silu(Tensor& x);               

Tensor softmax_lastdim(Tensor& x);

Tensor matmul(Tensor& a, Tensor& b);

Tensor linear(Tensor& x, Tensor& w);

// Tich vo huong sum_k a[k] * b[k], viet de compiler dung duoc SIMD (8 tong song song).
float dot_product(float* a, float* b, int n);

// So luong (thread) dung cho linear va attention (OpenMP). Khong co OpenMP: luon 1.
void set_num_threads(int n);
int get_num_threads();
