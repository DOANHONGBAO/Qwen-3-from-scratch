#include "qwen3/swiglu.hpp"

#include "qwen3/ops.hpp"

SwiGLU::SwiGLU(int hidden_size, int intermediate_size) {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw invalid_argument("SwiGLU: hidden_size va intermediate_size phai > 0");
    }
    this->hidden_size = hidden_size;
    this->intermediate_size = intermediate_size;
    gate_proj = Tensor({intermediate_size, hidden_size});
    up_proj = Tensor({intermediate_size, hidden_size});
    down_proj = Tensor({hidden_size, intermediate_size});
}

Tensor SwiGLU::forward(Tensor& x) {
    if (x.dim() < 1 || x.size(-1) != hidden_size) {
        throw invalid_argument("SwiGLU: chieu cuoi cua x phai bang hidden_size");
    }
    vector<int> ih = {intermediate_size, hidden_size};
    vector<int> hi = {hidden_size, intermediate_size};
    if (gate_proj.get_shape() != ih || up_proj.get_shape() != ih ||
        down_proj.get_shape() != hi) {
        throw invalid_argument("SwiGLU: weight sai shape (can [I,H], [I,H], [H,I])");
    }

    Tensor gate = linear(x, gate_proj);      
    Tensor up = linear(x, up_proj);          
    Tensor act = silu(gate);                 
    Tensor hidden = mul(act, up);            
    return linear(hidden, down_proj);       
}