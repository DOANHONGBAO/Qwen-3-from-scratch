#include "qwen3/rmsnorm.hpp"

RMSNorm::RMSNorm(int hidden_size, float eps) {
    if (hidden_size <= 0) {
        throw invalid_argument("RMSNorm: hidden_size phai > 0");
    }
    if (!(eps > 0.0f) || isinf(eps)) {
        throw invalid_argument("RMSNorm: eps phai la so duong huu han");
    }
    this->hidden_size = hidden_size;
    this->eps = eps;
    weight = Tensor({hidden_size});
    for (int i = 0; i < hidden_size; i++) {
        weight.at(i) = 1.0f;
    }
}

Tensor RMSNorm::forward(Tensor& x) {
    if (x.dim() < 1 || x.size(-1) != hidden_size) {
        throw invalid_argument("RMSNorm: chieu cuoi cua x phai bang hidden_size");
    }
    if (weight.dim() != 1 || weight.numel() != hidden_size) {
        throw invalid_argument("RMSNorm: weight phai co shape [hidden_size]");
    }

    Tensor out(x.get_shape());
    int d = hidden_size;
    int rows = x.numel() / d;          // so vector can chuan hoa
    float* px = x.data();
    float* pw = weight.data();
    float* po = out.data();

    for (int r = 0; r < rows; r++) {
        float* xr = px + r * d;        // hang thu r cua input
        float* yr = po + r * d;        // hang thu r cua output

        float sum_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            sum_sq += xr[i] * xr[i];
        }
        float mean_sq = sum_sq / static_cast<float>(d);
        float inv_rms = 1.0f / sqrt(mean_sq + eps);

        for (int i = 0; i < d; i++) {
            yr[i] = pw[i] * (xr[i] * inv_rms);   // cung thu tu nhu HF: weight * (x * rsqrt)
        }
    }
    return out;
}