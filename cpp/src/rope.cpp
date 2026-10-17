#include "qwen3/rope.hpp"

RoPE::RoPE(int head_dim, float rope_theta) {
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw invalid_argument("RoPE: head_dim phai la so chan > 0");
    }
    if (!(rope_theta > 0.0f) || isinf(rope_theta)) {
        throw invalid_argument("RoPE: rope_theta phai la so duong huu han");
    }
    this->head_dim = head_dim;
    this->rope_theta = rope_theta;

    int half = head_dim / 2;
    inv_freq.assign(half, 0.0f);
    for (int i = 0; i < half; i++) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[i] = 1.0f / pow(rope_theta, exponent);
    }
}

Tensor RoPE::forward(Tensor& x, int start_pos) {
    if (x.dim() != 4 || x.size(3) != head_dim) {
        throw invalid_argument("RoPE: x phai co shape [B, N, T, head_dim]");
    }
    if (start_pos < 0) {
        throw invalid_argument("RoPE: start_pos phai >= 0");
    }

    int B = x.size(0), N = x.size(1), T = x.size(2), D = head_dim;
    int half = D / 2;
    Tensor out(x.get_shape());
    float* px = x.data();
    float* po = out.data();
    vector<float> c(half), s(half);

    for (int t = 0; t < T; t++) {
        float pos = static_cast<float>(start_pos + t);
        for (int i = 0; i < half; i++) {
            float angle = pos * inv_freq[i];
            c[i] = cos(angle);
            s[i] = sin(angle);
        }
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                int base = ((b * N + n) * T + t) * D;   // dau vector [b, n, t, :]
                float* xr = px + base;
                float* yr = po + base;
                for (int i = 0; i < half; i++) {
                    float x1 = xr[i];
                    float x2 = xr[i + half];
                    yr[i] = x1 * c[i] - x2 * s[i];
                    yr[i + half] = x2 * c[i] + x1 * s[i];
                }
            }
        }
    }
    return out;
}