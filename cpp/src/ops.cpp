#include "qwen3/ops.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

static void check_same_shape(Tensor& a, Tensor& b, string op) {
    if (a.get_shape() != b.get_shape()) {
        throw invalid_argument(op + ": hai tensor phai cung shape");
    }
}


Tensor add(Tensor& a, Tensor& b) {
    check_same_shape(a, b, "add");
    Tensor out(a.get_shape());
    float* pa = a.data();
    float* pb = b.data();
    float* po = out.data();
    int n = a.numel();
    for (int i = 0; i < n; i++) po[i] = pa[i] + pb[i];
    return out;
}

Tensor mul(Tensor& a, Tensor& b) {
    check_same_shape(a, b, "mul");
    Tensor out(a.get_shape());
    float* pa = a.data();
    float* pb = b.data();
    float* po = out.data();
    int n = a.numel();
    for (int i = 0; i < n; i++) po[i] = pa[i] * pb[i];
    return out;
}

Tensor scale(Tensor& a, float s) {
    Tensor out(a.get_shape());
    float* pa = a.data();
    float* po = out.data();
    int n = a.numel();
    for (int i = 0; i < n; i++) po[i] = pa[i] * s;
    return out;
}


Tensor silu(Tensor& x) {
    Tensor out(x.get_shape());
    float* px = x.data();
    float* po = out.data();
    int n = x.numel();
    for (int i = 0; i < n; i++) po[i] = px[i] / (1.0f + exp(-px[i]));
    return out;
}


Tensor softmax_lastdim(Tensor& x) {
    if (x.dim() == 0) throw invalid_argument("softmax: tensor phai co it nhat 1 chieu");
    Tensor out(x.get_shape());
    int cols = x.size(-1);                      
    if (cols == 0) return out;
    int rows = x.numel() / cols;                
    float* px = x.data();
    float* po = out.data();

    for (int r = 0; r < rows; r++) {
        float* in = px + r * cols;              
        float* o = po + r * cols;

        float mx = in[0];
        for (int c = 1; c < cols; c++) mx = max(mx, in[c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            o[c] = exp(in[c] - mx);
            sum += o[c];
        }
        for (int c = 0; c < cols; c++) o[c] /= sum;
    }
    return out;
}


Tensor matmul(Tensor& a, Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2) {
        throw invalid_argument("matmul: can 2 tensor 2 chieu [M,K] x [K,N]");
    }
    int M = a.size(0), K = a.size(1), N = b.size(1);
    if (b.size(0) != K) {
        throw invalid_argument("matmul: so cot cua a phai bang so hang cua b");
    }
    Tensor out({M, N});                         
    float* pa = a.data();
    float* pb = b.data();
    float* po = out.data();

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float aik = pa[i * K + k];
            for (int j = 0; j < N; j++) {
                po[i * N + j] += aik * pb[k * N + j];
            }
        }
    }
    return out;
}


float dot_product(float* a, float* b, int n) {
    // 8 tong rieng (acc[0..7]) thay vi 1: cac phep cong khong phai cho nhau nua, va
    // compiler gom 8 phep nhan-cong thanh 1-2 lenh SIMD (SSE/AVX). Thu tu cong khac
    // vong lap don => ket qua co the lech bit cuoi so voi ban cu (sai so ~1e-7).
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        for (int u = 0; u < 8; u++) acc[u] += a[k + u] * b[k + u];
    }
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; k < n; k++) s += a[k] * b[k];   // phan du khi n khong chia het cho 8
    return s;
}

Tensor linear(Tensor& x, Tensor& w) {
    if (w.dim() != 2) throw invalid_argument("linear: w phai co shape [N, K]");
    if (x.dim() == 0) throw invalid_argument("linear: x phai co it nhat 1 chieu");
    int N = w.size(0), K = w.size(1);
    if (x.size(-1) != K) {
        throw invalid_argument("linear: chieu cuoi cua x phai bang so cot cua w");
    }
    int rows = (K == 0) ? 0 : x.numel() / K;

    vector<int> out_shape = x.get_shape();
    out_shape.back() = N;                      
    Tensor out(out_shape);
    float* px = x.data();
    float* pw = w.data();
    float* po = out.data();

    // Vong ngoai la hang n cua weight: moi hang w[n] (K so) duoc doc tu RAM 1 lan roi dung
    // cho MOI hang cua x. Prefill (rows = T) khong phai doc lai ca ma tran weight T lan.
    // Moi thread tinh 1 khoang n rieng => khong ghi chong len nhau, khong can khoa.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        float* wn = pw + n * K;
        for (int r = 0; r < rows; r++) {
            po[r * N + n] = dot_product(px + r * K, wn, K);
        }
    }
    return out;
}

void set_num_threads(int n) {
    if (n < 1) throw invalid_argument("set_num_threads: n phai >= 1");
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

int get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
