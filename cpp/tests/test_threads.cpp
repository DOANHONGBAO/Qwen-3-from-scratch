// Tests cho phan toi uu M4: dot_product (SIMD) va chay nhieu thread (OpenMP).
// Yeu cau quan trong: so thread KHONG duoc lam doi ket qua (moi phan tu output luon
// duoc tinh boi cung mot vong lap, chi khac thread nao tinh).
#include "qwen3/attention.hpp"
#include "qwen3/ops.hpp"
#include "test_utils.hpp"

// Gia tri gia ngau nhien co dinh (khong can thu vien random)
Tensor filled(vector<int> shape, int seed) {
    Tensor t(shape);
    unsigned int state = static_cast<unsigned int>(seed) * 2654435761u + 1u;
    for (int i = 0; i < t.numel(); i++) {
        state = state * 1664525u + 1013904223u;
        t.data()[i] = static_cast<float>(state >> 8) / 16777216.0f * 2.0f - 1.0f;   // [-1, 1)
    }
    return t;
}

bool identical(Tensor& a, Tensor& b) {
    if (a.get_shape() != b.get_shape()) return false;
    for (int i = 0; i < a.numel(); i++)
        if (a.data()[i] != b.data()[i]) return false;
    return true;
}

void test_dot_product() {
    // So voi tong tinh bang double, moi do dai (ca phan du khi n khong chia het cho 8)
    bool ok = true;
    for (int n : {0, 1, 7, 8, 9, 15, 16, 127, 128, 1000, 3072}) {
        Tensor a = filled({max(n, 1)}, n), b = filled({max(n, 1)}, n + 1);
        double exact = 0.0;
        for (int k = 0; k < n; k++) exact += static_cast<double>(a.data()[k]) * b.data()[k];
        float got = dot_product(a.data(), b.data(), n);
        if (fabs(got - exact) > 1e-5 * (1.0 + sqrt(static_cast<double>(n)))) ok = false;
    }
    check(ok, "dot_product khop tong double voi moi do dai n");
}

void test_linear_same_for_any_thread_count() {
    Tensor x = filled({3, 5, 96}, 1);    // prefill: 15 hang
    Tensor w = filled({77, 96}, 2);      // N = 77 khong chia het cho so thread
    Tensor x1 = filled({1, 1, 96}, 3);   // decode: 1 hang
    set_num_threads(1);
    Tensor ref = linear(x, w);
    Tensor ref1 = linear(x1, w);
    bool same = true;
    for (int t : {2, 3, 4}) {
        set_num_threads(t);
        Tensor y = linear(x, w);
        Tensor y1 = linear(x1, w);
        if (!identical(y, ref) || !identical(y1, ref1)) same = false;
    }
    set_num_threads(1);
    check(same, "linear: 1, 2, 3, 4 thread cho ket qua giong het tung bit");

    // va van dung ve toan hoc: so voi matmul(x, w^T) (thu tu cong khac => sai so nho)
    Tensor x2 = x.reshape({15, 96});
    Tensor wt = w.transpose(0, 1);
    Tensor m = matmul(x2, wt);
    bool close = true;
    for (int i = 0; i < m.numel(); i++)
        if (fabs(m.data()[i] - ref.data()[i]) > 1e-4f) close = false;
    check(close, "linear(x, w) == matmul(x, w^T)");
}

void test_attention_same_for_any_thread_count() {
    Attention attn(32, 4, 2, 16, 1e6f, 1e-6f);
    attn.q_proj = filled({64, 32}, 4);
    attn.k_proj = filled({32, 32}, 5);
    attn.v_proj = filled({32, 32}, 6);
    attn.o_proj = filled({32, 64}, 7);
    Tensor x = filled({2, 9, 32}, 8);
    set_num_threads(1);
    Tensor ref = attn.forward(x, nullptr, 0);
    set_num_threads(4);
    Tensor y = attn.forward(x, nullptr, 0);
    set_num_threads(1);
    check(identical(y, ref), "attention: 1 va 4 thread cho ket qua giong het");
}

int main() {
    cout << "  so thread toi da (OpenMP): " << get_num_threads() << "\n";
    test_dot_product();
    test_linear_same_for_any_thread_count();
    test_attention_same_for_any_thread_count();
    check_throws<invalid_argument>([] { set_num_threads(0); }, "set_num_threads(0)");
    return report("test_threads");
}
