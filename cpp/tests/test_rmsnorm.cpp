// Unit tests cho RMSNorm (cpp/include/qwen3/rmsnorm.hpp).
// Gia tri chuan duoc sinh boi scripts/export_cpp_rmsnorm_reference.py
// tu ham rms_norm cua ban PyTorch (da kiem chung voi Hugging Face).
#include "qwen3/rmsnorm.hpp"
#include "rmsnorm_reference.hpp"

int failures = 0;

void check(bool condition, string what) {
    if (!condition) {
        cout << "FAILED: " << what << "\n";
        failures++;
    }
}

// |c - t| <= atol + rtol * |t|, dem rieng NaN, in sai so lon nhat.
void check_close(Tensor& got, vector<int> shape, vector<float> expected, string what,
                 float atol = 1e-5f, float rtol = 1e-5f) {
    if (got.get_shape() != shape) {
        cout << "FAILED: " << what << " (sai shape)\n";
        failures++;
        return;
    }
    float max_err = 0.0f;
    int nan_count = 0;
    bool ok = true;
    for (int i = 0; i < got.numel(); i++) {
        float c = got.data()[i];
        float t = expected[i];
        if (isnan(c)) {
            nan_count++;
            ok = false;
            continue;
        }
        float err = fabs(c - t);
        if (err > atol + rtol * fabs(t)) ok = false;
        max_err = max(max_err, err);
    }
    cout << "  " << what << ": max |C++ - PyTorch| = " << max_err;
    if (nan_count > 0) cout << ", so phan tu NaN = " << nan_count;
    cout << "\n";
    check(ok, what + " khop PyTorch");
}

template <typename E, typename F>
void check_throws(F code, string what) {
    try {
        code();
    } catch (const E&) {
        return;
    }
    cout << "FAILED (khong nem loi): " << what << "\n";
    failures++;
}

// Tao RMSNorm va nap weight tu mang reference (vai tro cua weight loader sau nay).
RMSNorm make_norm(vector<int> w_shape, vector<float> w_values) {
    RMSNorm norm(w_shape[0], EPS);
    norm.weight = Tensor(w_shape, w_values);
    return norm;
}

// ------------------------------------------------------------ so voi PyTorch

void test_against_pytorch() {
    RMSNorm norm = make_norm(W_SHAPE, W);

    Tensor x(X_SHAPE, X);
    Tensor y = norm.forward(x);
    check_close(y, X_OUT_SHAPE, X_OUT, "hidden states [2,3,8]");

    Tensor big(BIG_SHAPE, BIG);
    Tensor y_big = norm.forward(big);
    check_close(y_big, BIG_OUT_SHAPE, BIG_OUT, "gia tri lon (~1e4)");

    Tensor small(SMALL_SHAPE, SMALL);
    Tensor y_small = norm.forward(small);
    check_close(y_small, SMALL_OUT_SHAPE, SMALL_OUT, "gia tri nho (~1e-4, eps co tac dung)");

    // QK-Norm: cung mot lop, chi khac hidden_size = head_dim va input 4 chieu
    RMSNorm qnorm = make_norm(WQ_SHAPE, WQ);
    Tensor q(Q_SHAPE, Q);
    Tensor y_q = qnorm.forward(q);
    check_close(y_q, Q_OUT_SHAPE, Q_OUT, "QK-Norm per-head [1,2,3,4]");
}

// ------------------------------------------------------------ tinh chat toan hoc

void test_default_weight_is_ones() {
    RMSNorm norm(5, 1e-6f);
    bool ones = norm.weight.get_shape() == vector<int>{5};
    for (int i = 0; i < 5; i++) {
        if (norm.weight.at(i) != 1.0f) ones = false;
    }
    check(ones, "weight mac dinh = 1, shape [5]");
}

void test_unit_rms_after_norm() {
    // weight = 1  =>  mean(y^2) = mean(x^2) / (mean(x^2) + eps) ~ 1
    RMSNorm norm(W_SHAPE[0], EPS);
    Tensor x(X_SHAPE, X);
    Tensor y = norm.forward(x);
    int d = norm.hidden_size;
    for (int r = 0; r < y.numel() / d; r++) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += y.data()[r * d + i] * y.data()[r * d + i];
        check(fabs(s / static_cast<float>(d) - 1.0f) < 1e-5f, "moi hang co RMS = 1");
    }
}

void test_scale_invariance() {
    // RMSNorm(c * x) == RMSNorm(x) voi c > 0 (khi eps rat nho so voi x^2)
    RMSNorm norm = make_norm(W_SHAPE, W);
    Tensor x(X_SHAPE, X);
    Tensor x3 = x;                     // ban sao
    for (int i = 0; i < x3.numel(); i++) x3.data()[i] *= 3.0f;
    Tensor y1 = norm.forward(x);
    Tensor y2 = norm.forward(x3);
    bool close = true;
    for (int i = 0; i < y1.numel(); i++) {
        if (fabs(y1.data()[i] - y2.data()[i]) > 1e-5f) close = false;
    }
    check(close, "RMSNorm(3x) == RMSNorm(x)");
}

void test_zero_input() {
    // x = 0: eps tranh chia cho 0, ket qua phai la 0 chu khong phai NaN
    RMSNorm norm(4, 1e-6f);
    Tensor x({2, 4});
    Tensor y = norm.forward(x);
    bool zeros = true;
    for (int i = 0; i < y.numel(); i++) {
        if (y.data()[i] != 0.0f) zeros = false;   // NaN != 0 nen cung bi bat
    }
    check(zeros, "x = 0 cho ra 0 (khong NaN)");
}

void test_rows_are_independent() {
    // Doi hang 1 khong duoc lam thay doi ket qua hang 0
    RMSNorm norm(3, 1e-6f);
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {1, 2, 3, 400, -5, 0.5f});
    Tensor ya = norm.forward(a);
    Tensor yb = norm.forward(b);
    bool same = true;
    for (int i = 0; i < 3; i++) {
        if (ya.at(0, i) != yb.at(0, i)) same = false;
    }
    check(same, "moi hang duoc chuan hoa doc lap");
}

void test_input_not_modified() {
    RMSNorm norm(3, 1e-6f);
    Tensor x({1, 3}, {1, 2, 3});
    norm.forward(x);
    check(x.at(0, 0) == 1.0f && x.at(0, 1) == 2.0f && x.at(0, 2) == 3.0f,
          "forward khong sua input");
}

void test_invalid_inputs() {
    check_throws<invalid_argument>([] { RMSNorm n(0, 1e-6f); }, "hidden_size = 0");
    check_throws<invalid_argument>([] { RMSNorm n(-3, 1e-6f); }, "hidden_size am");
    check_throws<invalid_argument>([] { RMSNorm n(4, 0.0f); }, "eps = 0");
    check_throws<invalid_argument>([] { RMSNorm n(4, -1e-6f); }, "eps am");
    check_throws<invalid_argument>([] { RMSNorm n(4, NAN); }, "eps = NaN");

    RMSNorm norm(4, 1e-6f);
    Tensor wrong({2, 5});
    check_throws<invalid_argument>([&] { norm.forward(wrong); }, "chieu cuoi khac hidden_size");
    Tensor scalar;
    check_throws<invalid_argument>([&] { norm.forward(scalar); }, "input scalar 0 chieu");

    RMSNorm bad(4, 1e-6f);
    bad.weight = Tensor({3});          // weight loader nap nham shape
    Tensor x({2, 4});
    check_throws<invalid_argument>([&] { bad.forward(x); }, "weight sai shape");
}

int main() {
    test_against_pytorch();
    test_default_weight_is_ones();
    test_unit_rms_after_norm();
    test_scale_invariance();
    test_zero_input();
    test_rows_are_independent();
    test_input_not_modified();
    test_invalid_inputs();

    if (failures == 0) {
        cout << "test_rmsnorm: tat ca kiem tra deu dung\n";
        return 0;
    }
    cout << "test_rmsnorm: " << failures << " kiem tra SAI\n";
    return 1;
}