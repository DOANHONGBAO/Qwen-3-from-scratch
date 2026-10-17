// Unit tests cho cac phep tinh trong cpp/include/qwen3/ops.hpp.
// Gia tri chuan (reference) duoc PyTorch tinh san trong ops_reference.hpp
// (sinh boi scripts/export_cpp_ops_reference.py).
#include "qwen3/ops.hpp"
#include "ops_reference.hpp"

int failures = 0;

void check(bool condition, string what) {
    if (!condition) {
        cout << "FAILED: " << what << "\n";
        failures++;
    }
}

// So sanh ket qua C++ voi PyTorch, cho phep sai so nho do lam tron float:
//   |c - t| <= atol + rtol * |t|   (giong torch.testing.assert_close)
// In ra sai so lon nhat de biet ket qua gan PyTorch den muc nao.
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
        if (isnan(c)) {             // NaN khong so sanh duoc -> dem rieng
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

// ------------------------------------------------------------ so voi PyTorch

void test_against_pytorch() {
    Tensor a(MATMUL_A_SHAPE, MATMUL_A), b(MATMUL_B_SHAPE, MATMUL_B);
    Tensor c = matmul(a, b);
    check_close(c, MATMUL_OUT_SHAPE, MATMUL_OUT, "matmul [3,5]x[5,4]");

    Tensor x(LINEAR_X_SHAPE, LINEAR_X), w(LINEAR_W_SHAPE, LINEAR_W);
    Tensor y = linear(x, w);
    check_close(y, LINEAR_OUT_SHAPE, LINEAR_OUT, "linear [2,3,7]x[6,7]^T");

    Tensor s(SOFTMAX_IN_SHAPE, SOFTMAX_IN);
    Tensor p = softmax_lastdim(s);
    check_close(p, SOFTMAX_OUT_SHAPE, SOFTMAX_OUT, "softmax (co gia tri ~1000 va -1e9)");

    Tensor e1(EW_A_SHAPE, EW_A), e2(EW_B_SHAPE, EW_B);
    Tensor sum = add(e1, e2);
    Tensor prod = mul(e1, e2);
    Tensor sc = scale(e1, 0.125f);
    check_close(sum, ADD_OUT_SHAPE, ADD_OUT, "add", 0.0f, 0.0f);     // phai bang tuyet doi
    check_close(prod, MUL_OUT_SHAPE, MUL_OUT, "mul", 0.0f, 0.0f);
    check_close(sc, SCALE_OUT_SHAPE, SCALE_OUT, "scale", 0.0f, 0.0f);

    Tensor v(SILU_IN_SHAPE, SILU_IN);
    Tensor sv = silu(v);
    check_close(sv, SILU_OUT_SHAPE, SILU_OUT, "silu [-8..8]");
}

// ------------------------------------------------------------ tinh chat toan hoc

void test_softmax_properties() {
    Tensor s(SOFTMAX_IN_SHAPE, SOFTMAX_IN);
    Tensor p = softmax_lastdim(s);
    int cols = p.size(-1);
    for (int r = 0; r < p.numel() / cols; r++) {
        float total = 0.0f;
        for (int c = 0; c < cols; c++) total += p.data()[r * cols + c];
        check(fabs(total - 1.0f) < 1e-6f, "softmax: tong moi hang = 1");
    }
    check(p.at(2, 4) == 0.0f && p.at(2, 5) == 0.0f, "softmax: o bi mask (-1e9) co xac suat 0");
}

void test_matmul_identity() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor eye({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Tensor c = matmul(a, eye);
    bool same = true;
    for (int i = 0; i < 6; i++) {
        if (c.data()[i] != a.data()[i]) same = false;
    }
    check(same, "matmul voi ma tran don vi giu nguyen");
}

void test_linear_equals_matmul_with_transpose() {
    // linear(x, w) = x * w^T. Neu w = b^T thi linear(x, b^T) phai bang matmul(x, b).
    Tensor x(MATMUL_A_SHAPE, MATMUL_A);   // [3, 5]
    Tensor b(MATMUL_B_SHAPE, MATMUL_B);   // [5, 4]
    Tensor bt = b.transpose(0, 1);        // [4, 5], dung layout [out, in] cua linear
    Tensor y1 = linear(x, bt);            // [3, 4]
    Tensor y2 = matmul(x, b);             // [3, 4]
    bool close = y1.get_shape() == y2.get_shape();
    for (int i = 0; i < y1.numel() && close; i++) {
        if (fabs(y1.data()[i] - y2.data()[i]) > 1e-5f) close = false;
    }
    check(close, "linear(x, b^T) == matmul(x, b)");
}

void test_invalid_inputs() {
    Tensor a({2, 3}), b({3, 2}), c({2, 2});
    check_throws<invalid_argument>([&] { add(a, b); }, "add: khac shape");
    check_throws<invalid_argument>([&] { mul(a, c); }, "mul: khac shape");
    check_throws<invalid_argument>([&] { matmul(a, c); }, "matmul: K khong khop");
    Tensor v3({2, 3, 4});
    check_throws<invalid_argument>([&] { matmul(v3, b); }, "matmul: khong phai 2 chieu");
    Tensor w({4, 5});
    check_throws<invalid_argument>([&] { linear(a, w); }, "linear: K khong khop");
}

int main() {
    test_against_pytorch();
    test_softmax_properties();
    test_matmul_identity();
    test_linear_equals_matmul_with_transpose();
    test_invalid_inputs();

    if (failures == 0) {
        cout << "test_ops: tat ca kiem tra deu dung\n";
        return 0;
    }
    cout << "test_ops: " << failures << " kiem tra SAI\n";
    return 1;
}