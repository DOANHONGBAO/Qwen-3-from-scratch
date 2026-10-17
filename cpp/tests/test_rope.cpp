// Unit tests cho RoPE (cpp/include/qwen3/rope.hpp).
// Gia tri chuan sinh boi scripts/export_cpp_rope_reference.py
// tu qwen3_scratch.layers.rope (da kiem chung voi Hugging Face).
#include "qwen3/rope.hpp"
#include "rope_reference.hpp"
#include "test_utils.hpp"

// Tich vo huong cua hai vector dau tien (b = 0, n = 0, t = 0) trong 2 tensor
float dot_first(Tensor& a, Tensor& b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += a.data()[i] * b.data()[i];
    return s;
}

// ------------------------------------------------------------ so voi PyTorch

void test_inv_freq_qwen3() {
    // Bang tan so cua cau hinh that: head_dim = 128, rope_theta = 1e6
    RoPE rope(128, THETA);
    Tensor got({64}, rope.inv_freq);
    check_close(got, INV_FREQ_128_SHAPE, INV_FREQ_128, "inv_freq (D=128, theta=1e6)", 0.0f, 1e-6f);
    check(rope.inv_freq[0] == 1.0f, "inv_freq[0] = 1 (cap xoay nhanh nhat: 1 radian/token)");
}

void test_against_pytorch() {
    RoPE rope(Q_SHAPE[3], THETA);

    Tensor q(Q_SHAPE, Q);
    Tensor yq = rope.forward(q, 0);
    check_close(yq, Q_OUT_SHAPE, Q_OUT, "prefill q [1,2,5,8], vi tri 0..4");

    Tensor k(K_SHAPE, K);
    Tensor yk = rope.forward(k, 7);
    check_close(yk, K_OUT_SHAPE, K_OUT, "k [2,1,3,8] sau 7 token trong cache, vi tri 7..9");

    Tensor far(FAR_SHAPE, FAR);
    Tensor yf = rope.forward(far, 30000);
    check_close(yf, FAR_OUT_SHAPE, FAR_OUT, "ngu canh dai, vi tri 30000..30001");
}

// ------------------------------------------------------------ tinh chat toan hoc

void test_position_zero_is_identity() {
    // Vi tri 0: goc = 0 => cos = 1, sin = 0 => khong doi
    RoPE rope(8, THETA);
    Tensor x(Q_SHAPE, Q);
    Tensor first({1, 2, 1, 8});
    for (int n = 0; n < 2; n++)
        for (int i = 0; i < 8; i++) first.at(0, n, 0, i) = x.at(0, n, 0, i);
    Tensor y = rope.forward(first, 0);
    bool same = true;
    for (int i = 0; i < y.numel(); i++) {
        if (y.data()[i] != first.data()[i]) same = false;
    }
    check(same, "vi tri 0 giu nguyen vector");
}

void test_rotation_preserves_length() {
    // Phep xoay khong doi do dai cua tung cap (x[i], x[i + D/2])
    RoPE rope(8, THETA);
    Tensor x(K_SHAPE, K);
    Tensor y = rope.forward(x, 7);
    bool ok = true;
    for (int v = 0; v < x.numel() / 8; v++) {
        for (int i = 0; i < 4; i++) {
            float a1 = x.data()[v * 8 + i], a2 = x.data()[v * 8 + i + 4];
            float b1 = y.data()[v * 8 + i], b2 = y.data()[v * 8 + i + 4];
            if (fabs((a1 * a1 + a2 * a2) - (b1 * b1 + b2 * b2)) > 1e-5f) ok = false;
        }
    }
    check(ok, "do dai moi cap duoc bao toan");
}

void test_relative_position() {
    // Tinh chat quan trong nhat: <rope(q, m), rope(k, n)> chi phu thuoc m - n.
    // q o vi tri 5, k o vi tri 2  vs  q o vi tri 105, k o vi tri 102: cung khoang cach 3
    RoPE rope(8, THETA);
    Tensor q({1, 1, 1, 8}, {0.3f, -1.2f, 0.8f, 0.5f, -0.7f, 1.1f, 0.2f, -0.4f});
    Tensor k({1, 1, 1, 8}, {1.0f, 0.4f, -0.6f, 0.9f, 0.1f, -1.3f, 0.7f, 0.2f});
    Tensor q5 = rope.forward(q, 5), k2 = rope.forward(k, 2);
    Tensor q105 = rope.forward(q, 105), k102 = rope.forward(k, 102);
    float near_pair = dot_first(q5, k2, 8);
    float far_pair = dot_first(q105, k102, 8);
    check(fabs(near_pair - far_pair) < 1e-4f, "q.k chi phu thuoc khoang cach (m - n)");

    Tensor k4 = rope.forward(k, 4);   // khoang cach khac (1) => tich khac
    check(fabs(dot_first(q5, k4, 8) - near_pair) > 1e-3f, "khoang cach khac cho tich khac");
}

void test_heads_and_batches_share_rotation() {
    // Cung mot vector dat o 2 head / 2 batch khac nhau, cung vi tri => ket qua giong het
    RoPE rope(4, THETA);
    Tensor x({2, 2, 1, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
    Tensor y = rope.forward(x, 9);
    bool same = true;
    for (int v = 1; v < 4; v++)
        for (int i = 0; i < 4; i++)
            if (y.data()[v * 4 + i] != y.data()[i]) same = false;
    check(same, "moi head va moi batch dung chung mot phep xoay");
}

void test_input_not_modified() {
    RoPE rope(8, THETA);
    Tensor x(K_SHAPE, K);
    rope.forward(x, 7);
    bool same = true;
    for (int i = 0; i < x.numel(); i++) {
        if (x.data()[i] != K[i]) same = false;
    }
    check(same, "forward khong sua input");
}

// ------------------------------------------------------------ input sai

void test_invalid_inputs() {
    check_throws<invalid_argument>([] { RoPE r(0, 1e6f); }, "head_dim = 0");
    check_throws<invalid_argument>([] { RoPE r(7, 1e6f); }, "head_dim le");
    check_throws<invalid_argument>([] { RoPE r(8, 0.0f); }, "rope_theta = 0");
    check_throws<invalid_argument>([] { RoPE r(8, NAN); }, "rope_theta = NaN");

    RoPE rope(8, THETA);
    Tensor x3({2, 3, 8});
    check_throws<invalid_argument>([&] { rope.forward(x3, 0); }, "x khong phai 4 chieu");
    Tensor wrong_d({1, 1, 2, 6});
    check_throws<invalid_argument>([&] { rope.forward(wrong_d, 0); }, "chieu cuoi khac head_dim");
    Tensor x({1, 1, 2, 8});
    check_throws<invalid_argument>([&] { rope.forward(x, -1); }, "start_pos am");
}

int main() {
    test_inv_freq_qwen3();
    test_against_pytorch();
    test_position_zero_is_identity();
    test_rotation_preserves_length();
    test_relative_position();
    test_heads_and_batches_share_rotation();
    test_input_not_modified();
    test_invalid_inputs();
    return report("test_rope");
}