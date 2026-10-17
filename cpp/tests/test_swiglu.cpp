// Unit tests cho SwiGLU MLP (cpp/include/qwen3/swiglu.hpp).
// Gia tri chuan sinh boi scripts/export_cpp_swiglu_reference.py
// tu SwiGLUFeedForward cua ban PyTorch (da kiem chung voi Hugging Face).
#include "qwen3/swiglu.hpp"
#include "qwen3/ops.hpp"
#include "swiglu_reference.hpp"
#include "test_utils.hpp"

// Tao SwiGLU va nap 3 weight tu reference (vai tro cua weight loader sau nay).
SwiGLU make_mlp() {
    SwiGLU mlp(GATE_W_SHAPE[1], GATE_W_SHAPE[0]);   // H = 8, I = 12
    mlp.gate_proj = Tensor(GATE_W_SHAPE, GATE_W);
    mlp.up_proj = Tensor(UP_W_SHAPE, UP_W);
    mlp.down_proj = Tensor(DOWN_W_SHAPE, DOWN_W);
    return mlp;
}

// ------------------------------------------------------------ so voi PyTorch

void test_intermediate_steps() {
    // Kiem tra TUNG BUOC: neu sai, biet ngay sai o gate, up, hidden hay down.
    SwiGLU mlp = make_mlp();
    Tensor x(X_SHAPE, X);
    Tensor gate = linear(x, mlp.gate_proj);
    check_close(gate, GATE_SHAPE, GATE, "buoc 1: gate = x * gate_proj^T");
    Tensor up = linear(x, mlp.up_proj);
    check_close(up, UP_SHAPE, UP, "buoc 2: up = x * up_proj^T");
    Tensor act = silu(gate);
    Tensor hidden = mul(act, up);
    check_close(hidden, HIDDEN_SHAPE, HIDDEN, "buoc 3: hidden = silu(gate) * up");
}

void test_forward_against_pytorch() {
    SwiGLU mlp = make_mlp();
    Tensor x(X_SHAPE, X);
    Tensor y = mlp.forward(x);
    check_close(y, OUT_SHAPE, OUT, "forward [2,3,8] -> [2,3,8]");
}

// ------------------------------------------------------------ tinh chat

void test_scalar_formula() {
    // H = I = 1, ca 3 weight = 1  =>  out = silu(x) * x  (tinh tay duoc)
    SwiGLU mlp(1, 1);
    mlp.gate_proj = Tensor({1, 1}, {1.0f});
    mlp.up_proj = Tensor({1, 1}, {1.0f});
    mlp.down_proj = Tensor({1, 1}, {1.0f});
    Tensor x({3, 1}, {-2.0f, 0.0f, 3.0f});
    Tensor y = mlp.forward(x);
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        float v = x.at(i, 0);
        float expected = v / (1.0f + exp(-v)) * v;
        if (fabs(y.at(i, 0) - expected) > 1e-6f) ok = false;
    }
    check(ok, "H=I=1, weight=1: out = silu(x) * x");
}

void test_fresh_layer_outputs_zero() {
    // Weight khoi tao = 0 (chua nap checkpoint) => output = 0, khong NaN
    SwiGLU mlp(8, 12);
    Tensor x(X_SHAPE, X);
    Tensor y = mlp.forward(x);
    bool zeros = y.get_shape() == X_SHAPE;
    for (int i = 0; i < y.numel(); i++) {
        if (y.data()[i] != 0.0f) zeros = false;
    }
    check(zeros, "weight = 0 cho output = 0, dung shape");
}

void test_tokens_are_independent() {
    // MLP xu ly tung token rieng: doi token 1 khong anh huong token 0
    SwiGLU mlp = make_mlp();
    Tensor x(X_SHAPE, X);
    Tensor x2 = x;
    for (int i = 0; i < 8; i++) x2.at(0, 1, i) = 100.0f;   // sua token (b=0, t=1)
    Tensor y1 = mlp.forward(x);
    Tensor y2 = mlp.forward(x2);
    bool same = true;
    for (int i = 0; i < 8; i++) {
        if (y1.at(0, 0, i) != y2.at(0, 0, i)) same = false;
    }
    check(same, "token 0 khong bi anh huong khi sua token 1");
}

// ------------------------------------------------------------ input sai

void test_invalid_inputs() {
    check_throws<invalid_argument>([] { SwiGLU m(0, 4); }, "hidden_size = 0");
    check_throws<invalid_argument>([] { SwiGLU m(4, -1); }, "intermediate_size am");

    SwiGLU mlp(4, 6);
    Tensor wrong({2, 5});
    check_throws<invalid_argument>([&] { mlp.forward(wrong); }, "chieu cuoi khac hidden_size");

    Tensor x({2, 4});
    SwiGLU bad_gate(4, 6);
    bad_gate.gate_proj = Tensor({5, 4});
    check_throws<invalid_argument>([&] { bad_gate.forward(x); }, "gate_proj sai shape");

    SwiGLU swapped(4, 6);
    swapped.down_proj = Tensor({6, 4});   // nham [I,H] thay vi [H,I]
    check_throws<invalid_argument>([&] { swapped.forward(x); }, "down_proj bi dao chieu");
}

int main() {
    test_intermediate_steps();
    test_forward_against_pytorch();
    test_scalar_formula();
    test_fresh_layer_outputs_zero();
    test_tokens_are_independent();
    test_invalid_inputs();
    return report("test_swiglu");
}