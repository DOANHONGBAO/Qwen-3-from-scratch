// Unit tests cho Attention (cpp/include/qwen3/attention.hpp) va KVCache.
// Gia tri chuan sinh boi scripts/export_cpp_attention_reference.py
// tu GroupedQueryAttention cua ban PyTorch (da kiem chung voi Hugging Face).
#include "qwen3/attention.hpp"
#include "qwen3/ops.hpp"
#include "attention_reference.hpp"
#include "test_utils.hpp"

Attention make_attention() {
    Attention attn(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, THETA, EPS);
    attn.q_proj = Tensor(Q_PROJ_SHAPE, Q_PROJ);
    attn.k_proj = Tensor(K_PROJ_SHAPE, K_PROJ);
    attn.v_proj = Tensor(V_PROJ_SHAPE, V_PROJ);
    attn.o_proj = Tensor(O_PROJ_SHAPE, O_PROJ);
    attn.q_norm.weight = Tensor(Q_NORM_SHAPE, Q_NORM);
    attn.k_norm.weight = Tensor(K_NORM_SHAPE, K_NORM);
    return attn;
}

// Lay token [start, end) cua x [B, T, H]
Tensor slice_tokens(Tensor& x, int start, int end) {
    int B = x.size(0), T = x.size(1), H = x.size(2);
    Tensor out({B, end - start, H});
    for (int b = 0; b < B; b++)
        for (int t = start; t < end; t++)
            for (int i = 0; i < H; i++) out.at(b, t - start, i) = x.at(b, t, i);
    return out;
}

// Noi theo chieu token: [B, T1, H] + [B, T2, H] -> [B, T1 + T2, H]
Tensor concat_tokens(Tensor& a, Tensor& c) {
    int B = a.size(0), T1 = a.size(1), T2 = c.size(1), H = a.size(2);
    Tensor out({B, T1 + T2, H});
    for (int b = 0; b < B; b++)
        for (int i = 0; i < H; i++) {
            for (int t = 0; t < T1; t++) out.at(b, t, i) = a.at(b, t, i);
            for (int t = 0; t < T2; t++) out.at(b, T1 + t, i) = c.at(b, t, i);
        }
    return out;
}

// ------------------------------------------------------------ so voi PyTorch

void test_qk_after_norm_and_rope() {
    // Lap lai buoc 1-3 cua forward de kiem tra rieng QK-Norm + RoPE
    Attention attn = make_attention();
    Tensor x(X_SHAPE, X);
    int B = 2, T = 5;
    Tensor q = linear(x, attn.q_proj);
    Tensor q4 = q.reshape({B, T, HEADS, HEAD_DIM});
    Tensor qn = attn.q_norm.forward(q4);
    Tensor qt = qn.transpose(1, 2);
    Tensor qr = attn.rope.forward(qt, 0);
    check_close(qr, Q_ROPE_SHAPE, Q_ROPE, "q sau QK-Norm + RoPE [2,4,5,8]");

    Tensor k = linear(x, attn.k_proj);
    Tensor k4 = k.reshape({B, T, KV_HEADS, HEAD_DIM});
    Tensor kn = attn.k_norm.forward(k4);
    Tensor kt = kn.transpose(1, 2);
    Tensor kr = attn.rope.forward(kt, 0);
    check_close(kr, K_ROPE_SHAPE, K_ROPE, "k sau QK-Norm + RoPE [2,2,5,8]");
}

void test_forward_against_pytorch() {
    Attention attn = make_attention();
    Tensor x(X_SHAPE, X);
    Tensor y = attn.forward(x, nullptr, 0);
    check_close(y, OUT_SHAPE, OUT, "attention khong cache [2,5,16]");
}

void test_cached_against_pytorch() {
    // prefill 3 token, roi 1, roi 1 token (giong sinh token that)
    Attention attn = make_attention();
    Tensor x(X_SHAPE, X);
    KVCache cache(1, 16);
    Tensor a = slice_tokens(x, 0, 3);
    Tensor b = slice_tokens(x, 3, 4);
    Tensor c = slice_tokens(x, 4, 5);
    Tensor ya = attn.forward(a, &cache, 0);
    cache.commit(3);
    Tensor yb = attn.forward(b, &cache, 0);
    cache.commit(1);
    Tensor yc = attn.forward(c, &cache, 0);
    cache.commit(1);
    Tensor ab = concat_tokens(ya, yb);
    Tensor all = concat_tokens(ab, yc);
    check_close(all, OUT_CACHED_SHAPE, OUT_CACHED, "attention co cache 3+1+1 token");
    check_close(all, OUT_SHAPE, OUT, "co cache == khong cache");
    check(cache.seq_len == 5 && cache.layer_length(0) == 5, "cache giu 5 token");
}

// ------------------------------------------------------------ tinh chat

void test_causal() {
    // Doi token cuoi khong duoc lam thay doi output cua cac token truoc
    Attention attn = make_attention();
    Tensor x(X_SHAPE, X);
    Tensor x2 = x;
    for (int i = 0; i < HIDDEN; i++) x2.at(0, 4, i) = 50.0f;
    Tensor y1 = attn.forward(x, nullptr, 0);
    Tensor y2 = attn.forward(x2, nullptr, 0);
    bool same = true;
    for (int t = 0; t < 4; t++)
        for (int i = 0; i < HIDDEN; i++)
            if (y1.at(0, t, i) != y2.at(0, t, i)) same = false;
    check(same, "causal: token tuong lai khong anh huong token truoc");
    bool changed = false;
    for (int i = 0; i < HIDDEN; i++)
        if (y1.at(0, 4, i) != y2.at(0, 4, i)) changed = true;
    check(changed, "token bi sua thi output cua no doi");
}

void test_batches_independent() {
    Attention attn = make_attention();
    Tensor x(X_SHAPE, X);
    Tensor x2 = x;
    for (int t = 0; t < 5; t++)
        for (int i = 0; i < HIDDEN; i++) x2.at(1, t, i) = 1.0f;
    Tensor y1 = attn.forward(x, nullptr, 0);
    Tensor y2 = attn.forward(x2, nullptr, 0);
    bool same = true;
    for (int t = 0; t < 5; t++)
        for (int i = 0; i < HIDDEN; i++)
            if (y1.at(0, t, i) != y2.at(0, t, i)) same = false;
    check(same, "batch 0 khong bi anh huong khi sua batch 1");
}

void test_single_token_attends_to_itself() {
    // Mot token duy nhat: softmax tren 1 phan tu = 1 => ctx = v, out = o_proj(v lap theo nhom)
    Attention attn = make_attention();
    Tensor x({1, 1, HIDDEN});
    for (int i = 0; i < HIDDEN; i++) x.at(0, 0, i) = X[static_cast<size_t>(i)];
    Tensor y = attn.forward(x, nullptr, 0);
    Tensor v = linear(x, attn.v_proj);                     // [1, 1, Nkv*D]
    Tensor ctx({1, 1, HEADS * HEAD_DIM});
    int group = HEADS / KV_HEADS;
    for (int h = 0; h < HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            ctx.at(0, 0, h * HEAD_DIM + d) = v.at(0, 0, (h / group) * HEAD_DIM + d);
    Tensor expected = linear(ctx, attn.o_proj);
    bool close = true;
    for (int i = 0; i < HIDDEN; i++)
        if (fabs(y.at(0, 0, i) - expected.at(0, 0, i)) > 1e-5f) close = false;
    check(close, "1 token: output = o_proj(v), dung nhom GQA h / group");
}

// ------------------------------------------------------------ input sai

void test_invalid_inputs() {
    check_throws<invalid_argument>([] { Attention a(16, 3, 2, 8, 1e6f, 1e-6f); }, "Nq khong chia het cho Nkv");
    check_throws<invalid_argument>([] { Attention a(16, 4, 2, 7, 1e6f, 1e-6f); }, "head_dim le");
    check_throws<invalid_argument>([] { Attention a(0, 4, 2, 8, 1e6f, 1e-6f); }, "hidden_size = 0");

    Attention attn = make_attention();
    Tensor wrong({1, 3, HIDDEN + 1});
    check_throws<invalid_argument>([&] { attn.forward(wrong, nullptr, 0); }, "chieu cuoi khac hidden_size");
    Tensor x2d({3, HIDDEN});
    check_throws<invalid_argument>([&] { attn.forward(x2d, nullptr, 0); }, "x khong phai 3 chieu");

    Attention bad = make_attention();
    bad.k_proj = Tensor({HEADS * HEAD_DIM, HIDDEN});      // nham so head cua k
    Tensor x(X_SHAPE, X);
    check_throws<invalid_argument>([&] { bad.forward(x, nullptr, 0); }, "k_proj sai shape");

    KVCache small(1, 4);
    check_throws<invalid_argument>([&] { attn.forward(x, &small, 0); }, "cache day (5 token > max_length 4)");
}

int main() {
    test_qk_after_norm_and_rope();
    test_forward_against_pytorch();
    test_cached_against_pytorch();
    test_causal();
    test_batches_independent();
    test_single_token_attends_to_itself();
    test_invalid_inputs();
    return report("test_attention");
}
