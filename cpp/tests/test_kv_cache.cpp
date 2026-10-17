// Unit tests cho KVCache (cpp/include/qwen3/kv_cache.hpp).
#include "qwen3/kv_cache.hpp"
#include "test_utils.hpp"

// Tensor [B, N, T, D] co gia tri = 1000*b + 100*n + 10*(start + t) + d (de doc nguoc lai)
Tensor numbered(int B, int N, int T, int D, int start) {
    Tensor x({B, N, T, D});
    for (int b = 0; b < B; b++)
        for (int n = 0; n < N; n++)
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    x.at(b, n, t, d) = static_cast<float>(1000 * b + 100 * n + 10 * (start + t) + d);
    return x;
}

void test_append_and_read_back() {
    KVCache cache(2, 10);
    for (int layer = 0; layer < 2; layer++) {
        Tensor k = numbered(2, 3, 4, 5, 0);
        Tensor v = numbered(2, 3, 4, 5, 0);
        cache.append(layer, k, v);
    }
    cache.commit(4);
    for (int layer = 0; layer < 2; layer++) {
        Tensor k = numbered(2, 3, 1, 5, 4);
        Tensor v = numbered(2, 3, 1, 5, 4);
        cache.append(layer, k, v);
    }
    cache.commit(1);
    check(cache.seq_len == 5 && cache.layer_length(1) == 5, "seq_len = 4 + 1");
    bool ok = true;
    for (int j = 0; j < 5; j++)
        for (int b = 0; b < 2; b++)
            for (int h = 0; h < 3; h++)
                for (int d = 0; d < 5; d++) {
                    float expected = static_cast<float>(1000 * b + 100 * h + 10 * j + d);
                    if (cache.key(1, j, b, h)[d] != expected || cache.value(0, j, b, h)[d] != expected) ok = false;
                }
    check(ok, "doc lai dung key/value theo (token, batch, head)");
    check(cache.nbytes() == 2LL * 2 * 5 * 2 * 3 * 5 * 4, "nbytes = 2 (K,V) * L * S * B * Nkv * D * 4");
}

void test_rules() {
    KVCache cache(2, 6);
    Tensor k = numbered(1, 2, 3, 4, 0);
    cache.append(0, k, k);
    check_throws<runtime_error>([&] { cache.append(0, k, k); }, "1 layer append 2 lan trong 1 forward");
    check_throws<runtime_error>([&] { cache.commit(3); }, "commit khi layer 1 chua append");
    cache.rollback();
    check(cache.layer_length(0) == 0 && cache.seq_len == 0, "rollback bo append chua commit");

    cache.append(0, k, k);
    cache.append(1, k, k);
    cache.commit(3);
    Tensor big = numbered(1, 2, 4, 4, 3);
    check_throws<invalid_argument>([&] { cache.append(0, big, big); }, "vuot max_length (3 + 4 > 6)");
    Tensor other = numbered(2, 2, 1, 4, 3);
    check_throws<invalid_argument>([&] { cache.append(0, other, other); }, "batch khac phien hien tai");
    Tensor v = numbered(1, 2, 2, 4, 3);
    Tensor k1 = numbered(1, 2, 1, 4, 3);
    check_throws<invalid_argument>([&] { cache.append(0, k1, v); }, "k va v khac shape");
    check_throws<out_of_range>([&] { cache.append(2, k1, k1); }, "layer ngoai pham vi");

    cache.reset();
    check(cache.seq_len == 0 && cache.nbytes() == 0, "reset xoa het");
    check_throws<invalid_argument>([] { KVCache c(0, 4); }, "num_layers = 0");
}

int main() {
    test_append_and_read_back();
    test_rules();
    return report("test_kv_cache");
}
