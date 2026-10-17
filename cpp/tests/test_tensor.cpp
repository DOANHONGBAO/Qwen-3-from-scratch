// Unit tests cho class Tensor (cpp/include/qwen3/tensor.hpp).
// Khong dung thu vien test: moi kiem tra sai se in ra "FAILED" va dem loi.
// Chuong trinh tra ve 0 neu moi kiem tra deu dung -> CTest bao PASSED.
#include "qwen3/tensor.hpp"

int failures = 0;

// Kiem tra mot dieu kien. Neu sai: in ten kiem tra va tang bien dem loi.
void check(bool condition, string what) {
    if (!condition) {
        cout << "FAILED: " << what << "\n";
        failures++;
    }
}

// Tao tensor co gia tri 0, 1, 2, ... theo thu tu bo nho (giong torch.arange).
Tensor arange(vector<int> shape) {
    Tensor t(shape);
    for (int i = 0; i < t.numel(); i++) t.data()[i] = (float)i;
    return t;
}

// ------------------------------------------------------------ cac bai test

void test_shape_strides_numel() {
    Tensor t({2, 3, 4});
    check(t.get_shape() == vector<int>({2, 3, 4}), "shape {2,3,4}");
    check(t.get_strides() == vector<int>({12, 4, 1}), "strides {12,4,1}");
    check(t.dim() == 3, "dim = 3");
    check(t.numel() == 24, "numel = 24");
    check(t.size(0) == 2 && t.size(-1) == 4, "size(0)=2, size(-1)=4");
}

void test_zero_initialised() {
    Tensor t({3, 5});
    bool all_zero = true;
    for (int i = 0; i < t.numel(); i++) {
        if (t.data()[i] != 0.0f) all_zero = false;
    }
    check(all_zero, "tensor moi tao toan so 0");
}

void test_constructor_with_data() {
    Tensor t({2, 2}, {1, 2, 3, 4});
    check(t.at(0, 0) == 1 && t.at(0, 1) == 2 && t.at(1, 0) == 3 && t.at(1, 1) == 4,
          "data theo thu tu row-major");
}

void test_at_matches_offset_formula() {
    Tensor t = arange({2, 3, 4});
    // vi tri = i*12 + j*4 + k
    check(t.at(0, 0, 0) == 0, "at(0,0,0) = 0");
    check(t.at(0, 1, 0) == 4, "at(0,1,0) = 4");
    check(t.at(1, 0, 0) == 12, "at(1,0,0) = 12");
    check(t.at(1, 2, 3) == 23, "at(1,2,3) = 23 (phan tu cuoi)");
    t.at(1, 2, 3) = 99.0f;   // ghi qua tham chieu
    check(t.data()[23] == 99.0f, "ghi qua at() thay doi dung o nho");
}

void test_4d_indexing() {
    Tensor t = arange({2, 3, 4, 5});
    check(t.get_strides() == vector<int>({60, 20, 5, 1}), "strides 4D");
    check(t.at(1, 2, 3, 4) == 119, "at(1,2,3,4) = 119");
}

void test_reshape() {
    Tensor t = arange({2, 3, 4});
    Tensor r = t.reshape({6, 4});
    check(r.get_shape() == vector<int>({6, 4}), "reshape shape {6,4}");
    check(r.at(5, 3) == 23 && r.at(1, 0) == 4, "reshape giu nguyen thu tu du lieu");
    r.at(0, 0) = -1.0f;
    check(t.at(0, 0, 0) == 0, "reshape tao ban sao: sua r khong doi t");
}

void test_transpose_2d() {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});   // [[1,2,3],[4,5,6]]
    Tensor u = t.transpose(0, 1);              // [[1,4],[2,5],[3,6]]
    check(u.get_shape() == vector<int>({3, 2}), "transpose 2D shape {3,2}");
    check(u.get_strides() == vector<int>({2, 1}), "transpose -> strides lien tuc moi");
    vector<float> expected = {1, 4, 2, 5, 3, 6};
    bool same = true;
    for (int i = 0; i < 6; i++) {
        if (u.data()[i] != expected[i]) same = false;
    }
    check(same, "transpose 2D sap xep lai bo nho: 1 4 2 5 3 6");
}

void test_transpose_3d_all_elements() {
    Tensor t = arange({2, 3, 4});
    Tensor u = t.transpose(1, 2);              // shape {2, 4, 3}
    check(u.get_shape() == vector<int>({2, 4, 3}), "transpose(1,2) shape {2,4,3}");
    bool ok = true;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k++)
                if (u.at(i, k, j) != t.at(i, j, k)) ok = false;
    check(ok, "u[i][k][j] == t[i][j][k] cho moi phan tu");
    Tensor back = u.transpose(-1, -2);         // chi so am, transpose 2 lan = ban dau
    bool same = back.get_shape() == t.get_shape();
    for (int i = 0; i < 24 && same; i++) {
        if (back.data()[i] != t.data()[i]) same = false;
    }
    check(same, "transpose 2 lan tro ve tensor ban dau");
}

void test_empty_and_scalar() {
    Tensor e;
    check(e.numel() == 0 && e.dim() == 1, "tensor rong: shape {0}");
    Tensor z({3, 0});
    check(z.numel() == 0, "shape co chieu 0 -> 0 phan tu");
    Tensor s(vector<int>{});
    check(s.numel() == 1 && s.dim() == 0, "shape {} -> 1 phan tu (scalar)");
}

// Kiem tra mot doan code co nem exception kieu E hay khong.
template <typename E, typename F>
void check_throws(F code, string what) {
    try {
        code();
    } catch (const E&) {
        return;   // dung: da nem loi dung kieu
    }
    cout << "FAILED (khong nem loi): " << what << "\n";
    failures++;
}

void test_invalid_inputs() {
    check_throws<invalid_argument>([] { Tensor t({2, -1}); }, "kich thuoc am");
    check_throws<invalid_argument>([] { Tensor t({2, 2}, {1, 2, 3}); }, "data thieu phan tu");
    Tensor t = arange({2, 3});
    check_throws<out_of_range>([&] { t.at(2, 0); }, "at: chi so hang vuot gioi han");
    check_throws<out_of_range>([&] { t.at(0, -1); }, "at: chi so am");
    check_throws<out_of_range>([&] { t.at(0); }, "at: sai so chieu (1 chi so cho tensor 2D)");
    check_throws<out_of_range>([&] { t.size(2); }, "size: chieu khong ton tai");
    check_throws<invalid_argument>([&] { t.reshape({4, 2}); }, "reshape sai so phan tu");
    check_throws<out_of_range>([&] { t.transpose(0, 2); }, "transpose: chieu khong ton tai");
}

int main() {
    test_shape_strides_numel();
    test_zero_initialised();
    test_constructor_with_data();
    test_at_matches_offset_formula();
    test_4d_indexing();
    test_reshape();
    test_transpose_2d();
    test_transpose_3d_all_elements();
    test_empty_and_scalar();
    test_invalid_inputs();

    if (failures == 0) {
        cout << "test_tensor: tat ca kiem tra deu dung\n";
        return 0;
    }
    cout << "test_tensor: " << failures << " kiem tra SAI\n";
    return 1;
}