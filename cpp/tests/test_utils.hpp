// Ham kiem tra dung chung cho cac file test (moi file test la 1 chuong trinh rieng).
// Chi #include file nay trong DUNG MOT file .cpp cua moi test executable.
#pragma once

#include "qwen3/tensor.hpp"

int failures = 0;

void check(bool condition, string what) {
    if (!condition) {
        cout << "FAILED: " << what << "\n";
        failures++;
    }
}

// |c - t| <= atol + rtol * |t| (giong torch.testing.assert_close), dem rieng NaN,
// in sai so lon nhat de biet C++ gan PyTorch den muc nao.
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

// In ket qua cuoi va tra ve ma thoat cho CTest (0 = PASSED).
int report(string name) {
    if (failures == 0) {
        cout << name << ": tat ca kiem tra deu dung\n";
        return 0;
    }
    cout << name << ": " << failures << " kiem tra SAI\n";
    return 1;
}