#pragma once

#include "qwen3/tensor.hpp"

// Doc / ghi file .safetensors (dinh dang checkpoint cua Hugging Face):
//   [8 byte: N = do dai header, little-endian] [N byte: header JSON] [du lieu tensor]
// Header: { "ten": {"dtype": "BF16", "shape": [V, H], "data_offsets": [dau, cuoi]}, ... }
// data_offsets tinh tu byte dau tien SAU header. Khong co code Python ben trong
// (khac pickle .bin), nen doc file khong the chay code la.
class SafetensorsEntry {
public:
    string name;
    string dtype;          // "F32", "BF16" hoac "F16"
    vector<int> shape;
    long long begin = 0;   // byte bat dau (tinh tu sau header)
    long long end = 0;
};

class SafetensorsFile {
public:
    SafetensorsFile(string path);   // chi doc header, chua doc du lieu

    vector<string> names();
    bool has(string name);
    SafetensorsEntry& entry(string name);

    // Doc tensor `name`, doi sang float32, ghi vao dst (dst phai co dung count phan tu).
    void read_into(string name, float* dst, int count);
    Tensor load(string name);

    string path;
    long long data_start = 0;       // = 8 + N
    vector<SafetensorsEntry> entries;
};

// Ghi cac tensor ra file safetensors (float32). Dung cho test va cong cu.
void save_safetensors(string path, vector<string> names, vector<Tensor*> tensors);

// Doi 1 so bfloat16 / float16 (dang bit 16) sang float32.
float bf16_to_float(uint16_t bits);
float f16_to_float(uint16_t bits);
