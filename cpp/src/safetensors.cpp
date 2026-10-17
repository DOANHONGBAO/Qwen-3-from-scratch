#include "qwen3/safetensors.hpp"

#include "qwen3/json.hpp"

static int dtype_size(string dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    throw invalid_argument("safetensors: dtype chua ho tro: " + dtype);
}

float bf16_to_float(uint16_t bits) {
    // bfloat16 = 16 bit CAO cua float32 -> chi can dich trai 16 bit
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

float f16_to_float(uint16_t bits) {
    // float16: 1 bit dau, 5 bit mu (bias 15), 10 bit dinh tri
    uint32_t sign = (static_cast<uint32_t>(bits) >> 15) & 1u;
    uint32_t exp = (static_cast<uint32_t>(bits) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(bits) & 0x3FFu;
    float value;
    if (exp == 0) {
        value = ldexp(static_cast<float>(mant), -24);            // so 0 va so subnormal
    } else if (exp == 31) {
        value = mant == 0 ? INFINITY : NAN;                      // vo cung / NaN
    } else {
        value = ldexp(static_cast<float>(mant + 1024u), static_cast<int>(exp) - 25);
    }
    return sign ? -value : value;
}

SafetensorsFile::SafetensorsFile(string path) {
    this->path = path;
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("safetensors: khong mo duoc file " + path);
    f.seekg(0, ios::end);
    long long file_size = static_cast<long long>(f.tellg());
    f.seekg(0, ios::beg);

    unsigned char len_bytes[8];
    if (file_size < 8 || !f.read(reinterpret_cast<char*>(len_bytes), 8)) {
        throw runtime_error("safetensors: file qua ngan " + path);
    }
    unsigned long long n = 0;
    for (int i = 7; i >= 0; i--) n = (n << 8) | len_bytes[i];     // little-endian
    if (n == 0 || static_cast<long long>(n) > file_size - 8 || n > 100000000ull) {
        throw runtime_error("safetensors: do dai header khong hop le");
    }
    string header(static_cast<size_t>(n), '\0');
    f.read(&header[0], static_cast<streamsize>(n));
    data_start = 8 + static_cast<long long>(n);
    long long data_size = file_size - data_start;

    Json root = Json::parse(header);
    if (root.type != "object") throw runtime_error("safetensors: header phai la object JSON");
    for (size_t i = 0; i < root.keys.size(); i++) {
        if (root.keys[i] == "__metadata__") continue;
        Json& info = root.values[i];
        SafetensorsEntry e;
        e.name = root.keys[i];
        e.dtype = info.get("dtype").as_string();
        long long count = 1;
        for (Json& d : info.get("shape").items) {
            int s = d.as_int();
            if (s < 0) throw runtime_error("safetensors: shape am o " + e.name);
            e.shape.push_back(s);
            count *= s;
        }
        Json& offsets = info.get("data_offsets");
        if (offsets.items.size() != 2) throw runtime_error("safetensors: data_offsets sai o " + e.name);
        e.begin = static_cast<long long>(offsets.items[0].as_number());
        e.end = static_cast<long long>(offsets.items[1].as_number());
        if (e.begin < 0 || e.end < e.begin || e.end > data_size) {
            throw runtime_error("safetensors: data_offsets vuot ra ngoai file o " + e.name);
        }
        if (e.end - e.begin != count * dtype_size(e.dtype)) {
            throw runtime_error("safetensors: so byte khong khop shape/dtype o " + e.name);
        }
        if (count > 2147483647LL) throw runtime_error("safetensors: tensor qua lon cho int: " + e.name);
        entries.push_back(e);
    }
}

vector<string> SafetensorsFile::names() {
    vector<string> out;
    for (SafetensorsEntry& e : entries) out.push_back(e.name);
    return out;
}

bool SafetensorsFile::has(string name) {
    for (SafetensorsEntry& e : entries) {
        if (e.name == name) return true;
    }
    return false;
}

SafetensorsEntry& SafetensorsFile::entry(string name) {
    for (SafetensorsEntry& e : entries) {
        if (e.name == name) return e;
    }
    throw invalid_argument("safetensors: khong co tensor '" + name + "' trong " + path);
}

void SafetensorsFile::read_into(string name, float* dst, int count) {
    SafetensorsEntry& e = entry(name);
    int item = dtype_size(e.dtype);
    if ((e.end - e.begin) / item != count) {
        throw invalid_argument("safetensors: '" + name + "' co so phan tu khac voi cho chua");
    }
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("safetensors: khong mo duoc file " + path);
    f.seekg(data_start + e.begin);

    if (e.dtype == "F32") {   // doc thang vao dst (may x86 la little-endian giong file)
        f.read(reinterpret_cast<char*>(dst), static_cast<streamsize>(count) * 4);
        if (!f) throw runtime_error("safetensors: doc thieu du lieu o " + name);
        return;
    }
    // BF16 / F16: doc tung khoi 1M phan tu de khong ton them nhieu RAM
    int chunk = 1 << 20;
    vector<uint16_t> buf(static_cast<size_t>(min(count, chunk)));
    for (int done = 0; done < count; done += chunk) {
        int n = min(chunk, count - done);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<streamsize>(n) * 2);
        if (!f) throw runtime_error("safetensors: doc thieu du lieu o " + name);
        if (e.dtype == "BF16") {
            for (int i = 0; i < n; i++) dst[done + i] = bf16_to_float(buf[static_cast<size_t>(i)]);
        } else {
            for (int i = 0; i < n; i++) dst[done + i] = f16_to_float(buf[static_cast<size_t>(i)]);
        }
    }
}

Tensor SafetensorsFile::load(string name) {
    Tensor t(entry(name).shape);
    read_into(name, t.data(), t.numel());
    return t;
}

void save_safetensors(string path, vector<string> names, vector<Tensor*> tensors) {
    if (names.size() != tensors.size()) {
        throw invalid_argument("save_safetensors: so ten khac so tensor");
    }
    string header = "{";
    long long offset = 0;
    for (size_t i = 0; i < names.size(); i++) {
        long long bytes = static_cast<long long>(tensors[i]->numel()) * 4;
        string shape;
        vector<int> dims = tensors[i]->get_shape();
        for (size_t d = 0; d < dims.size(); d++) {
            if (d > 0) shape += ",";
            shape += to_string(dims[d]);
        }
        if (i > 0) header += ",";
        header += "\"" + names[i] + "\":{\"dtype\":\"F32\",\"shape\":[" + shape +
                  "],\"data_offsets\":[" + to_string(offset) + "," + to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += "}";
    while (header.size() % 8 != 0) header += ' ';   // dinh dang khuyen nghi canh 8 byte

    ofstream f(path, ios::binary);
    if (!f) throw runtime_error("save_safetensors: khong ghi duoc " + path);
    unsigned long long n = header.size();
    for (int i = 0; i < 8; i++) {
        char b = static_cast<char>((n >> (8 * i)) & 0xFFull);
        f.write(&b, 1);
    }
    f.write(header.data(), static_cast<streamsize>(header.size()));
    for (Tensor* t : tensors) {
        f.write(reinterpret_cast<char*>(t->data()), static_cast<streamsize>(t->numel()) * 4);
    }
    if (!f) throw runtime_error("save_safetensors: loi khi ghi " + path);
}
