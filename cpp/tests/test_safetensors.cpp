// Unit tests cho bo doc JSON va safetensors (cpp/include/qwen3/json.hpp, safetensors.hpp).
#include "qwen3/json.hpp"
#include "qwen3/safetensors.hpp"
#include "test_utils.hpp"

string temp_path(string name) {
    return filesystem::temp_directory_path().string() + "/qwen3_cpp_test_" + name;
}

// Ghi file safetensors "bang tay": header JSON + byte du lieu tuy y
void write_raw(string path, string header, vector<unsigned char> data) {
    ofstream f(path, ios::binary);
    unsigned long long n = header.size();
    for (int i = 0; i < 8; i++) {
        char b = static_cast<char>((n >> (8 * i)) & 0xFF);
        f.write(&b, 1);
    }
    f.write(header.data(), static_cast<streamsize>(header.size()));
    f.write(reinterpret_cast<char*>(data.data()), static_cast<streamsize>(data.size()));
}

void test_json() {
    Json j = Json::parse(R"({"a": 1, "b": [true, false, null], "c": {"d": -2.5e3},
                            "s": "x\"y\\z\nà中", "e": []})");
    check(j.get("a").as_int() == 1, "so nguyen");
    check(j.get("b").items.size() == 3 && j.get("b").items[0].as_bool() && j.get("b").items[2].type == "null",
          "mang true/false/null");
    check(j.get("c").get("d").as_number() == -2500.0, "object long nhau, so mu");
    check(j.get("s").as_string() == "x\"y\\z\n\xc3\xa0\xe4\xb8\xad", "escape va \\u -> UTF-8");
    check(j.get("e").type == "array" && j.get("e").items.empty(), "mang rong");
    check(j.has("a") && !j.has("zz"), "has()");
    check(j.keys[0] == "a" && j.keys[4] == "e", "giu thu tu key trong file");

    check_throws<invalid_argument>([] { Json::parse("{\"a\": }"); }, "thieu gia tri");
    check_throws<invalid_argument>([] { Json::parse("[1, 2"); }, "mang chua dong");
    check_throws<invalid_argument>([] { Json::parse("{\"a\": 1} x"); }, "ky tu thua");
    check_throws<invalid_argument>([] { Json::parse("\"abc"); }, "chuoi chua dong");
    check_throws<invalid_argument>([] { Json::parse("{\"a\": 1.5}").get("a").as_int(); }, "1.5 khong phai int");
    check_throws<invalid_argument>([] { Json::parse("{}").get("x"); }, "key khong ton tai");
}

void test_half_precision_conversion() {
    check(bf16_to_float(0x3F80) == 1.0f, "bf16 0x3F80 = 1");
    check(bf16_to_float(0xC020) == -2.5f, "bf16 0xC020 = -2.5");
    check(bf16_to_float(0x0000) == 0.0f, "bf16 0 = 0");
    check(f16_to_float(0x3C00) == 1.0f, "f16 0x3C00 = 1");
    check(f16_to_float(0x3800) == 0.5f, "f16 0x3800 = 0.5");
    check(f16_to_float(0xC000) == -2.0f, "f16 0xC000 = -2");
    check(f16_to_float(0x7BFF) == 65504.0f, "f16 lon nhat = 65504");
    check(f16_to_float(0x0001) == ldexp(1.0f, -24), "f16 subnormal nho nhat = 2^-24");
    check(isinf(f16_to_float(0x7C00)) && isnan(f16_to_float(0x7E00)), "f16 inf / NaN");
}

void test_read_mixed_dtypes() {
    // F32 [2] = {1.5, -3}, BF16 [3] = {1, -2.5, 0}, F16 [1] = {0.5}
    vector<unsigned char> data;
    for (float f : {1.5f, -3.0f}) {
        unsigned char b[4];
        memcpy(b, &f, 4);
        data.insert(data.end(), b, b + 4);
    }
    for (uint16_t h : {uint16_t(0x3F80), uint16_t(0xC020), uint16_t(0x0000), uint16_t(0x3800)}) {
        data.push_back(static_cast<unsigned char>(h & 0xFF));
        data.push_back(static_cast<unsigned char>(h >> 8));
    }
    string header = R"({"__metadata__": {"format": "pt"},
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "BF16", "shape": [3, 1], "data_offsets": [8, 14]},
        "c": {"dtype": "F16", "shape": [1], "data_offsets": [14, 16]}})";
    string path = temp_path("mixed.safetensors");
    write_raw(path, header, data);

    SafetensorsFile st(path);
    check(st.names() == vector<string>{"a", "b", "c"}, "doc 3 tensor, bo qua __metadata__");
    Tensor a = st.load("a"), b = st.load("b"), c = st.load("c");
    check(a.at(0) == 1.5f && a.at(1) == -3.0f, "F32 doc dung");
    check(b.get_shape() == vector<int>{3, 1} && b.at(0, 0) == 1.0f && b.at(1, 0) == -2.5f && b.at(2, 0) == 0.0f,
          "BF16 -> float32 dung, dung shape");
    check(c.at(0) == 0.5f, "F16 -> float32 dung");
    check_throws<invalid_argument>([&] { st.load("zz"); }, "tensor khong ton tai");
}

void test_round_trip() {
    Tensor w({2, 3}, {1, 2, 3, 4, 5, -6.25f});
    Tensor v({4}, {0.1f, 0.2f, 0.3f, 0.4f});
    string path = temp_path("roundtrip.safetensors");
    save_safetensors(path, {"w", "v"}, {&w, &v});
    SafetensorsFile st(path);
    Tensor w2 = st.load("w"), v2 = st.load("v");
    bool same = w2.get_shape() == w.get_shape() && v2.get_shape() == v.get_shape();
    for (int i = 0; i < 6; i++) same = same && w2.data()[i] == w.data()[i];
    for (int i = 0; i < 4; i++) same = same && v2.data()[i] == v.data()[i];
    check(same, "save_safetensors -> doc lai giong het");
    check(st.data_start % 8 == 0, "header duoc canh 8 byte");
}

void test_invalid_files() {
    string p1 = temp_path("bad_offsets.safetensors");
    write_raw(p1, R"({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}})", vector<unsigned char>(8));
    check_throws<runtime_error>([&] { SafetensorsFile st(p1); }, "data_offsets vuot ra ngoai file");

    string p2 = temp_path("bad_size.safetensors");
    write_raw(p2, R"({"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}})", vector<unsigned char>(8));
    check_throws<runtime_error>([&] { SafetensorsFile st(p2); }, "so byte khong khop shape");

    string p3 = temp_path("bad_dtype.safetensors");
    write_raw(p3, R"({"a": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}})", vector<unsigned char>(8));
    check_throws<invalid_argument>([&] { SafetensorsFile st(p3); }, "dtype chua ho tro");

    string p4 = temp_path("short.safetensors");
    { ofstream f(p4, ios::binary); f << "abc"; }
    check_throws<runtime_error>([&] { SafetensorsFile st(p4); }, "file qua ngan");
    check_throws<runtime_error>([&] { SafetensorsFile st(temp_path("khong_co.safetensors")); }, "file khong ton tai");
}

int main() {
    test_json();
    test_half_precision_conversion();
    test_read_mixed_dtypes();
    test_round_trip();
    test_invalid_files();
    return report("test_safetensors");
}
