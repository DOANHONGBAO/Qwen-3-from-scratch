#pragma once

#include <bits/stdc++.h>
using namespace std;

// Bo doc JSON toi gian, du cho config.json va header cua file safetensors.
// Ho tro: null, true/false, so, chuoi (co escape), mang, object.
class Json {
public:
    string type = "null";        // "null", "bool", "number", "string", "array", "object"
    bool boolean = false;
    double number = 0.0;
    string text;
    vector<Json> items;          // phan tu cua mang
    vector<string> keys;         // object: keys[i] -> values[i] (giu thu tu trong file)
    vector<Json> values;

    static Json parse(string source);

    bool has(string key);
    Json& get(string key);       // nem loi neu khong co key
    int as_int();                // so nguyen, nem loi neu khong phai so nguyen
    double as_number();
    string as_string();
    bool as_bool();
};

string read_text_file(string path);
