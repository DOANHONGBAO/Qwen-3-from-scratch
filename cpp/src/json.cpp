#include "qwen3/json.hpp"

// Parser de quy (recursive descent): moi ham doc mot loai gia tri tai vi tri pos.
class JsonParser {
public:
    JsonParser(string& source) : s(source) {}

    Json parse_document() {
        Json v = parse_value();
        skip_spaces();
        if (pos != s.size()) fail("con ky tu thua sau gia tri JSON");
        return v;
    }

private:
    string& s;
    size_t pos = 0;

    void fail(string what) {
        throw invalid_argument("JSON: " + what + " (vi tri " + to_string(pos) + ")");
    }

    void skip_spaces() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) {
            pos++;
        }
    }

    bool match(string word) {
        if (s.compare(pos, word.size(), word) == 0) {
            pos += word.size();
            return true;
        }
        return false;
    }

    Json parse_value() {
        skip_spaces();
        if (pos >= s.size()) fail("het du lieu");
        char c = s[pos];
        Json v;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            v.type = "string";
            v.text = parse_string();
            return v;
        }
        if (match("true")) {
            v.type = "bool";
            v.boolean = true;
            return v;
        }
        if (match("false")) {
            v.type = "bool";
            return v;
        }
        if (match("null")) return v;
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(string("ky tu khong hop le '") + c + "'");
        return v;
    }

    Json parse_number() {
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() && (isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.' ||
                                  s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
            pos++;
        }
        string token = s.substr(start, pos - start);
        size_t used = 0;
        Json v;
        v.type = "number";
        try {
            v.number = stod(token, &used);
        } catch (const exception&) {
            pos = start;
            fail("so khong hop le '" + token + "'");
        }
        if (used != token.size()) {
            pos = start;
            fail("so khong hop le '" + token + "'");
        }
        return v;
    }

    // Ma Unicode (\uXXXX) -> cac byte UTF-8
    void append_utf8(string& out, unsigned int cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    unsigned int parse_hex4() {
        if (pos + 4 > s.size()) fail("\\u thieu chu so");
        unsigned int cp = 0;
        for (int i = 0; i < 4; i++) {
            char h = s[pos++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<unsigned int>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned int>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned int>(h - 'A' + 10);
            else fail("\\u co ky tu khong phai hex");
        }
        return cp;
    }

    string parse_string() {
        pos++;  // bo dau "
        string out;
        while (true) {
            if (pos >= s.size()) fail("chuoi chua dong dau \"");
            char c = s[pos++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos >= s.size()) fail("escape bi cat");
            char e = s[pos++];
            if (e == '"') out += '"';
            else if (e == '\\') out += '\\';
            else if (e == '/') out += '/';
            else if (e == 'b') out += '\b';
            else if (e == 'f') out += '\f';
            else if (e == 'n') out += '\n';
            else if (e == 'r') out += '\r';
            else if (e == 't') out += '\t';
            else if (e == 'u') {
                unsigned int cp = parse_hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF && match("\\u")) {   // cap surrogate UTF-16
                    unsigned int low = parse_hex4();
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                }
                append_utf8(out, cp);
            } else {
                fail(string("escape khong hop le \\") + e);
            }
        }
        return out;
    }

    Json parse_array() {
        pos++;  // bo [
        Json v;
        v.type = "array";
        skip_spaces();
        if (pos < s.size() && s[pos] == ']') {
            pos++;
            return v;
        }
        while (true) {
            v.items.push_back(parse_value());
            skip_spaces();
            if (pos < s.size() && s[pos] == ',') {
                pos++;
                continue;
            }
            if (pos < s.size() && s[pos] == ']') {
                pos++;
                return v;
            }
            fail("mang thieu ',' hoac ']'");
        }
    }

    Json parse_object() {
        pos++;  // bo {
        Json v;
        v.type = "object";
        skip_spaces();
        if (pos < s.size() && s[pos] == '}') {
            pos++;
            return v;
        }
        while (true) {
            skip_spaces();
            if (pos >= s.size() || s[pos] != '"') fail("key cua object phai la chuoi");
            string key = parse_string();
            skip_spaces();
            if (pos >= s.size() || s[pos] != ':') fail("thieu ':' sau key");
            pos++;
            v.keys.push_back(key);
            v.values.push_back(parse_value());
            skip_spaces();
            if (pos < s.size() && s[pos] == ',') {
                pos++;
                continue;
            }
            if (pos < s.size() && s[pos] == '}') {
                pos++;
                return v;
            }
            fail("object thieu ',' hoac '}'");
        }
    }
};

Json Json::parse(string source) {
    JsonParser parser(source);
    return parser.parse_document();
}

bool Json::has(string key) {
    if (type != "object") return false;
    for (size_t i = 0; i < keys.size(); i++) {
        if (keys[i] == key) return true;
    }
    return false;
}

Json& Json::get(string key) {
    if (type != "object") throw invalid_argument("JSON: khong phai object, khong co key '" + key + "'");
    for (size_t i = 0; i < keys.size(); i++) {
        if (keys[i] == key) return values[i];
    }
    throw invalid_argument("JSON: khong co key '" + key + "'");
}

double Json::as_number() {
    if (type != "number") throw invalid_argument("JSON: gia tri khong phai so");
    return number;
}

int Json::as_int() {
    double x = as_number();
    if (x != floor(x) || x < -2147483648.0 || x > 2147483647.0) {
        throw invalid_argument("JSON: gia tri khong phai so nguyen int");
    }
    return static_cast<int>(x);
}

string Json::as_string() {
    if (type != "string") throw invalid_argument("JSON: gia tri khong phai chuoi");
    return text;
}

bool Json::as_bool() {
    if (type != "bool") throw invalid_argument("JSON: gia tri khong phai true/false");
    return boolean;
}

string read_text_file(string path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("khong mo duoc file: " + path);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
