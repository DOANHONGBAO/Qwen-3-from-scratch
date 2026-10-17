#include "qwen3/tensor.hpp"


Tensor::Tensor() {
    shape_ = {0};
    compute_strides();
}

Tensor::Tensor(vector<int> shape) {
    for (int s : shape) {
        if (s < 0) {
            throw invalid_argument("Tensor: kich thuoc mot chieu khong duoc am");
        }
    }
    shape_ = shape;
    compute_strides();

    int n = 1;
    for (int s : shape_) n = n * s;   
    data_.assign(n, 0.0f);            
}

Tensor::Tensor(vector<int> shape, vector<float> data) : Tensor(shape) {
    if ((int)data.size() != numel()) {
        throw invalid_argument("Tensor: so phan tu cua data khong khop voi shape");
    }
    data_ = move(data);  
}


vector<int> Tensor::get_shape() { return shape_; }
vector<int> Tensor::get_strides() { return strides_; }
int Tensor::dim() { return (int)shape_.size(); }
int Tensor::numel() { return (int)data_.size(); }

int Tensor::size(int d) {
    if (d < 0) d = d + dim();
    if (d < 0 || d >= dim()) {
        throw out_of_range("Tensor::size: chieu khong ton tai");
    }
    return shape_[d];
}


void Tensor::compute_strides() {
    int n = dim();
    strides_.assign(n, 1);
    for (int d = n - 2; d >= 0; d--) {
        strides_[d] = strides_[d + 1] * shape_[d + 1];
    }
}

int Tensor::offset(vector<int> indices) {
    if ((int)indices.size() != dim()) {
        throw out_of_range("Tensor::at: so chi so khac so chieu cua tensor");
    }
    int pos = 0;
    for (int d = 0; d < dim(); d++) {
        if (indices[d] < 0 || indices[d] >= shape_[d]) {
            throw out_of_range("Tensor::at: chi so vuot qua kich thuoc");
        }
        pos = pos + indices[d] * strides_[d];
    }
    return pos;
}


float& Tensor::at(int i) { return data_[offset({i})]; }
float& Tensor::at(int i, int j) { return data_[offset({i, j})]; }
float& Tensor::at(int i, int j, int k) { return data_[offset({i, j, k})]; }
float& Tensor::at(int i, int j, int k, int l) { return data_[offset({i, j, k, l})]; }

float* Tensor::data() { return data_.data(); }


Tensor Tensor::reshape(vector<int> new_shape) {
    int n = 1;
    for (int s : new_shape) n = n * s;
    if (n != numel()) {
        throw invalid_argument("Tensor::reshape: so phan tu khong khop");
    }
    return Tensor(new_shape, data_);   
}

Tensor Tensor::transpose(int dim0, int dim1) {
    int n = dim();
    if (dim0 < 0) dim0 = dim0 + n;
    if (dim1 < 0) dim1 = dim1 + n;
    if (dim0 < 0 || dim0 >= n || dim1 < 0 || dim1 >= n) {
        throw out_of_range("Tensor::transpose: chieu khong ton tai");
    }

    vector<int> new_shape = shape_;
    swap(new_shape[dim0], new_shape[dim1]);
    Tensor out(new_shape);

    vector<int> idx(n);
    for (int o = 0; o < out.numel(); o++) {
        int rest = o;
        for (int d = 0; d < n; d++) {
            idx[d] = rest / out.strides_[d];
            rest = rest % out.strides_[d];
        }
        swap(idx[dim0], idx[dim1]);
        int src = 0;
        for (int d = 0; d < n; d++) src = src + idx[d] * strides_[d];
        out.data_[o] = data_[src];
    }
    return out;
}