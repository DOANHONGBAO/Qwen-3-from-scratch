#pragma once
#include <bits/stdc++.h>
using namespace std;



class Tensor {
public:
    Tensor();
    Tensor(vector<int> shape);
    Tensor(vector<int> shape,vector<float> data);

    vector<int> shape;
    vector<int> strides;
    int dim();
    int numel();
    int size(int d);


    float& at(int i);
    float& at(int i,int j);
    float& at(int i,int j,int k);
    float& at(int i,int j,int k,int l);
    float* data();


    Tensor reshape(vector<int> new_shape);
    Tensor transpose(int dim0,int dim1);

private:
    vector<float> data_;
    vector<int> shape_;
    vector<int> strides_;

    void compute_strides();
    int offset(vector<int> indices);
};