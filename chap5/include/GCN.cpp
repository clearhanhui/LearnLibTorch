/* 
 * @File    :   GCN.cpp
 * @Time    :   2022/07/21
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "GCN.h"
#include <iostream>
#include <cmath>

GCNLayerImpl::GCNLayerImpl(int in_features, int out_features){
    w = torch::randn((in_features, out_features));
    b = torch::randn((out_features));
    w.requires_grad_();
    b.requires_grad_();
    float dev = sqrt(out_features);
    torch::nn::init::uniform_(w, -dev, dev);
    torch::nn::init::uniform_(b, -dev, dev);
    register_parameter("w", w);
    register_parameter("b", b);
}

torch::Tensor GCNLayerImpl::forward(torch::Tensor x, torch::Tensor a){
    torch::Tensor out = torch::mm(a, torch::mm(x, w));
    return out;
}

GCN::GCN(){
    dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(0.5));
    gc1 = GCNLayer(1433, 16);
    gc2 = GCNLayer(16, 7);
    register_module("gc1", gc1);
    register_module("gc2", gc2);
    register_module("dropout", dropout);
}

torch::Tensor GCN::forward(torch::Tensor x, torch::Tensor a){
    x = F::relu(gc1(x, a));
    x = dropout(x);
    x = gc2(x, a);
    return x;
}