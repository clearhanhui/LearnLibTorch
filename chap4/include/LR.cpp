/* 
 * @File    :   LR.cpp
 * @Time    :   2022/07/11
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "LR.h"

LinearRegression::LinearRegression(int in_dim, int out_dim){
    lin = torch::nn::Linear(in_dim, out_dim);
    lin = register_module("lin", lin);
}

torch::Tensor LinearRegression::forward(torch::Tensor x){
    x = lin->forward(x);
    return x;
}