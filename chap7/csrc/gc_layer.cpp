#include <iostream>
#include <torch/extension.h>

torch::Tensor gc_forward(torch::Tensor a, 
                         torch::Tensor x, 
                         torch::Tensor w,
                         torch::Tensor b) {
  return a.mm(x).mm(w) + b;
}

torch::Tensor gc_backward(torch::Tensor a, 
                          torch::Tensor x, 
                          torch::Tensor g) {
  return a.mm(x).t().mm(g);
}

PYBIND11_MODULE(gc_cpp, m) {
  m.def("forward", &gc_forward, "gc forward");
  m.def("backward", &gc_backward, "gc backward");
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &gc_forward, "gc forward");
//   m.def("backward", &gc_backward, "gc backward");
// }