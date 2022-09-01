/*
 * @File    :   TensorTransform.h
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_transform() {
  torch::Tensor a = torch::randn({2, 3});
  torch::Tensor b = a.transpose(0, 1);
  torch::Tensor c = a.reshape({-1, 1});
  torch::Tensor d = a.view({3, 2});
  torch::Tensor e = a.toType(torch::kFloat32);

  std::cout <<a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << d << std::endl;
  std::cout << e << std::endl;

  // 读写数据：accessor 接口提供自动索引的能力
  // 转置、expand，等操作只是改变了数据的索引方式，并没有改变
  // 内存种的数据排列，accessor 可以帮助我们自动计算索引获取；
  // 或者使用 data_ptr 手动计算也可，但需要获取 stride 信息。
  // 直接使用 tensor 索引的方式也可以达到同样的效果，但是这种
  // 方法返回的是一个右值 tensor 变量，accessor 是指针层面。
  torch::Tensor f = torch::tensor({{1,2,3}, {4,5,6}, {7,8,9}}).t();
  at::TensorAccessor<long, 2> f_accessor = f.accessor<long, 2>();
  std::cout << sizeof(f_accessor.data()) << std::endl; // 输出 8，表示指针的长度
  std::cout << sizeof(f_accessor[0][2]) << std::endl;  // 同上，表示 long 的长度
  // 下面打印的地址一致
  std::cout << f_accessor.data() << std::endl
            << &f_accessor[0][0] << std::endl
            << &f_accessor[0][1] << std::endl;
  // 连续地址输出结果，可以发现仍然是转置之前的顺序，
  // contiguous() 可以强行把内存顺序修改。
  for (int i = 0; i < 9; ++i){
    std::cout << *(f_accessor.data() + i) << " ";
    if (i == 8) {
      std::cout << std::endl;
    }
  }
  f_accessor[2][0] = 100;
  f[2][1] = 101; // &f[2][1] 会报错
  std::cout << f << std::endl;

  torch::Tensor g = torch::tensor({{3},{1}});
  torch::Tensor gg = g.expand({2, 2});
  auto g_accessor = g.accessor<long, 2>();
  auto gg_accessor = gg.accessor<long, 2>();
  // 下面打印的地址一致，都是数字 3 对应的地址。
  std::cout << g_accessor.data() << std::endl 
            << gg_accessor.data() << std::endl;
  std::cout << &g_accessor[0][0] << std::endl
            << &gg_accessor[0][1] << std::endl
            << &gg_accessor[0][1] << std::endl;

}
