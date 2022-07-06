/*
 * @File    :   TensorBasics.cpp
 * @Time    :   2022/07/06
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "include/TensorIndexSlice.h"
#include "include/TensorInit.h"
#include "include/TensorAttribute.h"
#include "include/TensorTransform.h"
#include "include/TensorCalculate.h"

int main() {
  tensor_init();
  tensor_index_slice();
  tensor_attribute();
  tensor_transform();
  tensor_calculate();

  return 0;
}