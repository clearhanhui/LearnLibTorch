#!/bin/bash
echo "Download from http://yann.lecun.com/exdb/mnist"
mkdir -p MNIST/raw
wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O MNIST/raw/train-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O MNIST/raw/train-labels-idx1-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O MNIST/raw/t10k-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O MNIST/raw/t10k-labels-idx1-ubyte.gz
gzip -dfk MNIST/raw/train-images-idx3-ubyte.gz
gzip -dfk MNIST/raw/train-labels-idx1-ubyte.gz
gzip -dfk MNIST/raw/t10k-images-idx3-ubyte.gz
gzip -dfk MNIST/raw/t10k-labels-idx1-ubyte.gz
