#!/bin/bash
echo "Download from https://github.com/tkipf/pygcn/tree/master/data/cora "
mkdir cora
wget https://github.com/tkipf/pygcn/raw/master/data/cora/cora.cites -O cora/cora.cites
wget https://github.com/tkipf/pygcn/raw/master/data/cora/cora.content -O cora/cora.content
