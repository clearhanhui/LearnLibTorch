#!/bin/bash
echo "Download from https://github.com/tkipf/pygcn/tree/master/data/cora "
mkdir cora
wget -q https://github.com/tkipf/pygcn/raw/master/data/cora/cora.cites -O cora/cora.cites
wget -q https://github.com/tkipf/pygcn/raw/master/data/cora/cora.content -O cora/cora.content
