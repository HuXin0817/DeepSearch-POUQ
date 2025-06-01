#!/bin/bash

# Step 1: 下载 SIFT 数据集
wget -q -O siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz

# Step 2: 解压数据集
tar -xvzf siftsmall.tar.gz

# Step 3: 创建并进入 build 目录
mkdir -p build
cd build

# Step 4: 编译项目
cmake clean
cmake ..
cmake --build . --config Release

# Step 5: 执行程序
./DeepSearch ../siftsmall/siftsmall_base.fvecs ../siftsmall/siftsmall_query.fvecs ../siftsmall/siftsmall_groundtruth.ivecs graph.index 1 10 120 0
