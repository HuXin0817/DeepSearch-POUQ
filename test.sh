#!/bin/bash

# Step 1: 下载 SIFT 数据集
wget -O sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

# Step 2: 解压数据集
tar -xvzf sift.tar.gz

# Step 3: 创建并进入 build 目录
mkdir -p build
cd build

# Step 4: 编译项目
cmake ..
cmake --build . --config Release

# Step 5: 执行程序
./DeepSearch ../sift/sift_base.fvecs ../sift/sift_query.fvecs ../sift/sift_groundtruth.ivecs graph.index 1 10 120 0
