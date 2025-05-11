# 创建 hnsw 库
add_library(hnsw_lib STATIC
        hnsw.cpp
)

# 公开头文件路径（允许父目录或其他模块访问）
target_include_directories(hnsw_lib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # 头文件路径指向 hnsw 目录
)
