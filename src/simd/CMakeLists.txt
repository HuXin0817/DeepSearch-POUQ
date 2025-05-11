# 创建 quant 库
add_library(simdlib STATIC
        fp32_quant.cpp
        sq4_quant.cpp
        sq8_quant.cpp
)

# 公开头文件路径（允许父目录或其他模块访问）
target_include_directories(simdlib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # 头文件路径指向 quant 目录
)
