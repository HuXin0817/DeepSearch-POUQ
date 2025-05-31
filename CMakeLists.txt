cmake_minimum_required(VERSION 3.15)
project(DeepSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0  # 使用最新稳定版本
)
FetchContent_MakeAvailable(Catch2)

# 全局添加 src 目录到头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 添加子目录
add_subdirectory(src/core)
add_subdirectory(src/hnsw)
add_subdirectory(src/distance)
add_subdirectory(tests)

# 设置通用编译选项（所有平台共用）
set(COMMON_FLAGS "-Wall -Wextra -O3 -fpic -fexceptions -fno-rtti")

# 平台特定配置
if (APPLE)
    # macOS 配置（移除 -fopenmp 选项）
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
else ()
    # Linux 配置（保留 OpenMP 支持）
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -lrt -fopenmp -ftree-vectorize")

    # 仅 x86 架构添加 -march=native
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|i[3-6]86)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
endif ()

# 可执行文件配置
add_executable(DeepSearch
        src/main.cpp
        src/neighbor.cpp
)

# 链接平台特定库
if (UNIX AND NOT APPLE)
    target_link_libraries(DeepSearch PRIVATE rt)
endif ()

# 链接 hnsw 库
target_link_libraries(DeepSearch
        PRIVATE
        hnsw_lib
)

# 启用测试
enable_testing()
