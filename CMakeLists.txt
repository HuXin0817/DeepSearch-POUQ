cmake_minimum_required(VERSION 3.22)
project(DeepSearch)

set(CMAKE_CXX_STANDARD 17)

# 全局添加 src 目录到头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 添加子目录（会自动执行 src/hnsw/CMakeLists.txt）
add_subdirectory(src/hnsw)

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

# macOS 下尝试查找 OpenMP（非强制）
if (APPLE)
    # 设置 OpenMP 路径（Homebrew 安装的 libomp）
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

    find_package(OpenMP)  # 移除了 REQUIRED
    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP on macOS")
        target_link_libraries(DeepSearch PRIVATE OpenMP::OpenMP_CXX)
    else ()
        message(WARNING "OpenMP not found on macOS, continuing without it")
    endif ()
endif ()

# 链接平台特定库
if (UNIX AND NOT APPLE)
    target_link_libraries(DeepSearch PRIVATE rt)
endif ()

# 链接 hnsw 库
target_link_libraries(DeepSearch
        PRIVATE
        hnsw_lib
)
