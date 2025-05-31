# Distance computation module

set(DISTANCE_SOURCES
        computers.cpp
)

set(DISTANCE_HEADERS
        computers.h
)

# 创建距离计算库
add_library(deepsearch_distance ${DISTANCE_SOURCES})

# 设置包含目录
target_include_directories(deepsearch_distance PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

# 链接依赖
#target_link_libraries(deepsearch_distance)

# SIMD编译选项
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # 检查编译器支持
    include(CheckCXXCompilerFlag)

    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)

    if (COMPILER_SUPPORTS_AVX512)
        target_compile_options(deepsearch_distance PRIVATE -mavx512f -mavx512dq)
        target_compile_definitions(deepsearch_distance PRIVATE HAVE_AVX512=1)
    endif ()

    if (COMPILER_SUPPORTS_AVX2)
        target_compile_options(deepsearch_distance PRIVATE -mavx2 -mfma)
        target_compile_definitions(deepsearch_distance PRIVATE HAVE_AVX2=1)
    endif ()

    if (COMPILER_SUPPORTS_SSE41)
        target_compile_options(deepsearch_distance PRIVATE -msse4.1)
        target_compile_definitions(deepsearch_distance PRIVATE HAVE_SSE41=1)
    endif ()
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    # ARM NEON support - on ARM64, NEON is always available
    # No special compiler flags needed for Apple Silicon
    if (APPLE)
        # Apple Silicon (ARM64) - NEON is always available
        # ARM NEON支持
        target_compile_options(deepsearch_distance PRIVATE -mcpu=apple-m1)
        target_compile_definitions(deepsearch_distance PRIVATE HAVE_NEON=1)
    else ()
        # Other ARM platforms may need explicit NEON flags
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON_FPU)

        if (COMPILER_SUPPORTS_NEON_FPU)
            target_compile_options(deepsearch_distance PRIVATE -mfpu=neon)
        endif ()
        target_compile_definitions(deepsearch_distance PRIVATE HAVE_NEON=1)
    endif ()
endif ()

# 设置C++标准
set_target_properties(deepsearch_distance PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 安装头文件
install(FILES ${DISTANCE_HEADERS}
        DESTINATION include/deepsearch/distance
)

# 安装库
install(TARGETS deepsearch_distance
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
