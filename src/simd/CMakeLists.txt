add_library(deepsearch_simd
        simd_utils.h
        simd_utils.cpp
        distance_functions.h
        distance_functions.cpp
)

target_include_directories(deepsearch_simd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 创建一个函数来配置 SIMD 支持
function(configure_simd_support target_name)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # 检查编译器支持
        include(CheckCXXCompilerFlag)

        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)

        if (COMPILER_SUPPORTS_AVX512)
            target_compile_options(${target_name} PRIVATE -mavx512f -mavx512dq)
            target_compile_definitions(${target_name} PRIVATE HAVE_AVX512=1)
            message(STATUS "${target_name}: AVX512 support enabled")
        endif ()

        if (COMPILER_SUPPORTS_AVX2)
            target_compile_options(${target_name} PRIVATE -mavx2 -mfma)
            target_compile_definitions(${target_name} PRIVATE HAVE_AVX2=1)
            message(STATUS "${target_name}: AVX2 support enabled")
        endif ()

        if (COMPILER_SUPPORTS_SSE41)
            target_compile_options(${target_name} PRIVATE -msse4.1)
            target_compile_definitions(${target_name} PRIVATE HAVE_SSE41=1)
            message(STATUS "${target_name}: SSE4.1 support enabled")
        endif ()
        
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        # ARM NEON support
        if (APPLE)
            # Apple Silicon (ARM64) - NEON is always available
            target_compile_options(${target_name} PRIVATE -mcpu=apple-m1)
            target_compile_definitions(${target_name} PRIVATE HAVE_NEON=1)
            message(STATUS "${target_name}: Apple Silicon NEON support enabled")
        else ()
            # Other ARM platforms
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON_FPU)

            if (COMPILER_SUPPORTS_NEON_FPU)
                target_compile_options(${target_name} PRIVATE -mfpu=neon)
            endif ()
            target_compile_definitions(${target_name} PRIVATE HAVE_NEON=1)
            message(STATUS "${target_name}: ARM NEON support enabled")
        endif ()
    endif ()
endfunction()

# 为 simd 库配置 SIMD 支持
configure_simd_support(deepsearch_simd)

# 设置C++标准
set_target_properties(deepsearch_simd PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 导出 SIMD 配置函数供其他模块使用
set(SIMD_CONFIGURE_FUNCTION_AVAILABLE TRUE PARENT_SCOPE)
