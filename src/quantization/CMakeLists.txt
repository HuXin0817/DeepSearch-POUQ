# 量化模块
set(QUANTIZATION_SOURCES
        quantizer_factory.cpp
        fp32_quant.cpp
        sq8_quant.cpp
        sq4_quant.cpp
)

set(QUANTIZATION_HEADERS
        quantizer.h
        fp32_quant.h
        sq8_quant.h
        sq4_quant.h
)

add_library(deepsearch_quantization ${QUANTIZATION_SOURCES})

target_include_directories(deepsearch_quantization PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(deepsearch_quantization
        PUBLIC
        deepsearch_core
        deepsearch_simd
        deepsearch_distance
)

# C++ 标准
set_target_properties(deepsearch_quantization PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 编译选项
target_compile_options(deepsearch_quantization PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# 安装
install(TARGETS deepsearch_quantization
        EXPORT DeepSearchTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(FILES ${QUANTIZATION_HEADERS}
        DESTINATION include/quantization
)
