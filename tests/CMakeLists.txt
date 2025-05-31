cmake_minimum_required(VERSION 3.12)

# 设置测试可执行文件
set(TEST_SOURCES
    unit/core/test_interfaces.cpp
    unit/core/test_config.cpp
    unit/core/test_memory_manager.cpp
    unit/core/test_exceptions.cpp
)

# 创建测试可执行文件
add_executable(deepsearch_tests
    ${TEST_SOURCES}
)

# 链接库
target_link_libraries(deepsearch_tests 
    PRIVATE 
    Catch2::Catch2WithMain
    deepsearch_core
)

# 设置包含目录
target_include_directories(deepsearch_tests 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

# 注册测试
include(CTest)
include(Catch)
catch_discover_tests(deepsearch_tests)
