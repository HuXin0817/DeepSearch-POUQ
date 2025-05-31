#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "core/exceptions.h"
#include "core/memory_manager.h"

using namespace deepsearch::core;

TEST_CASE("AlignedBuffer functionality", "[memory][aligned_buffer]") {
  SECTION("Basic operations") {
    AlignedBuffer<float> buffer(1000, 32);

    // 测试对齐
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data());
    REQUIRE(addr % 32 == 0);

    // 测试大小
    REQUIRE(buffer.size() == 1000);

    // 测试读写
    buffer[0] = 1.5f;
    buffer[999] = 2.5f;
    REQUIRE(buffer[0] == 1.5f);
    REQUIRE(buffer[999] == 2.5f);
  }

  SECTION("Iterators") {
    AlignedBuffer<int> buffer(10, 16);

    int value = 0;
    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
      *it = value++;
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
      REQUIRE(buffer[i] == static_cast<int>(i));
    }
  }

  SECTION("Move semantics") {
    AlignedBuffer<double> buffer1(100, 64);
    buffer1[0] = 3.14;
    buffer1[99] = 2.71;

    AlignedBuffer<double> buffer2(std::move(buffer1));
    REQUIRE(buffer2[0] == 3.14);
    REQUIRE(buffer2[99] == 2.71);
    REQUIRE(buffer2.size() == 100);
  }
}

TEST_CASE("MemoryPool functionality", "[memory][memory_pool]") {
  SECTION("Basic allocation and deallocation") {
    MemoryPool<int> pool(100);

    std::vector<int*> ptrs;
    for (int i = 0; i < 50; ++i) {
      int* ptr = pool.allocate();
      REQUIRE(ptr != nullptr);
      *ptr = i;
      ptrs.push_back(ptr);
    }

    for (size_t i = 0; i < ptrs.size(); ++i) {
      REQUIRE(*ptrs[i] == static_cast<int>(i));
    }

    for (int* ptr : ptrs) {
      pool.deallocate(ptr);
    }

    int* ptr = pool.allocate();
    REQUIRE(ptr != nullptr);
    pool.deallocate(ptr);
  }

  SECTION("Multiple chunks") {
    MemoryPool<float> pool(64);

    std::vector<float*> ptrs;
    for (int i = 0; i < 200; ++i) {
      float* ptr = pool.allocate();
      REQUIRE(ptr != nullptr);
      *ptr = static_cast<float>(i) * 1.5f;
      ptrs.push_back(ptr);
    }

    for (size_t i = 0; i < ptrs.size(); ++i) {
      REQUIRE(*ptrs[i] == static_cast<float>(i) * 1.5f);
    }

    for (float* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }
}

TEST_CASE("AlignedAllocator functionality", "[memory][aligned_allocator]") {
  SECTION("Basic allocation") {
    AlignedAllocator<float> allocator(32);

    float* ptr = allocator.allocate(100);
    REQUIRE(ptr != nullptr);

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    REQUIRE(addr % 32 == 0);

    ptr[0] = 1.0f;
    ptr[99] = 2.0f;
    REQUIRE(ptr[0] == 1.0f);
    REQUIRE(ptr[99] == 2.0f);

    allocator.deallocate(ptr, 100);
  }

  SECTION("Different alignments") {
    std::vector<size_t> alignments = {16, 32, 64, 128, 256};

    for (size_t alignment : alignments) {
      AlignedAllocator<double> allocator(alignment);
      double* ptr = allocator.allocate(50);
      REQUIRE(ptr != nullptr);

      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
      REQUIRE(addr % alignment == 0);

      allocator.deallocate(ptr, 50);
    }
  }

  SECTION("Equality comparison") {
    AlignedAllocator<int> alloc1(64);
    AlignedAllocator<int> alloc2(64);
    AlignedAllocator<int> alloc3(32);

    REQUIRE(alloc1 == alloc2);
    REQUIRE_FALSE(alloc1 == alloc3);
    REQUIRE(alloc1 != alloc3);
    REQUIRE_FALSE(alloc1 != alloc2);
  }
}
